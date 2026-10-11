# -*- coding: utf-8 -*-
r"""cn373 在 141 缺口种上的真实精度——兜底能力的首次实测。

背景：cn373 的验收成绩 59.25% 全部来自五区 228 种（自训模型主场）；
它留在系统里的唯一理由是缺口种兜底，但 merged val 里缺口种零样本、从未评过。
现在 gap 补抓攒下了真值图，直接测。结论决定 cn373 去留：
  精度尚可 → 值得当按需/云端兜底；精度也烂 → cn373 可整体退休，自训模型即完全体。

Usage: python eval_cn373_gap.py [--per 6] [--cap 800]
"""
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from torchvision import transforms as T

GAP_CSV = r"E:\code\code tools\data\national_gap\dataset.csv"
GAP_IMG = r"E:\code\code tools\data\national_gap\images"
CN373_TFLITE = r"E:\code\observebird\android\app\src\main\assets\models\birdnet_373_float32.tflite"
TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    per = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 6
    cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 800

    # 类下标 = labels_birdnet373.txt 行号（gap 表的 birdnet_index 列是空的）
    sp2idx = {}
    for i, line in enumerate(open(
            r"E:\code\observebird\android\app\src\main\assets\models\labels_birdnet373.txt",
            encoding="utf-8")):
        sp2idx[line.split("\t")[0].strip()] = i

    rows = list(csv.DictReader(open(GAP_CSV, encoding="utf-8")))
    by_sp = defaultdict(list)
    for r in rows:
        idx = sp2idx.get(r["scientific_name"], -1)
        p = os.path.join(GAP_IMG, r["file"])
        if 0 <= idx < 373 and os.path.exists(p):
            by_sp[r["scientific_name"]].append((idx, p))
    rng = random.Random(7)
    sample = []
    for s in sorted(by_sp):
        rs = by_sp[s][:]
        rng.shuffle(rs)
        sample += [(s, i, p) for i, p in rs[:per]]
    rng.shuffle(sample)
    sample = sample[:cap]
    print(f"缺口种可评 {len(by_sp)} 种, 抽样 {len(sample)} 张 (每类上限 {per})")

    it = Interpreter(model_path=CN373_TFLITE, num_threads=4)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    top1 = top3 = top5 = n = 0
    per_sp = defaultdict(lambda: [0, 0])  # 种 -> [对, 总]
    for s, gt, p in sample:
        img = Image.open(p).convert("RGB")
        x = (TF(img).numpy().transpose(1, 2, 0)[None]).astype(np.float32)
        it.set_tensor(inp["index"], x)
        it.invoke()
        logits = it.get_tensor(out["index"])[0][:373]
        rank = np.argsort(logits)[::-1]
        top1 += int(rank[0] == gt)
        top3 += int(gt in rank[:3])
        top5 += int(gt in rank[:5])
        per_sp[s][0] += int(rank[0] == gt)
        per_sp[s][1] += 1
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(sample)}  top1 {top1/n*100:.1f}%", flush=True)

    print(f"\ncn373 缺口种实测: top1 {top1/n*100:.2f}%  top3 {top3/n*100:.2f}%  "
          f"top5 {top5/n*100:.2f}%  (n={n}, {len(per_sp)} 种)")
    good = [(s, a, b) for s, (a, b) in per_sp.items() if b >= 4]
    good.sort(key=lambda t: -t[2])
    g = [(s, a, b) for s, a, b in good if a > 0]
    z = [(s, a, b) for s, a, b in good if a == 0]
    print(f"抽样>=4张的种里: 有对 {len(g)} 种 / 全错 {len(z)} 种")
    print("全错代表:", ", ".join(f"{s}({a}/{b})" for s, a, b in z[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

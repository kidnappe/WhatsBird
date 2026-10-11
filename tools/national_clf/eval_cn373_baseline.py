# -*- coding: utf-8 -*-
r"""给预装兜底 cn373 补一个我们自己口径的成绩（全国 merged val 上评测）。

用途：全国模型（训练中）的验收基线——同一条 val 划分（grouped_split seed=0 val 20%），
同批图，cn373 vs 新全国模型才有可比性。原作者 README 的 62.16% 是他们自己的测试集口径，
与本口径不可比；且 cn373 无逐轮训练日志（只有 Training_log.png 图片），监控页曲线无法
提供——产出为单点 run（models/cn373_baseline/history.json），进历史对比页。

类别映射：五区 dataset.csv 的 birdnet_index 列 = 该种在 cn373 384 维输出里的类下标
（有效类前 373）。cn373 类表没有的种跳过（这正是它覆盖不了的部分）。

Usage: python eval_cn373_baseline.py [--cap 2500]
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

sys.path.insert(0, r"E:\code\observebird\tools")
from train import grouped_split  # noqa: E402  与正在跑的全国训练同一划分函数/种子

NAT_CSV = r"E:\code\code tools\data\national\dataset.csv"
CN373_TFLITE = r"E:\code\observebird\android\app\src\main\assets\models\birdnet_373_float32.tflite"
OUT_DIR = r"E:\code\code tools\models\cn373_baseline"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 2500

    # 与全国训练完全一致的 val 划分
    rows = list(csv.DictReader(open(NAT_CSV, encoding="utf-8")))
    _, va = grouped_split(rows, val_frac=0.2, seed=0)
    print(f"全国 merged val: {len(va)} 张")

    # 物种 → cn373 类下标（取自五区 dataset 的 birdnet_index 列）
    sp2idx = {}
    for region in ["huabei", "huazhong", "huanan_sw", "huadong", "dongbei"]:
        src = os.path.join(DATA, region, "dataset.csv") if (DATA := r"E:\code\code tools\data") else ""
        for r in csv.DictReader(open(src, encoding="utf-8")):
            s = r["scientific_name"]
            if s not in sp2idx:
                try:
                    sp2idx[s] = int(r["birdnet_index"])
                except (ValueError, KeyError):
                    sp2idx[s] = -1

    # val 里 cn373 认识的种，分层抽样
    by_sp = defaultdict(list)
    for r in va:
        if sp2idx.get(r["scientific_name"], -1) >= 0:
            by_sp[r["scientific_name"]].append(r)
    rng = random.Random(3)
    sample = []
    per = max(1, cap // max(1, len(by_sp)))
    for s in sorted(by_sp):
        rs = by_sp[s]
        rng.shuffle(rs)
        sample += [(s, r) for r in rs[:per]]
    print(f"可评物种 {len(by_sp)}/244，抽样 {len(sample)} 张")

    it = Interpreter(model_path=CN373_TFLITE, num_threads=4)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]

    top1 = top3 = top5 = n = 0
    for s, r in sample:
        p = os.path.normpath(r["file"])
        img = Image.open(p).convert("RGB")
        x = (TF(img).numpy().transpose(1, 2, 0)[None]).astype(np.float32)
        it.set_tensor(inp["index"], x)
        it.invoke()
        logits = it.get_tensor(out["index"])[0][:373]
        rank = np.argsort(logits)[::-1]
        gt = sp2idx[s]
        top1 += int(rank[0] == gt)
        top3 += int(gt in rank[:3])
        top5 += int(gt in rank[:5])
        n += 1
        if n % 500 == 0:
            print(f"  {n}/{len(sample)}", flush=True)

    acc = (top1 / n, top3 / n, top5 / n)
    print(f"cn373 基线（我们口径，全国 merged val）: top1 {acc[0]*100:.2f}%  "
          f"top3 {acc[1]*100:.2f}%  top5 {acc[2]*100:.2f}%  (n={n})")

    os.makedirs(OUT_DIR, exist_ok=True)
    import json
    hist = {
        "args": {"model": "cn373_placeholder(BirdNet ConvNeXt 11.2M, iNat NC 数据)",
                 "note": "原作者未留逐轮日志——单点补评，无训练曲线可画；"
                         "口径=全国 merged val(grouped_split seed=0 val 20%)"},
        "species": sorted(sp2idx),
        "history": [{"epoch": 1, "top1": acc[0], "top3": acc[1], "top5": acc[2],
                     "note": "单点补评（非训练曲线）"}],
        "best_top1": acc[0], "params": 11200000,
        "n_images": n, "n_species": len(by_sp),
    }
    with open(os.path.join(OUT_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)
    print(f"→ {OUT_DIR}\\history.json（历史对比页将扫出 cn373_baseline 条目）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

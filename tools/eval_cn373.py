"""全国 373 类模型（birdnet_373_float32.tflite）本地复测。

评测集：
  A) inat_eval/ 500 图（50 种）——真值取文件名里的学名
  B) 三大区 dataset.csv 分层抽样（每种 min(20, n)）——真值取 csv 的 scientific_name

预处理与部署逐位一致（TfLiteClassifier.kt）：Resize256 -> CenterCrop224 -> ImageNet 归一化。
模型输出 [1,384]，有效类 373（CN373_TENSOR_DIM=384）。
"""
import csv
import glob
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ASSET = r"E:/code/observebird/android/app/src/main/assets/models/birdnet_373_float32.tflite"
LABELS = r"E:/code/observebird/android/app/src/main/assets/models/labels_birdnet373.txt"
INAT_EVAL = r"E:/code/code tools/data/inat_eval"
DATA = r"E:/code/code tools/data"
TENSOR_DIM = 384

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x  # HWC


def main():
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=ASSET, num_threads=4)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    print(f"模型 {os.path.basename(ASSET)}  输入{inp['shape']} 输出{outp['shape']}")

    labels = [l.split("\t")[0].strip() for l in
              open(LABELS, encoding="utf-8").read().splitlines() if l.strip()]
    sci2idx = {s: i for i, s in enumerate(labels)}

    def run_set(name, items, per_region=None):
        """items: list[(path, sci)]"""
        n = top1 = top3 = top5 = 0
        skipped = Counter()
        per = Counter() if per_region is None else per_region
        conf = []
        t0 = time.time()
        for k, (path, sci) in enumerate(items):
            gi = sci2idx.get(sci)
            if gi is None:
                skipped[sci] += 1
                continue
            try:
                x = preprocess(path)
            except Exception:
                skipped["<decode失败>"] += 1
                continue
            it.set_tensor(inp["index"], x[None, ...])
            it.invoke()
            logits = it.get_tensor(outp["index"])[0][:len(labels)].copy()
            order = np.argsort(-logits)[:5]
            r = int(np.where(order == gi)[0][0]) if gi in order else 9
            n += 1
            top1 += r == 0
            top3 += r < 3
            top5 += r < 5
            conf.append(float(1.0 / (1.0 + np.exp(-logits[order[0]]))))
            per[(name, "n")] = per.get((name, "n"), 0) + 1
            for rk, cnt in (("t1", r == 0), ("t3", r < 3), ("t5", r < 5)):
                key = (name if per_region is None else path.split("data")[1].split("/")[1] if "/data/" in path.replace("\\", "/") else name, rk)
                per[key] = per.get(key, 0) + int(cnt)
            if (k + 1) % 500 == 0:
                print(f"  {name} {k+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
        print(f"\n[{name}] 有效 {n} 图（不在373类表跳过 {sum(skipped.values())}）")
        if n:
            print(f"  top1 {top1/n*100:.2f}%  top3 {top3/n*100:.2f}%  top5 {top5/n*100:.2f}%  "
                  f"平均置信 {np.mean(conf):.2f}  用时 {time.time()-t0:.0f}s")
        return per

    # A) inat_eval
    items_a = []
    for f in sorted(glob.glob(os.path.join(INAT_EVAL, "*.jpg"))):
        m = re.match(r"\d+_(.+?)_\d+\.jpg$", os.path.basename(f), re.I)
        if m:
            items_a.append((f, m.group(1).replace("_", " ")))
    run_set("inat_eval(50种500图)", items_a)

    # B) 三大区分层抽样
    import random
    random.seed(0)
    for region in ["huabei", "huazhong", "huanan_sw"]:
        rows = list(csv.DictReader(
            open(os.path.join(DATA, region, "dataset.csv"), encoding="utf-8")))
        by_sp = {}
        for r in rows:
            by_sp.setdefault(r["scientific_name"], []).append(r["file"])
        items = []
        for sci, files in sorted(by_sp.items()):
            for f in random.sample(files, min(20, len(files))):
                p = f if os.path.isabs(f) else os.path.join(DATA, region, "images", f)
                items.append((p, sci))
        run_set(f"{region}抽样", items)


if __name__ == "__main__":
    main()

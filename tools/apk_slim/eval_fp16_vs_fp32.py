"""fp16 瘦身决策器：把 7 个 BirdNET 分类器换成 fp16 权重前，先证明「不掉点」。

背景（2026-10-13）：蓝奏云单文件上限 100MB，GitHub 下载太慢，release APK
118.86MB 必须瘦身。拆开看 141.22MB 未压缩内容里 assets/models 占 102.8MB，
其中 7 个分类器（全国 + 五区 + 拒识门）全是 float32，合计 71.7MB；
每个导出目录里**早就躺着同源的 model_float16.tflite**（体积对半），只是从没装进 App。

本脚本做三件事，任一步不过就别换：
  1. 用 `fp16_io32.py` 把 fp16 张量 IO 的那份，补成 fp32 IO（端上零改动）；
  2. 用 App 现在的 fp32 资产当基准，逐图对比 top-1 准确率 / 一致率 / top-5 重合；
  3. 打印对照表，供人工拍板。

口径与端上 `TfLiteClassifier` 逐位一致（短边 256 → 居中裁 224 → ImageNet 归一化），
类别顺序取 `assets/models/labels_*.txt`（行序即类别索引）。
fp32 基准读 `android/models/fp32/`（2026-10-13 起 App assets 里只剩 fp16）。
val 集用各区 `frozen_val.csv`——就是当初记录「华北 rec 60.21%」那一份，
所以 fp32 侧必须复现出文档里的量级；复现不出来说明口径错了，先修口径再谈 fp16。

用法：
  python tools/apk_slim/eval_fp16_vs_fp32.py --n 1200
  python tools/apk_slim/eval_fp16_vs_fp32.py --n 300 --only huabei national
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp16_io32 import convert as fp16_io32_convert  # noqa: E402

ROOT = r"E:\code\observebird"
ASSETS = os.path.join(ROOT, r"android\app\src\main\assets\models")
# fp32 基准：2026-10-13 起 App assets 里只有 fp16 了，fp32 原件存档在 android/models/fp32/
FP32_DIR = os.path.join(ROOT, r"android\models\fp32")
EXPORTS = r"E:\code\code tools\models"
DATA = r"E:\code\code tools\data"
STAGE = os.path.join(ROOT, r"dist\apk_slim\staging")

INPUT = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# key -> (App 现役 fp32 资产, fp16 源(相对 EXPORTS), val csv(相对 DATA), 图片根目录, labels 资产)
REGIONS = {
    "huabei":    ("birdnet_huabei.tflite",    "huabei_mnv4_rec",
                  r"huabei\frozen_val.csv",    r"huabei\images",    "labels_huabei.txt"),
    "huazhong":  ("birdnet_huazhong.tflite",  "huazhong_mnv4_25k",
                  r"huazhong\frozen_val.csv",  r"huazhong\images",  "labels_huazhong.txt"),
    "huadong":   ("birdnet_huadong.tflite",   "huadong_mnv4_300_rec",
                  r"huadong\frozen_val.csv",   r"huadong\images",   "labels_huadong.txt"),
    "huanan_sw": ("birdnet_huanan_sw.tflite", "huanan_sw_mnv4_rec",
                  r"huanan_sw\frozen_val.csv", r"huanan_sw\images", "labels_huanan_sw.txt"),
    "dongbei":   ("birdnet_dongbei.tflite",   "dongbei_mnv4_rec",
                  r"dongbei\frozen_val.csv",   r"dongbei\images",   "labels_dongbei.txt"),
    "national":  ("birdnet_national_v2.tflite", "national_mnv4_v1",
                  r"national_eval244\dataset.csv", None,            "labels_national_v2.txt"),
    "reject":    ("birdnet_reject.tflite",    "reject_mnv4_v1",
                  r"reject_clf\dataset.csv",   r"reject_clf\images", "labels_reject.txt"),
}


def load_labels(fname):
    out = []
    with open(os.path.join(ASSETS, fname), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(line.split("\t")[0].strip())
    return out


def preprocess(path):
    """与 TfLiteClassifier.centerCrop224 + fillInput 同口径。"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = 256.0 / min(w, h)
    im = im.resize((max(round(w * scale), INPUT), max(round(h * scale), INPUT)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - INPUT) // 2, (h - INPUT) // 2
    im = im.crop((l, t, l + INPUT, t + INPUT))
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a[None])          # NHWC，与 tflite 输入张量同布局


def make_interp(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]


def run_all(it, inp, out, batch):
    res = []
    for x in batch:
        it.set_tensor(inp["index"], x)
        it.invoke()
        res.append(it.get_tensor(out["index"])[0].copy())
    return np.stack(res)


def build_candidate(key, asset32):
    """fp16 源 -> 补 CAST -> 暂存区。返回候选路径。"""
    _, src_dir, _, _, _ = REGIONS[key]
    src = os.path.join(EXPORTS, src_dir, "export", "tflite", "model_float16.tflite")
    if not os.path.exists(src):
        sys.exit(f"{key}: 找不到 fp16 源 {src}")
    os.makedirs(STAGE, exist_ok=True)
    dst = os.path.join(STAGE, asset32)
    b, c = fp16_io32_convert(src, dst)
    return dst, b, c


def sample(key, n, seed):
    _, _, csv_rel, img_root, lab = REGIONS[key]
    species = load_labels(lab)
    idx = {s: i for i, s in enumerate(species)}
    csvp = os.path.join(DATA, csv_rel)
    root = os.path.join(DATA, img_root) if img_root else ""
    rows = []
    with open(csvp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sn = (r.get("scientific_name") or "").strip()
            if sn in idx:
                p = r["file"]
                p = p if os.path.isabs(p) else os.path.join(root, p)
                rows.append((p, idx[sn]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = [r for r in rows if os.path.exists(r[0])][:n]
    return [(preprocess(p), y) for p, y in rows]


def eval_pair(tag, fp32, fp16, samples):
    it32, i32, o32 = make_interp(fp32)
    it16, i16, o16 = make_interp(fp16)
    assert i32["dtype"] is np.float32 == i16["dtype"], f"{tag}: 输入 dtype 不是 float32"
    assert o32["dtype"] is np.float32 == o16["dtype"], f"{tag}: 输出 dtype 不是 float32"
    assert list(i32["shape"]) == list(i16["shape"]), f"{tag}: 输入形状不一致"
    assert list(o32["shape"]) == list(o16["shape"]), f"{tag}: 输出形状不一致"

    y = np.array([s[1] for s in samples])
    xb = [s[0] for s in samples]
    a = run_all(it32, i32, o32, xb)
    b = run_all(it16, i16, o16, xb)

    t32, t16 = a.argmax(1), b.argmax(1)
    acc32, acc16 = float((t32 == y).mean()), float((t16 == y).mean())
    same = float((t32 == t16).mean())
    k = min(5, a.shape[1])
    t5 = float(np.mean([set(np.argsort(-r)[:k]) == set(np.argsort(-r2)[:k])
                        for r, r2 in zip(a, b)]))
    print(f"{tag:9s} n={len(y):5d}  fp32={acc32*100:6.2f}%  fp16={acc16*100:6.2f}%  "
          f"Δ={(acc16-acc32)*100:+5.2f}pp  top1一致={same*100:6.2f}%  "
          f"top5全同={t5*100:6.2f}%  max|Δlogit|={np.abs(a-b).max():.4f}")
    return acc32, acc16, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="每个模型抽图数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--build-only", action="store_true", help="只生成候选，不跑对比")
    a = ap.parse_args()
    keys = [k for k in REGIONS if not a.only or k in a.only]

    print(f"fp16 瘦身对照  n/模型={a.n}  seed={a.seed}")
    print(f"{'模型':9s} {'fp32':>9s} {'fp16':>9s} {'省下':>9s}")
    tot32 = tot16 = 0
    for k in keys:
        asset32 = REGIONS[k][0]
        p32 = os.path.join(FP32_DIR, asset32)
        cand, _, c = build_candidate(k, asset32)
        b = os.path.getsize(p32)
        tot32 += b
        tot16 += c
        print(f"{k:9s} {b/1048576:8.2f}M {c/1048576:8.2f}M {(b-c)/1048576:8.2f}M  -> {cand}")
    print(f"{'合计':9s} {tot32/1048576:8.2f}M {tot16/1048576:8.2f}M {(tot32-tot16)/1048576:8.2f}M")
    if a.build_only:
        return
    print()
    for k in keys:
        asset32 = REGIONS[k][0]
        cand = os.path.join(STAGE, asset32)
        samples = sample(k, a.n, a.seed)
        if not samples:
            print(f"{k:9s} 跳过：没取到图")
            continue
        eval_pair(k, os.path.join(FP32_DIR, asset32), cand, samples)


if __name__ == "__main__":
    main()
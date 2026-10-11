"""Quantise the BirdNet classifier and measure the real accuracy cost.

Path: PyTorch -> ONNX (fp32) -> [quantise] -> ONNX -> TFLite
Quantisers used (both from onnxruntime.quantization):
  * quantize_dynamic  — weights -> int8, activations stay fp32, NO calibration
  * quantize_static   — full int8, needs representative images (labels unused)

Accuracy is measured on a held-out split of real iNaturalist photos, so the
top-1/top-5 drop is a number rather than a guess.

Usage: python quantize_eval.py
"""
import csv
import os
import random

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

TOOLS = r"E:\code\code tools"
DATA = os.path.join(TOOLS, "data", "inat_eval")
REPO = os.path.join(TOOLS, "bird-cn-373", "code", "Bird-Classification-main")
EXPORT = os.path.join(TOOLS, "bird-cn-373", "export")
QDIR = os.path.join(EXPORT, "quant")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path):
    """Resize short side 256 -> CenterCrop 224 -> ImageNet normalise (NCHW)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) > 256:
        s = 256 / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))


def load_species_index():
    with open(os.path.join(REPO, "species_list.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["scientific_name"]: i for i, r in enumerate(rows)}


def load_split(seed=0, calib_frac=0.5):
    """Per-species split so calibration and eval both stay class-diverse."""
    with open(os.path.join(DATA, "labels.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if os.path.exists(os.path.join(DATA, r["file"]))]
    by = {}
    for r in rows:
        by.setdefault(r["scientific_name"], []).append(r)
    rng = random.Random(seed)
    calib, ev = [], []
    for sci, rs in by.items():
        rs = rs[:]
        rng.shuffle(rs)
        k = max(1, int(len(rs) * calib_frac))
        calib += rs[:k]
        ev += rs[k:] or rs[:1]
    return calib, ev


def infer_ort(path, batch):
    """Run an ONNX model one image at a time (export has a fixed batch dim)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 8
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    outs = []
    for x in batch:
        outs.append(s.run(None, {name: np.expand_dims(x, 0).astype(np.float32)})[0][0])
    return np.stack(outs)


def metrics(logits, truth, ks=(1, 3, 5)):
    order = np.argsort(-logits, axis=1)
    return {f"top{k}": sum(truth[i] in order[i, :k] for i in range(len(truth)))
            / max(1, len(truth)) for k in ks}


def report(name, logits, truth, mb, base=None):
    m = metrics(logits, truth)
    delta = ""
    if base:
        delta = (f"   Δtop1 {(m['top1']-base['top1'])*100:+5.2f}"
                 f"  Δtop5 {(m['top5']-base['top5'])*100:+5.2f}")
    print(f"  {name:<26} {mb:7.2f} MB  top1={m['top1']*100:5.2f}%  "
          f"top3={m['top3']*100:5.2f}%  top5={m['top5']*100:5.2f}%{delta}")
    return m


def main():
    import onnxruntime.quantization as q

    os.makedirs(QDIR, exist_ok=True)
    idx_map = load_species_index()
    calib_rows, eval_rows = load_split()
    n_species = len({r["scientific_name"] for r in calib_rows + eval_rows})
    print(f"校准集 {len(calib_rows)} 张 / 验证集 {len(eval_rows)} 张 / {n_species} 种\n")

    calib_x = [preprocess(os.path.join(DATA, r["file"])) for r in calib_rows]
    eval_x = [preprocess(os.path.join(DATA, r["file"])) for r in eval_rows]
    eval_y = np.array([idx_map[r["scientific_name"]] for r in eval_rows])

    fp32 = os.path.join(EXPORT, "birdnet_373.onnx")
    print("=== 基线 (ONNX fp32) ===")
    base_logits = infer_ort(fp32, eval_x)
    base = report("ONNX fp32", base_logits, eval_y,
                  os.path.getsize(fp32) / 1048576)

    # ---- per-species error breakdown, before quantisation ----
    order = np.argsort(-base_logits, axis=1)
    with open(os.path.join(REPO, "species_list.csv"), encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    names = [r["common_name"] for r in all_rows]
    sci_of = [r["scientific_name"] for r in all_rows]
    by_species = {}
    for i, r in enumerate(eval_rows):
        by_species.setdefault(r["scientific_name"], []).append(i)

    print("\n=== 逐种准确率（FP32，按准确率升序）===")
    print(f"  {'scientific_name':<26}{'中文':<10}{'n':>3}{'top1':>7}{'top5':>7}   典型 top1 预测")
    rowsout = []
    for sci, idxs in by_species.items():
        acc1 = float(np.mean([eval_y[i] in order[i, :1] for i in idxs]))
        acc5 = float(np.mean([eval_y[i] in order[i, :5] for i in idxs]))
        zh = next(r["common_name"] for r in eval_rows if r["scientific_name"] == sci)
        t1 = names[order[idxs[0]][0]] if order[idxs[0]][0] < len(names) else "?"
        rowsout.append((acc1, acc5, sci, zh, len(idxs), t1))
    for acc1, acc5, sci, zh, n, t1 in sorted(rowsout):
        print(f"  {sci:<26}{zh:<10}{n:>3}{acc1*100:6.0f}%{acc5*100:6.0f}%   {t1}")
    perfect = [r for r in rowsout if r[0] >= 0.99]
    zero = [r for r in rowsout if r[0] <= 0.01]
    print(f"\n  全对的种: {len(perfect)}/{len(rowsout)}   全错的种: {len(zero)}/{len(rowsout)}")
    if zero:
        print("  全错的种（预测成了什么）:")
        for acc1, acc5, sci, zh, n, t1 in zero:
            print(f"     {sci:<26}{zh:<10} -> {t1}")

    # ---- dynamic range: no calibration data needed ----
    print("\n=== 动态范围量化（权重 int8 / 激活 fp32，不需要校准数据）===")
    dyn = os.path.join(QDIR, "birdnet_dynamic_int8.onnx")
    try:
        q.quantize_dynamic(fp32, dyn, weight_type=q.QuantType.QInt8)
        report("dynamic int8", infer_ort(dyn, eval_x), eval_y,
               os.path.getsize(dyn) / 1048576, base)
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")

    # ---- full static int8: needs calibration images ----
    print("\n=== 全静态 int8（用校准集）===")

    class Calib:
        """Yields NCHW float32 batches; the ONNX input is [1,3,224,224]."""

        def __init__(self, data):
            self.data = [np.expand_dims(d, 0).astype(np.float32) for d in data]

        def get_next(self):
            return {"input": self.data.pop(0)} if self.data else None

        def __len__(self):
            return len(self.data)

    stat = os.path.join(QDIR, "birdnet_static_int8.onnx")
    try:
        q.quantize_static(
            fp32, stat, Calib(calib_x),
            quant_format=q.QuantFormat.QDQ,
            per_channel=True,
            activation_type=q.QuantType.QUInt8,
            weight_type=q.QuantType.QInt8,
        )
        report("static int8 (QDQ)", infer_ort(stat, eval_x), eval_y,
               os.path.getsize(stat) / 1048576, base)
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")



    print("\n产物目录:", QDIR)


if __name__ == "__main__":
    main()

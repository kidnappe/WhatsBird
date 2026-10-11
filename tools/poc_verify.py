"""ObserveBird PoC: verify ConvNeXt -> TFLite conversion + geomodel reranking.

Runs the same input through PyTorch / ONNX Runtime / TFLite and compares
numerically, then applies the BirdNET geomodel (Apache 2.0) spatial-temporal
prior and reports Top-5 before/after reranking.
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

TOOLS = r"E:\code\code tools"
REPO = os.path.join(TOOLS, "bird-cn-373", "code", "Bird-Classification-main")
EXPORT = os.path.join(TOOLS, "bird-cn-373", "export")
GEOMODEL = os.path.join(TOOLS, "geomodel")
WEIGHTS = os.path.join(TOOLS, "bird-cn-373", "model_v1base.pth")

sys.path.insert(0, REPO)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path):
    """Resize(256) -> CenterCrop(224) -> [0,1] -> ImageNet normalize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def load_species():
    with open(os.path.join(REPO, "species_list.csv"), encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    return [(r[0], r[1]) for r in rows]


def load_geomodel_index(species):
    """Map each of our species to its row index in the geomodel output."""
    lab = os.path.join(
        GEOMODEL, "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt")
    idx = {}
    with open(lab, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            parts = line.split("\t")
            if len(parts) >= 2:
                idx.setdefault(parts[1].strip().lower(), i)
    out, miss = [], []
    for sci, zh in species:
        j = idx.get(sci.lower())
        (out if j is not None else miss).append((sci, zh, j))
    return out, miss


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO, "birdData", "Ps_245167185.png")
    lat = float(sys.argv[2]) if len(sys.argv) > 2 else 39.9042
    lon = float(sys.argv[3]) if len(sys.argv) > 3 else 116.4074
    week = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    print(f"image={image}\nlat={lat} lon={lon} week={week}\n")
    x = preprocess(image)
    print("input tensor:", x.shape, x.dtype, f"range[{x.min():.2f},{x.max():.2f}]")

    species = load_species()
    print(f"species: {len(species)}")

    # ---------- 1. PyTorch ----------
    import torch
    from _BirdNet import getModel
    net = getModel("v1base", load_weight=WEIGHTS).eval()
    with torch.no_grad():
        pt = net(torch.from_numpy(x)).numpy()[0]

    # ---------- 2. ONNX Runtime ----------
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        os.path.join(EXPORT, "birdnet_373.onnx"), so,
        providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]

    # ---------- 3. TFLite ----------
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(
        model_path=os.path.join(EXPORT, "tflite", "birdnet_373_float32.tflite"))
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    # ONNX is NCHW; onnx2tf emits NHWC. Feed whatever the tensor declares.
    shape = tuple(int(v) for v in inp["shape"])
    nhwc = len(shape) == 4 and shape[-1] == 3 and shape[1] != 3
    feed = np.transpose(x, (0, 2, 3, 1)).copy() if nhwc else x
    print(f"\nTFLite 输入 {shape} = {'NHWC' if nhwc else 'NCHW'} "
          f"| 喂入 {feed.shape}")
    interp.set_tensor(inp["index"], feed)
    interp.invoke()
    tflite_out = interp.get_tensor(outp["index"])[0].copy()

    # ---------- 数值对齐 ----------
    print("\n=== 三方数值对齐 ===")
    for name, v in (("ONNX", onnx_out), ("TFLite", tflite_out)):
        diff = np.abs(v - pt)
        agree = int(np.argmax(v)) == int(np.argmax(pt))
        print(f"  PyTorch vs {name:<6} max|diff|={diff.max():.3e}  "
              f"argmax一致={agree}  cos={float(v @ pt / (np.linalg.norm(v) * np.linalg.norm(pt))):.6f}")

    logits = tflite_out.astype(np.float64)
    z = logits - logits.max()
    probs = np.exp(z) / np.exp(z).sum()

    def topk(p, k=5):
        idx = np.argsort(-p)[:k]
        return [(int(i), float(p[i])) for i in idx]

    def name(i):
        return f"{species[i][1]}({species[i][0]})" if i < len(species) else "Unused"

    print("\n=== 重排前 Top-5 (纯图像模型) ===")
    for i, p in topk(probs):
        print(f"  {name(i):<48} {p*100:6.2f}%")

    # ---------- geomodel 重排 ----------
# NOTE: the official geomodel .tflite needs select-TF-ops (FlexErf), which is a
# problem on Android. The .onnx build runs on plain ONNX Runtime, so we use that.
    import onnxruntime as ort
    gm_path = os.path.join(
        GEOMODEL, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx")
    go = ort.SessionOptions()
    go.log_severity_level = 3
    gs = ort.InferenceSession(gm_path, go, providers=["CPUExecutionProvider"])
    gin, gout = gs.get_inputs()[0], gs.get_outputs()[0]
    print("\n=== geomodel 规格 (ONNX Runtime) ===")
    print("  input :", gin.name, gin.shape, gin.type)
    print("  output:", gout.name, gout.shape, gout.type)

    vec = np.array([[lat, lon, week]], dtype=np.float32)
    g = gs.run(None, {gin.name: vec})[0][0].astype(np.float64)
    print(f"  输出长度: {len(g)}  min={g.min():.4f} max={g.max():.4f}")

    mapping, miss = load_geomodel_index(species)
    print(f"\n  物种->geomodel 索引: 命中 {len(mapping)}/{len(species)}  未命中 {len(miss)}")

    # 只在有意义的 373 类里重排（模型输出 384 维，后 11 维未使用）
    n = len(species)
    raw = np.zeros(n)
    geo = np.zeros(n)
    for k, (sci, zh, j) in enumerate(mapping):
        raw[k] = probs[k]
        geo[k] = g[j] if j is not None else 0.0

    print("\n  --- 北京(39.90,116.41) 第20周 该地最可能出现的 10 种 ---")
    for k in np.argsort(-geo)[:10]:
        print(f"     {species[k][1]:<12} geo={geo[k]:.4f}")

    print("\n=== 重排后 Top-5 (alpha=1.0) ===")
    final = raw * np.power(geo, 1.0)
    for i, p in topk(final / final.sum()):
        print(f"  {species[i][1]}({species[i][0]})".ljust(50)
              + f" raw={raw[i]*100:5.2f}%  geo={geo[i]:.4f}  final={p*100:6.2f}%")

    for alpha in (0.5, 1.0, 2.0):
        f = raw * np.power(geo, alpha)
        i = int(np.argmax(f))
        print(f"  alpha={alpha:<4} top1={species[i][1]:<10} final={f[i]*100:.2f}%")


if __name__ == "__main__":
    main()
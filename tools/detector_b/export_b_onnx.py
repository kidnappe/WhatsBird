# -*- coding: utf-8 -*-
r"""B ckpt → onnx（export_yolox_onnx.py 的单类版；A 的脚本只认 80 类官方 exp，勿改它）。

  <prefix>_dec.onnx  带解码 [1,N,6]（cx,cy,w,h,obj,cls1，像素尺度）——PC bench 用
  <prefix>_raw.onnx  无解码（TFLite 转换用，Kotlin 解码）

喂法口径同 A：letterbox(114) + BGR 0-255 不归一化（/255 会全零响应，实测）。
ckpt 选择：yolox 训练产物里 ema_model 通常是评测更优的权重，优先取它。

Usage: python export_b_onnx.py <best_ckpt.pth> <size> <out_prefix>
例:    python export_b_onnx.py "E:\code\code tools\models_b\512\best_ckpt.pth" 512 "E:\code\code tools\detectors\yolox\yolox_bird_512"
"""
import os
import sys

import numpy as np  # noqa: E402

REPO = r"E:\code\code tools\yolox_repo"
EXP_FILE = r"E:\code\observebird\tools\detector_b\yolox_bird_exp.py"
sys.path.insert(0, REPO)

import onnx  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from yolox.exp import get_exp  # noqa: E402
from yolox.models.network_blocks import SiLU  # noqa: E402
from yolox.utils import replace_module  # noqa: E402


def load_sd(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw.get("ema_model") or raw.get("model") or raw
    if isinstance(sd, dict) and "module" in sd:      # ModelEMA.state_dict 包了一层
        sd = sd["module"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    tag = "ema_model" if raw.get("ema_model") else "model"
    return sd, tag


def export(ckpt, size, prefix, decode):
    os.environ["B_SIZE"] = str(size)  # exp 文件在 import 时读它
    import importlib.util
    spec = importlib.util.spec_from_file_location("yolox_bird_exp", EXP_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp = mod.Exp()
    exp.test_size = (size, size)
    model = exp.get_model()
    model.eval()
    sd, tag = load_sd(ckpt)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"⚠️ strict 加载失败，退 strict=False：{e}")
        model.load_state_dict(sd, strict=False)
    print(f"ckpt 权重取自 [{tag}]，num_classes={exp.num_classes}")
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = decode

    out = f"{prefix}_{'dec' if decode else 'raw'}.onnx"
    dummy = torch.randn(1, 3, size, size)
    with torch.no_grad():
        torch.onnx.export(model, dummy, out, input_names=["images"],
                          output_names=["output"], opset_version=11, dynamo=False)
    from onnxsim import simplify
    m = onnx.load(out)
    simp, check = simplify(m)
    assert check, "onnxsim failed"
    onnx.save(simp, out)
    print(f"exported {out} ({os.path.getsize(out) / 1048576:.1f} MB)")


def smoke(prefix, size):
    """COCO bird 真图冒烟：6 通道口径 bird=obj×cls0，分要非零、框要压得住 GT。"""
    import cv2
    import json
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(prefix + "_dec.onnx", so, providers=["CPUExecutionProvider"])
    with open(r"E:/code/code tools/data/coco/annotations/instances_val2017.json") as f:
        ann = json.load(f)
    bird_cat = [c["id"] for c in ann["categories"] if c["name"] == "bird"][0]
    imgs = {im["id"]: im for im in ann["images"]}
    gt = {}
    for a in ann["annotations"]:
        if a["category_id"] == bird_cat and not a.get("iscrowd"):
            gt.setdefault(imgs[a["image_id"]]["file_name"], []).append(a["bbox"])
    prods, ious = [], []
    for fn in sorted(gt.keys())[:6]:
        bgr = cv2.imread(os.path.join(r"E:/code/code tools/data/coco/val2017", fn))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        r = min(size / h, size / w)
        nh, nw = round(h * r), round(w * r)
        p = np.full((size, size, 3), 114, np.float32)
        p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
        x = np.ascontiguousarray(p.transpose(2, 0, 1)[None])  # BGR 0-255
        out = s.run(None, {s.get_inputs()[0].name: x})[0][0]  # [N,6]
        bird = out[:, 4] * out[:, 5]
        k = int(np.argmax(bird))
        prods.append(float(bird[k]))
        cx, cy, bw2, bh2 = out[k, :4] / r
        det = (cx - bw2 / 2, cy - bh2 / 2, bw2, bh2)
        best = 0.0
        for gx, gy, gw, gh in gt[fn]:
            ix = max(0, min(det[0] + det[2], gx + gw) - max(det[0], gx))
            iy = max(0, min(det[1] + det[3], gy + gh) - max(det[1], gy))
            u = det[2] * det[3] + gw * gh - ix * iy
            best = max(best, ix * iy / u if u > 0 else 0.0)
        ious.append(best)
    print(f"  smoke[{os.path.basename(prefix)}]: bird top分 " +
          " ".join(f"{p:.2f}" for p in prods) +
          " | top框IoU " + " ".join(f"{i:.2f}" for i in ious))
    assert max(prods) > 0.2, "bird 响应过低——导出/喂法/类别数有问题"
    assert max(ious) > 0.3, "top框与 GT 无重叠——框尺度有问题"


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    ckpt, size, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    export(ckpt, size, prefix, decode=True)
    smoke(prefix, size)
    export(ckpt, size, prefix, decode=False)

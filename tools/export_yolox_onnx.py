"""从官方 YOLOX pth 自导出 onnx（release 0.1.1rc0 的 onnx 与 pth 不匹配，不可用）。

导出四个评测用配置（带解码，输出 [1,N,85] cx,cy,w,h,obj,80cls，输入像素尺度）：
  yolox_nano_416_dec.onnx / yolox_s_416_dec.onnx / yolox_s_512_dec.onnx / yolox_s_640_dec.onnx
标准喂法：letterbox(114) + BGR + /255（官方 demo ValTransform，无通道翻转、无 mean/std）。

Usage: python export_yolox_onnx.py
"""
import os
import sys

import numpy as np

REPO = r"E:\code\code tools\yolox_repo"
CKPT_DIR = r"E:\code\code tools\detectors\yolox"
sys.path.insert(0, REPO)

import onnx  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from yolox.exp import get_exp  # noqa: E402
from yolox.models.network_blocks import SiLU  # noqa: E402
from yolox.utils import replace_module  # noqa: E402


def export(name, ckpt_file, size, out_name, decode=True):
    exp = get_exp(None, name)
    exp.test_size = (size, size)
    model = exp.get_model()
    model.eval()
    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_file), map_location="cpu",
                      weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = decode

    dummy = torch.randn(1, 3, size, size)
    path = os.path.join(CKPT_DIR, out_name)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, path,
            input_names=["images"], output_names=["output"],
            opset_version=11, dynamo=False,
        )
    # onnxsim 化简（降低后端转换/推理的算子碎片）
    from onnxsim import simplify
    m = onnx.load(path)
    simp, check = simplify(m)
    assert check, "onnxsim failed"
    onnx.save(simp, path)
    mb = os.path.getsize(path) / 1048576
    print(f"exported {out_name} ({mb:.1f} MB)")


def smoke(out_name, size):
    """真实鸟图冒烟：标准喂法（BGR 0-255 不归一化）下 bird 分显著非零且框 IoU 合理。"""
    import cv2
    import json
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(os.path.join(CKPT_DIR, out_name), so,
                             providers=["CPUExecutionProvider"])
    with open(r"E:/code/code tools/data/coco/annotations/instances_val2017.json") as f:
        ann = json.load(f)
    bird_cat = [c["id"] for c in ann["categories"] if c["name"] == "bird"][0]
    imgs = {im["id"]: im for im in ann["images"]}
    gt = {}
    for a in ann["annotations"]:
        if a["category_id"] == bird_cat and not a.get("iscrowd"):
            gt.setdefault(imgs[a["image_id"]]["file_name"], []).append(a["bbox"])
    files = sorted(gt.keys())[:6]
    prods, ious = [], []
    for fn in files:
        bgr = cv2.imread(os.path.join(r"E:/code/code tools/data/coco/val2017", fn))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        r = min(size / h, size / w)
        nh, nw = round(h * r), round(w * r)
        p = np.full((size, size, 3), 114, np.float32)
        p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
        x = np.ascontiguousarray(p.transpose(2, 0, 1)[None])  # BGR 0-255
        out = s.run(None, {s.get_inputs()[0].name: x})[0][0]
        bird = out[:, 4] * out[:, 5 + 14]
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
    print(f"  smoke[{out_name}]: bird top分 " + " ".join(f"{p:.2f}" for p in prods)
          + " | top框IoU " + " ".join(f"{i:.2f}" for i in ious))
    assert max(prods) > 0.2, f"{out_name} bird 响应过低，导出/喂法有问题"
    assert max(ious) > 0.3, f"{out_name} top框与 GT 无重叠，框尺度/喂法有问题"


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    # raw = 无解码（TFLite 转换用，Kotlin 解码）；dec = 带解码（PC 评测用）
    jobs = [
        ("yolox_nano", "yolox_nano.pth", 416, "yolox_nano_416_dec.onnx", True),
        ("yolox_s", "yolox_s.pth", 416, "yolox_s_416_dec.onnx", True),
        ("yolox_s", "yolox_s.pth", 512, "yolox_s_512_dec.onnx", True),
        ("yolox_s", "yolox_s.pth", 640, "yolox_s_640_dec.onnx", True),
        ("yolox_s", "yolox_s.pth", 416, "yolox_s_416_raw.onnx", False),
        ("yolox_s", "yolox_s.pth", 512, "yolox_s_512_raw.onnx", False),
        ("yolox_s", "yolox_s.pth", 640, "yolox_s_640_raw.onnx", False),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, ckpt, size, out, dec in jobs:
        if only and only not in out:
            continue
        if os.path.exists(os.path.join(CKPT_DIR, out)) and not only:
            print(f"skip {out} (exists)")
            continue
        if not os.path.exists(os.path.join(CKPT_DIR, ckpt)):
            print(f"SKIP {out}: 缺 {ckpt}")
            continue
        export(name, ckpt, size, out, decode=dec)
        if dec:
            smoke(out, size)
        else:
            print(f"  raw 导出跳过冒烟（布局验证在转换脚本里做）")

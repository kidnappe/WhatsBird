"""确认 YOLOX raw（无解码）onnx 的输出张量布局——Kotlin 解码器的规格来源。

raw 输出为单张量 [1,N,85]（三个 level 已在图内按 P3(52)/P4(26)/P5(13) 拼接，
只差 decode）。方法：同一张鸟图分别跑 raw 与 dec（带解码，语义已知
[N,85]=cx,cy,w,h,obj,cls80，输入像素尺度），穷举通道序并应用 decode 公式对齐。
"""
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

CKPT_DIR = r"E:\code\code tools\detectors\yolox"
SIZE = 416
STRIDES = (8, 16, 32)


def main():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    raw = ort.InferenceSession(os.path.join(CKPT_DIR, "yolox_s_416_raw.onnx"), so,
                               providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(CKPT_DIR, "yolox_s_416_dec.onnx"), so,
                               providers=["CPUExecutionProvider"])

    with open(r"E:/code/code tools/data/coco/annotations/instances_val2017.json") as f:
        ann = json.load(f)
    bird_cat = [c["id"] for c in ann["categories"] if c["name"] == "bird"][0]
    imgs = {im["id"]: im for im in ann["images"]}
    gt = {}
    for a in ann["annotations"]:
        if a["category_id"] == bird_cat and not a.get("iscrowd"):
            gt.setdefault(imgs[a["image_id"]]["file_name"], []).append(a["bbox"])
    fn = sorted(gt.keys())[0]
    bgr = cv2.imread(os.path.join(r"E:/code/code tools/data/coco/val2017", fn))
    h, w = bgr.shape[:2]
    r = min(SIZE / h, SIZE / w)
    nh, nw = round(h * r), round(w * r)
    p = np.full((SIZE, SIZE, 3), 114, np.float32)
    p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
    x = np.ascontiguousarray(p.transpose(2, 0, 1)[None])

    cat = raw.run(None, {raw.get_inputs()[0].name: x})[0][0]
    ref = dec.run(None, {dec.get_inputs()[0].name: x})[0][0]
    assert cat.shape == ref.shape, (cat.shape, ref.shape)
    print("raw:", cat.shape, " dec:", ref.shape)

    # grid：按 head 自然顺序 P3(8)/P4(16)/P5(13)，行主序（y 外 x 内）
    grids, counts = [], []
    for st in STRIDES:
        hw = SIZE // st
        gy, gx = np.mgrid[0:hw, 0:hw]
        grids.append(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))
        counts.append(hw * hw)
    g = np.concatenate(grids, 0)
    st = np.repeat(np.array(STRIDES, np.float32), counts)

    def build(ch_order):
        if ch_order == "reg_first":          # reg4, obj, cls80
            xy_raw, wh_raw, obj, cls = cat[:, :2], cat[:, 2:4], cat[:, 4], cat[:, 5:]
        else:                                 # cls80, obj, reg4
            cls, obj, reg = cat[:, :80], cat[:, 80], cat[:, 81:]
            xy_raw, wh_raw = reg[:, :2], reg[:, 2:]
        xy = (xy_raw + g) * st[:, None]
        wh = np.exp(wh_raw) * st[:, None]
        return np.concatenate([xy, wh, obj[:, None], cls], 1)

    best = None
    for ch_order in ("reg_first", "cls_first"):
        out = build(ch_order)
        d = float(np.abs(out - ref).max())
        if best is None or d < best[0]:
            best = (d, ch_order)
        print(f"ch={ch_order}: max|diff| vs dec = {d:.4e}")
    print("\n最佳:", best)


if __name__ == "__main__":
    main()

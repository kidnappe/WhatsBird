# -*- coding: utf-8 -*-
r"""打样一张图，对比 int8 tflite 与同语义 onnx 基线的原始输出通道（逐通道 min/max/mean）。

判据（HANDOFF-DETECTOR-B-INT8-SOLVED.md）：tflite 各通道 min/max/mean 应与基线 onnx
同量级；cls/obj 通道（0~1 sigmoid）若 mean 饱和到接近 1 → 直接作废（几千假框）。
基线默认选 raw onnx（与转换输入同语义：ch0-3 未解码偏移，ch4/5 已 sigmoid）；
dec onnx 的 ch0-3 是像素框，与 raw 产物不可直接对账。

Usage:
  python debug_drq_channels.py --tflite <tflite> --onnx <raw|dec onnx> [--size 512] [--image <jpg>]
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--image", default=r"E:\code\code tools\data\coco\val2017\000000000139.jpg")
    a = ap.parse_args()

    bgr = cv2.imread(a.image)
    assert bgr is not None, a.image
    size = a.size
    r = min(size / bgr.shape[0], size / bgr.shape[1])
    nh, nw = round(bgr.shape[0] * r), round(bgr.shape[1] * r)
    p = np.full((size, size, 3), 114, np.float32)
    p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
    x_nchw = np.ascontiguousarray(p.transpose(2, 0, 1)[None])
    x_nhwc = np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1))

    it = Interpreter(model_path=a.tflite, num_threads=1)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x_nhwc)
    it.invoke()
    raw_tf = it.get_tensor(it.get_output_details()[0]["index"])
    if raw_tf.ndim == 3 and raw_tf.shape[1] < raw_tf.shape[2]:
        raw_tf = np.ascontiguousarray(raw_tf.transpose(0, 2, 1))

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    raw_on = s.run(None, {s.get_inputs()[0].name: x_nchw})[0]

    print(f"tflite {os.path.basename(a.tflite)} 输出 {raw_tf.shape} | "
          f"onnx {os.path.basename(a.onnx)} 输出 {raw_on.shape}")
    ct, co = raw_tf[0], raw_on[0]
    assert ct.shape == co.shape, (ct.shape, co.shape)
    worst = 0.0
    for ch in range(ct.shape[1]):
        dt = abs(ct[:, ch].mean() - co[:, ch].mean())
        worst = max(worst, dt)
        print("ch%d | tflite min/max/mean: %+.3f/%+.3f/%+.3f | onnx min/max/mean: %+.3f/%+.3f/%+.3f | Δmean %.3f" % (
            ch, ct[:, ch].min(), ct[:, ch].max(), ct[:, ch].mean(),
            co[:, ch].min(), co[:, ch].max(), co[:, ch].mean(), dt))
    print(f"最大 Δmean = {worst:.3f}（>0.05 需人工复核，cls/obj 饱和即作废）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

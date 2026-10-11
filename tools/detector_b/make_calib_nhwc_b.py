# -*- coding: utf-8 -*-
r"""把 calib_b_<size>.npy（NCHW）转成 NHWC 版本，供 onnx2tf 2.6.9 flatbuffer_direct -oiqt 校准用。

背景（2026-10-07 定案，见 docs/handoff/detector_b/HANDOFF-DETECTOR-B-INT8-SOLVED.md）：
flatbuffer_direct 产的 tflite 输入是 NHWC [1,S,S,3]，而 collect_calibration_ranges_from_tflite
把样本直接 set_tensor 进 tflite interpreter；喂 NCHW (1,3,S,S) 形状不匹配 →
resize_tensor_input 抛 concatenation.cc (1 != 2) → 校准静默失败 → 重试 attempt
重写 float32.tflite 撞上泄漏 interpreter 的文件锁（Errno 22）。
喂 NHWC 校准数据从根上消掉 resize 分支，两层崩溃一起消失。

Usage: python make_calib_nhwc_b.py
产物: E:\code\code tools\data\detector_b\calib\calib_b_{size}_nhwc.npy
"""
import os

import numpy as np

CALIB_DIR = r"E:\code\code tools\data\detector_b\calib"
SIZES = (640, 512)


def main():
    for size in SIZES:
        src = os.path.join(CALIB_DIR, f"calib_b_{size}.npy")
        dst = os.path.join(CALIB_DIR, f"calib_b_{size}_nhwc.npy")
        arr = np.load(src)
        out = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))
        np.save(dst, out)
        print(f"  {dst}  {arr.shape} -> {out.shape}  "
              f"({os.path.getsize(dst) / 1048576:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

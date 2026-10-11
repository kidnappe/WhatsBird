# -*- coding: utf-8 -*-
r"""把 calib_b_files.txt 的校准图打包成 onnx2tf -cind 用的 .npy（两个尺寸各一份）。

喂法口径与部署严格一致：letterbox(114) + BGR 0-255 不归一化。
-cind 附加 mean=0 std=1，数据进模型前不再做任何缩放。
⚠️ onnx2tf tflite_builder 只认 .npy（np.load 须返回裸数组），.npz 会报
   "could not convert string to float"。

Usage: python make_calib_npz_b.py
产物: E:\code\code tools\data\detector_b\calib\calib_b_{size}.npy
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\code\observebird\tools")
from quantize_ort import letterbox_nchw  # noqa: E402  同一 letterbox 实现，口径不漂移

CALIB_LIST = r"E:\code\code tools\data\detector_b\calib_b_files.txt"
OUT_DIR = r"E:\code\code tools\data\detector_b\calib"
SIZES = (640, 512)


def main():
    files = [l.strip() for l in open(CALIB_LIST, encoding="utf-8") if l.strip()]
    files = [f for f in files if os.path.exists(f)]
    print(f"校准图 {len(files)} 张 -> {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for size in SIZES:
        arr = np.stack([letterbox_nchw(cv2.imread(f), size)[0] for f in files
                        if cv2.imread(f) is not None]).astype(np.float32)
        out = os.path.join(OUT_DIR, f"calib_b_{size}.npy")
        # onnx2tf tflite_builder 的校准加载器只吃 .npy（np.load 返回裸数组）；
        # mean=0/std=1 由命令行给，数据本身保持 BGR 0-255 原值
        np.save(out, arr)
        print(f"  {out}  shape={arr.shape}  "
              f"({os.path.getsize(out) / 1048576:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

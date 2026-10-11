# -*- coding: utf-8 -*-
r"""产 YoloDetector.kt selfTest 的 PC 基准（换模型/换阈值时同步重跑）。

喂法管线与 YoloDetector.fillInput 逐项对齐：letterbox 114 右下补齐（图贴左上）+
BGR 0-255 不归一化 + float32 NHWC。输出 [1,C,N] 转置回 [1,N,C] 后走 decode_raw
（th=MIN_DET_SCORE，含同款贪心 NMS 0.45），框坐标 ÷ratio 回原图再归一化 0-1
（Detection 口径）。打印格式对齐 logcat "selftest ..." 行，装机后逐项比对。

Usage: python selftest_baseline.py <tflite> [image] [th]
  image 默认 App assets 的 selftest_bird.jpg；th 默认 0.45
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\code\observebird\tools")
from export_detector_tflite import decode_raw  # noqa: E402

SELFTEST_JPG = r"E:\code\observebird\android\app\src\main\assets\models\selftest_bird.jpg"


def main():
    tflite = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else SELFTEST_JPG
    th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45

    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tflite, num_threads=4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    size = int(inp["shape"][1])
    print(f"模型 {tflite}\n输入 {inp['shape']} {inp['dtype']} | th={th}")

    bgr = cv2.imread(img_path)
    assert bgr is not None, img_path
    h, w = bgr.shape[:2]
    ratio = min(size / h, size / w)
    nh = min(round(h * ratio), size)
    nw = min(round(w * ratio), size)
    p = np.full((size, size, 3), 114, np.float32)
    p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)

    it.set_tensor(inp["index"], p[None])
    it.invoke()
    got = it.get_tensor(it.get_output_details()[0]["index"])
    if got.ndim == 3 and got.shape[1] < got.shape[2]:
        got = np.ascontiguousarray(got.transpose(0, 2, 1))

    max_score = float((got[0][:, 4] * got[0][:, 5]).max())
    dets = decode_raw(got, size, score_th=th, cls_idx=0)
    nsrcw, nsrch = w * ratio, h * ratio
    kept = []
    for s, x1, y1, x2, y2 in dets:
        l = min(max(x1 / nsrcw, 0.0), 1.0)
        t = min(max(y1 / nsrch, 0.0), 1.0)
        r = min(max(x2 / nsrcw, 0.0), 1.0)
        b = min(max(y2 / nsrch, 0.0), 1.0)
        if r > l and b > t:
            kept.append((s, l, t, r, b))

    print("----- PC 基准（对齐 logcat selftest 行）-----")
    print(f"selftest {w}x{h} maxScore={max_score:.4f} dets={len(kept)} " +
          " ".join("s=%.2f l=%.2f t=%.2f r=%.2f b=%.2f" % d for d in kept[:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

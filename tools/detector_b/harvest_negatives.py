# -*- coding: utf-8 -*-
r"""从装机实测视频收割「非鸟」负样本 crop（拒识模型训练用）。

原理：实测视频里没有真鸟，检测器 (raw onnx, PC fp32) 检出的所有框都是 FP——
把 ≥阈值 的检出裁出来就是"检测器真实误检分布"的负样本，比随手拍更对症。

Usage:
  python harvest_negatives.py <video.mp4> [<video2.mp4> ...] --out <负样本目录>
  可选: --conf 0.40 --stride 5 (每 N 帧抽一帧) --max-per-video 400
产物: <out>/neg_XXXX.jpg + <out>/manifest.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, r"E:\code\observebird\tools")
from export_detector_tflite import decode_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default=r"E:\code\code tools\data\detector_b\ckpt_sweep\onnx\yolox_bird_640_best_raw.onnx")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-per-video", type=int, default=400)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    size = sess.get_inputs()[0].shape[2]

    manifest, n = [], 0
    for video in a.videos:
        cap = cv2.VideoCapture(video)
        assert cap.isOpened(), video
        fid, kept, total = -1, 0, 0
        while True:
            fid += 1
            ok, frame = cap.read()
            if not ok:
                break
            if fid % a.stride:
                continue
            total += 1
            h, w = frame.shape[:2]
            r = min(size / h, size / w)
            nh, nw = min(round(h * r), size), min(round(w * r), size)
            p = np.full((size, size, 3), 114, np.float32)
            p[:nh, :nw] = cv2.resize(frame, (nw, nh)).astype(np.float32)
            x = np.ascontiguousarray(p.transpose(2, 0, 1)[None])
            out = sess.run(None, {inp_name: x})[0]
            if out.ndim == 3 and out.shape[1] < out.shape[2]:
                out = np.ascontiguousarray(out.transpose(0, 2, 1))
            dets = decode_raw(out, size, score_th=a.conf, cls_idx=0)
            for s, x1, y1, x2, y2 in dets:
                # 输入尺度 → 原帧尺度（右下补齐 letterbox：左上对齐，无偏移）
                fx1, fy1 = int(max(x1 / r, 0)), int(max(y1 / r, 0))
                fx2, fy2 = int(min(x2 / r, w)), int(min(y2 / r, h))
                if fx2 - fx1 < 24 or fy2 - fy1 < 24:
                    continue
                crop = frame[fy1:fy2, fx1:fx2]
                name = f"neg_{n:05d}.jpg"
                cv2.imwrite(os.path.join(a.out, name), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                manifest.append({"file": name, "src": os.path.basename(video),
                                 "frame": fid, "det_score": round(float(s), 3)})
                n += 1
                kept += 1
                if kept >= a.max_per_video:
                    break
            if kept >= a.max_per_video:
                break
        cap.release()
        print(f"{os.path.basename(video)}: 抽帧 {total}，收割 FP crop {kept}")
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"合计负样本 {n} 个 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

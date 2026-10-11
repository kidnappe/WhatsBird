# -*- coding: utf-8 -*-
"""一鸟多框 PC 实证：shiji_test 50 张（用户实机测试同一批图）跑 640-best raw onnx，
对比 NMS 0.45（现役）与更低阈值下重复框的存活情况。
重复框判定：top1 与 top2 检出 IoU ≥ 0.3（同一只鸟的嵌套/平移框）。
"""
import json
import os
import sys

import cv2
import numpy as np

OBS_TOOLS = r"E:\code\observebird\tools"
sys.path.insert(0, OBS_TOOLS)
sys.path.insert(0, os.path.join(OBS_TOOLS, "detector_b"))
import bench_detector_a as A  # noqa: E402

DATA = r"E:\code\code tools\data"
RAW_ONNX = os.path.join(DATA, "detector_b", "ckpt_sweep", "onnx", "yolox_bird_640_best_raw.onnx")
TEST_DIR = os.path.join(DATA, "shiji_test")
STRIDES = [8, 16, 32]
DET_TH = 0.45


def decode(raw, size, nms_th, score_th=DET_TH, cls_idx=0):
    """decode_raw 复制品，NMS 阈值参数化（原版硬编码 0.45）。"""
    c = raw[0]
    grids, counts = [], []
    for st in STRIDES:
        hw = size // st
        gy, gx = np.mgrid[0:hw, 0:hw]
        grids.append(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))
        counts.append(hw * hw)
    g = np.concatenate(grids, 0)
    st = np.repeat(np.array(STRIDES, np.float32), counts)
    xy = (c[:, :2] + g) * st[:, None]
    wh = np.exp(c[:, 2:4]) * st[:, None]
    bird = c[:, 4] * c[:, 5 + cls_idx]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    m = bird >= score_th
    dets = [(float(bird[i]), *boxes[i].tolist()) for i in np.where(m)[0]]
    dets.sort(key=lambda d: -d[0])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            ix = max(0, min(d[3], k[3]) - max(d[1], k[1]))
            iy = max(0, min(d[4], k[4]) - max(d[2], k[2]))
            u = (d[3] - d[1]) * (d[4] - d[2]) + (k[3] - k[1]) * (k[4] - k[2]) - ix * iy
            if u > 0 and ix * iy / u >= nms_th:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    u = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def main():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.log_severity_level = 3
    sess = ort.InferenceSession(RAW_ONNX, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = int(inp.shape[2])

    manifest = json.load(open(os.path.join(TEST_DIR, "manifest.json"), encoding="utf-8"))
    n_multi = 0
    rows = []
    for it in manifest:
        bgr = cv2.imread(os.path.join(TEST_DIR, it["file"]))
        if bgr is None:
            continue
        x, r = A.letterbox(bgr, size)
        raw = sess.run(None, {inp.name: x[None] if x.ndim == 3 else x})[0]
        # raw [1,N,C]；框在 letterbox 输入尺度，换算回原图只需 /r（letterbox 不平移）
        d45 = decode(raw, size, nms_th=0.45)
        d30 = decode(raw, size, nms_th=0.30)
        iou12 = iou(d45[0], d45[1]) if len(d45) >= 2 else None
        dup = iou12 is not None and iou12 >= 0.3
        n_multi += dup
        rows.append({"file": it["file"], "zh": it["zh"], "n45": len(d45),
                     "n30": len(d30), "iou12": None if iou12 is None else round(iou12, 2),
                     "dup": dup})
    print(f"n={len(rows)}  NMS0.45 下 top2 构成同鸟重复框(IoU>=0.3)的图数: {n_multi}")
    print(f"NMS0.45 -> 0.30 后检出数变化: sum {sum(r['n45'] for r in rows)} -> "
          f"{sum(r['n30'] for r in rows)}")
    print("\n重复框案例（现役 NMS 下同一只鸟两个框都活着）:")
    for r in rows:
        if r["dup"]:
            print(f"  IoU12={r['iou12']} n45={r['n45']}  {r['zh']}  {r['file'][:44]}")
    print("\n多检出但非重复（IoU<0.3，第二框是别处 FP，降 NMS 不解决）:")
    for r in rows:
        if r["n45"] >= 2 and not r["dup"]:
            print(f"  IoU12={r['iou12']} n45={r['n45']}  {r['zh']}")
    out = os.path.join(DATA, "crop_regime", "dup_analysis.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n明细 -> {out}")


if __name__ == "__main__":
    main()

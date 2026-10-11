# -*- coding: utf-8 -*-
r"""在训练集照片上跑仓库自带的检测器 + 区域分类器，输出 App 那条三段式管线的真实结果。

口径全部照着 Kotlin 抄，不自己发明：
  检测  YoloDetector.kt / bench_detb_tflite.py：letterbox(114) BGR 0-255，NHWC float32，
        decode_raw（含 NMS 0.45），单类。
  扩框  ThreeStagePipeline.kt：best.expand(ModelRegistry.BOX_EXPAND_RATIO)，本版 = 4.0f。
  分类  TfLiteClassifier.kt：centerCrop224（短边缩到 256 再中心裁 224）
        + ImageNet 归一 + softmax，取 Top-5。

用途有两个：
  1. 自检——数据集文件名带学名，拿它当标准答案，预测对不上就说明预处理写错了；
  2. 给下载页画「识别状态」用，框和置信度都是真跑出来的，不是编的。

Usage:
  python real_identify.py <photo.jpg> [--model huazhong] [--json out.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = r"E:\code\observebird"
MODELS = os.path.join(REPO, r"android\app\src\main\assets\models")
sys.path.insert(0, os.path.join(REPO, "tools"))

BOX_EXPAND_RATIO = 4.0     # ModelRegistry.kt
INPUT = 224                # TfLiteClassifier.kt companion
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

REGION = {  # 区域 -> (分类器权重, 标签文件)；与 RegionClassifierRouter 同名
    "huazhong": ("birdnet_huazhong.tflite", "labels_huazhong.txt"),
    "huadong": ("birdnet_huadong.tflite", "labels_huadong.txt"),
    "huanan_sw": ("birdnet_huanan_sw.tflite", "labels_huanan_sw.txt"),
    "huabei": ("birdnet_huabei.tflite", "labels_huabei.txt"),
    "dongbei": ("birdnet_dongbei.tflite", "labels_dongbei.txt"),
}


def load_labels(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            pairs.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else "?"))
    return pairs


def letterbox(bgr, size):
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    import cv2
    pad = np.full((size, size, 3), 114, np.float32)
    pad[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
    return pad, r


def detect_best(photo_path, model="yolox_bird_640_best_int8.tflite"):
    from ai_edge_litert.interpreter import Interpreter
    from export_detector_tflite import decode_raw
    import cv2

    it = Interpreter(model_path=os.path.join(MODELS, model), num_threads=8)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    size = int(inp["shape"][1])
    bgr = cv2.imread(photo_path)
    if bgr is None:
        raise SystemExit("读不出图：" + photo_path)
    pad, r = letterbox(bgr, size)
    it.set_tensor(inp["index"], pad[None])
    it.invoke()
    got = it.get_tensor(it.get_output_details()[0]["index"])
    if got.ndim == 3 and got.shape[1] < got.shape[2]:
        got = np.ascontiguousarray(got.transpose(0, 2, 1))
    dets = decode_raw(got, size, score_th=0.35, cls_idx=0)
    if not dets:
        return None, (bgr.shape[1], bgr.shape[0])
    s, x1, y1, x2, y2 = max(dets, key=lambda d: d[0])
    h, w = bgr.shape[:2]
    return {"score": float(s),
            "box": [x1 / r, y1 / r, x2 / r, y2 / r],
            "img_wh": [w, h]}, (w, h)


def expand_crop(box, wh, ratio=BOX_EXPAND_RATIO):
    """照 ThreeStagePipeline.kt 的 Rect.expand：四边按 ratio 外扩后夹回画面。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * ratio / 2.0, (y2 - y1) * ratio / 2.0
    w, h = wh
    return [max(0.0, cx - hw), max(0.0, cy - hh),
            min(float(w), cx + hw), min(float(h), cy + hh)]


def center_crop_224(img):
    w, h = img.size
    scale = 256.0 / min(w, h)
    nw, nh = max(INPUT, round(w * scale)), max(INPUT, round(h * scale))
    s = img.resize((nw, nh), Image.BILINEAR)   # createScaledBitmap(..., true) ≈ 双线性
    left, top = (nw - INPUT) // 2, (nh - INPUT) // 2
    return s.crop((left, top, left + INPUT, top + INPUT))


def classify(crop_img, region):
    from ai_edge_litert.interpreter import Interpreter
    tfl, lab = REGION[region]
    it = Interpreter(model_path=os.path.join(MODELS, tfl), num_threads=8)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    arr = np.asarray(center_crop_224(crop_img), np.float32) / 255.0
    arr = (arr - MEAN) / STD
    it.set_tensor(inp["index"], arr[None].astype(np.float32))
    it.invoke()
    logits = it.get_output_details()[0]
    raw = it.get_tensor(logits["index"])[0].astype(np.float64)
    e = np.exp(raw - raw.max())
    probs = e / e.sum()
    labels = load_labels(os.path.join(MODELS, lab))
    order = np.argsort(-probs)[:5]
    return [{"sci": labels[i][0] if i < len(labels) else "?",
             "zh": labels[i][1] if i < len(labels) else "?",
             "p": float(probs[i])} for i in order]


def run(photo_path, region="huazhong"):
    det, wh = detect_best(photo_path)
    out = {"photo": photo_path, "region": region, "img_wh": wh}
    if det is None:
        out["detected"] = False
        return out
    img = Image.open(photo_path).convert("RGB")
    rect = expand_crop(det["box"], wh)
    out["detected"] = True
    out["det"] = {"score": round(det["score"], 4),
                  "box": [round(v, 1) for v in det["box"]],
                  "crop_rect": [round(v, 1) for v in rect]}
    out["top5"] = classify(img.crop(tuple(int(v) for v in rect)), region)
    out["top1"] = out["top5"][0]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--region", default="huazhong")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = run(a.photo, a.region)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(txt)
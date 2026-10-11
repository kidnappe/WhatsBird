"""onnxruntime 静态量化（QDQ）：raw onnx -> QDQ onnx，再交 onnx2tf 转 int8 tflite。

校准数据：COCO bird 80 张 + 华北 20 张，预处理与部署严格一致
（letterbox 114 + BGR 0-255 不归一化，ONNX 输入序 NCHW，batch=1 逐张喂）。

Usage: python quantize_ort.py <raw.onnx> <size> <out_qdq.onnx> [calib_list.txt]
  calib_list.txt 缺省用内置 COCO80+华北20 配方；B 路线传 calib_b_files.txt
  （make_calib_b.py 产物，华北100+难例60）。
"""
import json
import os
import sys

import cv2
import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

CKPT_DIR = r"E:\code\code tools\detectors\yolox"


def letterbox_nchw(bgr, size):
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    p = np.full((size, size, 3), 114, np.float32)
    p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
    return p.transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,S,S]


def calib_files():
    with open(r"E:/code/code tools/data/coco/annotations/instances_val2017.json") as f:
        ann = json.load(f)
    bird_cat = [c["id"] for c in ann["categories"] if c["name"] == "bird"][0]
    imgs = {im["id"]: im for im in ann["images"]}
    gt = {}
    for a in ann["annotations"]:
        if a["category_id"] == bird_cat and not a.get("iscrowd"):
            gt.setdefault(imgs[a["image_id"]]["file_name"], []).append(a["bbox"])
    files = [os.path.join(r"E:/code/code tools/data/coco/val2017", f)
             for f in sorted(gt.keys())]
    files = [f for f in files if os.path.exists(f)][:80]
    files += sorted(os.path.join(r"E:/code/code tools/data/huabei/images", f)
                    for f in os.listdir(r"E:/code/code tools/data/huabei/images")
                    if f.endswith(".jpg"))[:20]
    return files


class Reader(CalibrationDataReader):
    def __init__(self, onnx_path, size, calib_list=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_name = sess.get_inputs()[0].name
        self.size = size
        files = None
        if calib_list and os.path.exists(calib_list):
            files = [l.strip() for l in open(calib_list, encoding="utf-8") if l.strip()]
            files = [f for f in files if os.path.exists(f)]
            print(f"校准清单: {calib_list} ({len(files)} 张)")
        if not files:
            files = calib_files()
        self.data = [letterbox_nchw(cv2.imread(f), size) for f in files
                     if cv2.imread(f) is not None]
        self.idx = 0
        print(f"校准样本: {len(self.data)}")

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        x = self.data[self.idx]
        self.idx += 1
        return {self.input_name: x}


def main():
    onnx_path, size, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    calib_list = sys.argv[4] if len(sys.argv) > 4 else None  # 可选：calib_b_files.txt 绝对路径
    quantize_static(
        onnx_path, out_path, Reader(onnx_path, size, calib_list),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    print(f"QDQ onnx -> {out_path} ({os.path.getsize(out_path)/1048576:.1f} MB)")


if __name__ == "__main__":
    main()

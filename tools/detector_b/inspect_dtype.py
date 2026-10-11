# -*- coding: utf-8 -*-
"""解剖现役 s512 与新转换的 drq 产物的 dtype 布局，判定现役的量化形态。"""
from ai_edge_litert.interpreter import Interpreter
import collections
import os

for p, tag in [
    # 2026-10-13：这份 s512 int8 已从 App assets 移出（APK 瘦身，详见 android/models/README.md），
    # 逐位相同的副本在数据盘 detectors 下，改从那里读。
    (r"E:\code\code tools\detectors\yolox\tflite_s512_int8\yolox_s_512_int8.tflite", "退役s512"),
    (r"E:\code\code tools\data\detector_b\ckpt_sweep\tflite\alt_512\yolox_bird_512_ep35_qdq_dynamic_range_quant.tflite", "新drq512"),
]:
    it = Interpreter(model_path=p, num_threads=1)
    it.allocate_tensors()
    d = it.get_tensor_details()
    c = collections.Counter(str(t["dtype"]).replace("<class 'numpy.", "").replace("'>", "") for t in d)
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    print(tag, "dtype分布", dict(c),
          "| 输入", inp["shape"], inp["dtype"],
          "| 输出", out["shape"], out["dtype"],
          "| 大小", round(os.path.getsize(p) / 1048576, 2), "MB")

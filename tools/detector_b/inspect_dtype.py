# -*- coding: utf-8 -*-
"""解剖现役 s512 与新转换的 drq 产物的 dtype 布局，判定现役的量化形态。"""
from ai_edge_litert.interpreter import Interpreter
import collections
import os

for p, tag in [
    (r"E:\code\observebird\android\app\src\main\assets\models\yolox_s_512.tflite", "现役s512"),
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

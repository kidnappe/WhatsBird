# -*- coding: utf-8 -*-
r"""ai_edge_quantizer 路线：float32 tflite → 真 int8（全整型、float 输入输出）。

背景（2026-10-07 定案，见 docs/handoff/detector_b/HANDOFF-DETECTOR-B-INT8-SOLVED.md）：
onnx2tf 2.6.9 的 -oiqt 严格整型构建器对本模型结构性坏死（LOGISTIC 固定 scale 合同 /
CONCATENATION 共享 scale / QDQ 白名单 / per-channel 权重轴，四连坏）。
现役 s512 int8（2026-10-05 产）经时间线与产物残骸（integer_quant.tflite.tmp + 当日
ai_edge_quantizer 0.9.0 安装记录 + 4 分钟空档）比对，实为两步法所产：
  onnx2tf → float32 tflite（raw 输入，本就无碍）→ ai_edge_quantizer 静态量化。
本脚本复刻第二步。校准喂 NHWC [1,S,S,3] BGR 0-255（与部署口径一致，make_calib_nhwc_b.py 产物）。

Usage: python quantize_aeq.py <float32.tflite> <out_int8.tflite> <calib_nhwc.npy> [max_samples]
"""
import sys

import numpy as np

SIG_KEY = "serving_default"  # onnx2tf flatbuilder 产物自带该 signature


def _patch_qsv_min_max():
    """致命坑修复：ai_edge_quantizer 0.9.0 给所有算子注册的默认 QSV 更新函数是
    滑动平均（register_quantized_op 默认参数 qsv_utils.moving_average_update，
    smoothing=0.95），对 min_max_uniform_quantize 语义完全错误——校准集中稀有的
    强响应极值（如 obj logit +6.7）会被大量无鸟样本平均掉，量化窗口塌缩，强检出
    在 int8 里被削顶（实测 obj 恒 0.5、cls 恒 0.87）。
    注意：函数对象在 import 注册时就绑死为默认参数，patch 模块属性无效，
    必须遍历 _algorithm_registry 逐算子替换。MIN_MAX 语义 = 逐元素取 min/max。"""
    from ai_edge_quantizer import algorithm_manager as am

    def _min_max_update(qsv, new_qsv, smoothing_factor=1.0):
        if not qsv:
            return new_qsv
        return {
            "min": np.minimum(np.asarray(qsv["min"]), np.asarray(new_qsv["min"])),
            "max": np.maximum(np.asarray(qsv["max"]), np.asarray(new_qsv["max"])),
        }

    n = 0
    for algo_info in am._alg_manager_instance._algorithm_registry.values():
        for op_info in algo_info.quantized_ops.values():
            if op_info.update_qsv_func is am.qsv_utils.moving_average_update:
                op_info.update_qsv_func = _min_max_update
                n += 1
    print(f"[qsv-patch] 已替换 {n} 个算子的滑动平均 → 真 min/max", flush=True)


def main():
    float_model, out_path, calib_path = sys.argv[1], sys.argv[2], sys.argv[3]
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9

    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=float_model, num_threads=1)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    in_name = inp["name"]
    print(f"float 模型输入: {in_name} {inp['shape']} {inp['dtype']}", flush=True)

    calib = np.load(calib_path, mmap_mode="r")
    n = min(len(calib), max_samples)
    samples = []
    for i in range(n):
        arr = np.asarray(calib[i], dtype=np.float32)
        assert arr.shape == tuple(inp["shape"][1:]), (i, arr.shape, inp["shape"])
        samples.append({in_name: arr[None]})
    print(f"校准样本: {n}", flush=True)

    from ai_edge_quantizer import quantizer
    from ai_edge_quantizer.qtyping import TFLOperationName

    _patch_qsv_min_max()
    q = quantizer.Quantizer(float_model=float_model)
    # 显式枚举计算算子（'*' 通配会连 INPUT/OUTPUT 边界一起量化成 int8；
    # 边界不配 → 保持 float32 输入输出，量化器在边界内自动插 QUANTIZE/DEQUANTIZE，
    # 与现役 s512（375 int8 + 96 int32 + float32 输入输出）同型
    skip = {TFLOperationName.ALL_SUPPORTED, TFLOperationName.INPUT,
            TFLOperationName.OUTPUT, TFLOperationName.CUSTOM_OP}
    n_ok = 0
    for op in TFLOperationName:
        if op in skip:
            continue
        try:
            q.add_static_config(
                regex=".*",
                operation_name=op,
                activation_num_bits=8,
                weight_num_bits=8,
            )
            n_ok += 1
        except ValueError:
            pass  # 该算子无 int8 计算策略（如 EMBEDDING_LOOKUP），模型里没有则无关紧要
    print(f"已配置算子 {n_ok} 个", flush=True)
    cal_res = q.calibrate({SIG_KEY: samples}, num_threads=8)
    print("校准完成", flush=True)
    q.quantize(calibration_result=cal_res, serialize_to_path=out_path)
    print(f"已产出: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

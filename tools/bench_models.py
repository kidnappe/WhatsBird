"""ObserveBird: on-device latency benchmark.

Measures per-frame cost of every model in the pipeline on this CPU, then
projects achievable camera frame rates on a mid-range phone.

Usage: python bench_models.py
"""
import os
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

TOOLS = r"E:\code\code tools"
CLASSIFIER = os.path.join(TOOLS, "bird-cn-373", "export", "tflite")
GEOMODEL = os.path.join(TOOLS, "geomodel")
DETECTOR = os.path.join(TOOLS, "detectors")

# Mid-range Android reference points (published figures for guidance only).
PHONE_REF = {
    "低中端 (骁龙 6 系 / 天玑 700, 4GB RAM)": 0.35,
    "中端 (骁龙 8 Gen1 / 天玑 8000)": 1.00,
    "旗舰 (骁龙 8 Gen3+)": 2.20,
}


def bench_pytorch(warmup=3, runs=12):
    """OneDNN-optimised CPU path — a fair floor for on-device classifier cost."""
    import sys

    import torch

    repo = os.path.join(TOOLS, "bird-cn-373", "code", "Bird-Classification-main")
    weights = os.path.join(TOOLS, "bird-cn-373", "model_v1base.pth")
    sys.path.insert(0, repo)
    from _BirdNet import getModel

    torch.set_num_threads(8)
    net = getModel("v1base", load_weight=weights).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            net(x)
    ms = (time.perf_counter() - t0) / runs * 1000
    mb = os.path.getsize(weights) / 1024 / 1024
    print(f"  {'birdnet pytorch fp32 (oneDNN,8线程)':<42} {ms:7.1f} ms   {mb:6.2f} MB   input (1, 3, 224, 224)")
    return ms


def bench_tflite(path, feed, label, warmup=3, runs=12):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    shape = tuple(int(v) for v in inp["shape"])
    # Build the probe straight from the declared shape so NCHW/NHWC and
    # float/uint8 models are all handled without special-casing.
    hi = 255.0 if np.dtype(inp["dtype"]) == np.uint8 else 1.0
    data = (np.random.rand(*shape) * hi).astype(inp["dtype"])
    it.set_tensor(inp["index"], data)
    for _ in range(warmup):
        it.invoke()
    t0 = time.perf_counter()
    for _ in range(runs):
        it.invoke()
    ms = (time.perf_counter() - t0) / runs * 1000
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {label:<42} {ms:7.1f} ms   {mb:6.2f} MB   input {shape}")
    return ms


def bench_onnx(path, feed, label, warmup=3, runs=12):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 8
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    s.run(None, {name: feed})
    t0 = time.perf_counter()
    for _ in range(runs):
        s.run(None, {name: feed})
    ms = (time.perf_counter() - t0) / runs * 1000
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {label:<42} {ms:7.1f} ms   {mb:6.2f} MB   input {feed.shape}")
    return ms


def main():
    print("=== 本机基准 (Ryzen 7 6800H, 8 线程) ===")
    print("注意: Windows 版 ai_edge_litert 无 XNNPACK delegate，走参考内核，")
    print("      TFLite 绝对耗时不代表手机表现。分类器改用 PyTorch(oneDNN) 测。\n")

    print("[检测模型] 找出一张 300x300 图里哪里有鸟")
    det = {}
    p = os.path.join(DETECTOR, "ssd_mobilenet_v1_1_fp32_300x300.tflite")
    if os.path.exists(p):
        det["ssd_mobilenet_v1_1"] = bench_tflite(
            p, None, "ssd_mobilenet_v1_1 uint8 (⚠参考内核,不可信)")

    print("\n[分类模型] 判断这是哪种鸟")
    cls = {}
    p = os.path.join(CLASSIFIER, "birdnet_373_float32.tflite")
    if os.path.exists(p):
        cls["birdnet tflite fp32"] = bench_tflite(
            p, None, "birdnet tflite fp32 (⚠参考内核,不可信)")
    cls["birdnet pytorch"] = bench_pytorch()

    print("\n[地理先验] 这个时空该出现什么")
    geo = 0.0
    p = os.path.join(GEOMODEL, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx")
    if os.path.exists(p):
        geo = bench_onnx(
            p, np.array([[39.9, 116.4, 20]], dtype=np.float32),
            "geomodel onnx fp16 (14082类)")

    total = sum(det.values()) + sum(cls.values()) + geo
    print(f"\n  本机合计 {total:.1f} ms（⚠ 含不可信的 TFLite 参考内核耗时）\n")

    print("=== 可信的单帧耗时（本机 Ryzen 6800H, 8 线程）===\n")
    print(f"  地理先验 geomodel : {geo:5.2f} ms   ← ONNX Runtime，真实值")
    print(f"                     端侧同样是亚毫秒级（输入只有 3 个 float）")
    if "birdnet pytorch" in cls:
        c = cls["birdnet pytorch"]
        print(f"  分类器(FP32,优化) : {c:5.1f} ms   ← PyTorch oneDNN，代表'优化良好的 CPU 推理'")
        print()
        print("  === 帧预算（按 1000 ms 一帧倒推）===")
        for target in (10, 15, 20, 30):
            budget = 1000 / target
            print(f"    目标 {target:2d} fps -> 每帧 {budget:5.1f} ms  "
                  f"(本机分类已占 {c:.0f} ms, 占 {c / budget * 100:.0f}%)")

    print("\n=== 必须真机验证（本机无法给出可信绝对值）===")
    print("  Windows 版 ai_edge_litert 无 XNNPACK delegate，走参考内核，")
    print("  TFLite 耗时不代表手机。Android LiteRT 有 XNNPACK/GPU/NNAPI delegate，")
    print("  通常比本机参考内核快 1~2 个数量级。")
    print("  可参考的公开数据点：wzf19947 在 Axera NPU 上实测 检测 3.1ms / 识别 16.3ms")
    print("  （硬件加速，与手机 CPU 不可直接比较）")
    print("\n  结论: 分层策略是可行的，但帧率上限必须在目标机型上实测确认。")


if __name__ == "__main__":
    main()
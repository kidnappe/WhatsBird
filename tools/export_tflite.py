"""Export a trained huabei checkpoint (.pt) to App-ready TFLite.

Pipeline (mirrors the validated bird-cn-373 export):
  best.pt -> timm model -> ONNX (batch=1, NCHW opset 17)
          -> numeric check vs PyTorch (ORT)
          -> onnx2tf -> float32 TFLite (NHWC)
          -> numeric re-check via ai_edge_litert

Usage:
  python tools/export_tflite.py --ckpt "E:/code/code tools/models/huabei_mnv4_300/best.pt" \
      --out-dir "E:/code/code tools/models/huabei_mnv4_300/export"
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=8,
                    help="随机张量样本数用于数值对齐检查")
    ap.add_argument("--species-list", default=r"E:\code\code tools\bird-cn-373\code"
                    r"\Bird-Classification-main\species_huabei_150.csv",
                    help="学名->中文名对照 csv（scientific_name,common_name 列）")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model_name, species = ck["model_name"], ck["species"]
    n_cls = len(species)
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"模型 {model_name}  {n_cls} 类  ckpt top1={ck.get('top1', 0)*100:.2f}%")

    # 1) 重建网络并加载权重
    import timm
    net = timm.create_model(model_name, pretrained=False, num_classes=n_cls)
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    print("权重加载 strict=True 通过")

    # 2) 导出 ONNX（固定 batch=1，与 373 导出规格一致）
    onnx_path = os.path.join(a.out_dir, "model.onnx")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        net, dummy, onnx_path, opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes=None,
    )
    print(f"ONNX 导出 -> {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.2f} MB)")

    # 3) 数值对齐：PyTorch vs ORT
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst, worst_arg = 0.0, True
    for _ in range(a.samples):
        x = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()[0]
        got = sess.run(None, {"input": x})[0][0]
        d = float(np.abs(ref - got).max())
        worst = max(worst, d)
        worst_arg &= int(np.argmax(ref)) == int(np.argmax(got))
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-9))
    print(f"ONNX 对齐: max|diff|={worst:.3e}  argmax一致={worst_arg}  cos={cos:.6f}")
    if worst > 1e-3 or not worst_arg:
        sys.exit("ONNX 数值对齐失败，先修导出再继续")

    # 4) onnx2tf -> float32 TFLite
    tflite_dir = os.path.join(a.out_dir, "tflite")
    os.makedirs(tflite_dir, exist_ok=True)
    cmd = ["onnx2tf", "-i", onnx_path, "-o", tflite_dir]
    print("onnx2tf 转换中…")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tfl_path = os.path.join(tflite_dir, "model_float32.tflite")
    if not os.path.exists(tfl_path):
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        sys.exit("onnx2tf 未产出 tflite")
    print(f"TFLite -> {tfl_path}  ({os.path.getsize(tfl_path)/1e6:.2f} MB)")

    # 5) TFLite 数值复验（NHWC）
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tfl_path, num_threads=1)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    print(f"TFLite 规格: 输入 {inp['shape']} {inp['dtype']}  输出 {outp['shape']}")
    xin = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    it.set_tensor(inp["index"], xin)
    it.invoke()
    tl = it.get_tensor(outp["index"])[0].copy()
    d = float(np.abs(ref - tl).max())
    cos = float(np.dot(ref, tl) / (np.linalg.norm(ref) * np.linalg.norm(tl) + 1e-9))
    print(f"TFLite 对齐: max|diff|={d:.3e}  argmax一致={int(np.argmax(ref))==int(np.argmax(tl))}  cos={cos:.6f}")

    # 6) labels 资产（学名->中文名，顺序=类别索引）+ App 换文件说明
    import csv
    zh_map = {}
    if os.path.exists(a.species_list):
        with open(a.species_list, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                zh_map[row["scientific_name"]] = row.get("common_name", "")
    lab = os.path.join(tflite_dir, "labels.txt")
    with open(lab, "w", encoding="utf-8") as f:
        for sci in species:  # ckpt["species"] 顺序即类别索引
            f.write(f"{sci}\t{zh_map.get(sci, '')}\n")
    meta = {"model": model_name, "n_species": n_cls,
            "ckpt_top1": ck.get("top1"), "input": "NHWC float32 [1,224,224,3]",
            "preprocess": "Resize256->CenterCrop224->ImageNet mean/std"}
    with open(os.path.join(a.out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("\nApp 接入（3 步）：")
    print(f"  1) 拷 {tfl_path} -> android/app/src/main/assets/models/birdnet_huabei.tflite")
    print(f"  2) 拷 {lab}     -> android/app/src/main/assets/models/labels_birdnet373.txt（覆盖）")
    print("  3) ModelRegistry: CLASSIFIER_ASSET 改 birdnet_huabei.tflite，RAW_OUTPUT_DIM="
          f"{n_cls}，NUM_SPECIES={n_cls}")


if __name__ == "__main__":
    main()

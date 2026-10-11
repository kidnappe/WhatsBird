"""YOLOX raw onnx -> float32 TFLite（复用 export_tflite.py 的 onnx2tf + 数值验证模式）。

输入必须是 export_yolox_onnx.py 产出的 *_raw.onnx（无解码，喂法 BGR 0-255 letterbox）。
raw 输出规格（probe_raw_layout.py 已验证，Kotlin 解码器照此实现）：
  [1, N, 85]，N = 52²+26²+13²(416)；通道 = [reg4(xy,wh), obj, cls80]
  anchor 顺序 P3(stride8)→P4(16)→P5(32)，每层行主序（y 外 x 内）
  decode: cx=(raw0+gx)*stride, cy=(raw1+gy)*stride, w=exp(raw2)*stride, h=exp(raw3)*stride
  xyxy = cx∓w/2, cy∓h/2（输入像素尺度），再除 letterbox ratio 映射回原图。

验证：① 随机张量 ORT vs TFLite 逐输出对齐 ② 真鸟图「dec onnx 检测」vs
「TFLite raw + numpy 解码」top1 一致且 IoU≥0.95。

Usage:
  python tools/export_detector_tflite.py --onnx "E:/code/code tools/detectors/yolox/yolox_s_512_raw.onnx"
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

STRIDES = (8, 16, 32)


def letterbox(bgr, size):
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    p = np.full((size, size, 3), 114, np.float32)
    p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
    return np.ascontiguousarray(p.transpose(2, 0, 1)[None]), r


def decode_raw(raw, size, score_th=0.35, cls_idx=14):
    """raw [1,N,C] -> [(score, x1,y1,x2,y2)]（输入像素尺度，bird 类，含 NMS 0.45）。

    C=85：A 路线 COCO 80 类，bird 列 = 5+14；C=6：B 路线单类，bird 列 = 5+0。
    """
    c = raw[0]
    n = c.shape[0]
    ch = c.shape[1]
    assert ch in (6, 85), f"raw 通道数 {ch} 不在已知规格 (6/85)"
    cls_off = 5 + cls_idx
    grids, counts = [], []
    for st in STRIDES:
        hw = size // st
        gy, gx = np.mgrid[0:hw, 0:hw]
        grids.append(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))
        counts.append(hw * hw)
    assert n == sum(counts), (n, counts)
    g = np.concatenate(grids, 0)
    st = np.repeat(np.array(STRIDES, np.float32), counts)
    xy = (c[:, :2] + g) * st[:, None]
    wh = np.exp(c[:, 2:4]) * st[:, None]
    obj = c[:, 4]
    bird = obj * c[:, cls_off]
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
            if u > 0 and ix * iy / u >= 0.45:
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


def align_layout(got, ref):
    """onnx2tf 可能转置输出（如 [1,85,N] vs [1,N,85]）——形状感知对齐，禁止盲 reshape。"""
    if got.shape == ref.shape:
        return got
    if got.ndim == 3 and got.transpose(0, 2, 1).shape == ref.shape:
        return got.transpose(0, 2, 1)
    raise SystemExit(f"无法对齐输出布局: got {got.shape} vs ref {ref.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--nclasses", type=int, default=80,
                    help="raw 输出通道 = 5+nclasses：A 路线 80 类（默认），B 单类传 1")
    a = ap.parse_args()
    cls_idx = a.nclasses - 1
    size = int(json.loads(os.path.basename(a.onnx).split("_")[2]))
    name = os.path.splitext(os.path.basename(a.onnx))[0]
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"尺寸 {size}，输出 {a.out_dir}")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    # 2) onnx2tf -> float32 TFLite（产物名 = <onnx名>_float32.tflite）
    cand = [os.path.join(a.out_dir, name + "_float32.tflite"),
            os.path.join(a.out_dir, "model_float32.tflite")]
    tfl_path = next((p for p in cand if os.path.exists(p)), None)
    if tfl_path:
        print(f"已有 TFLite，跳过转换: {tfl_path}")
    else:
        cmd = ["onnx2tf", "-i", a.onnx, "-o", a.out_dir]
        print("onnx2tf 转换中…")
        r = subprocess.run(cmd, capture_output=True, text=True)
        tfl_path = next((p for p in cand if os.path.exists(p)), None)
        if not tfl_path:
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
            sys.exit("onnx2tf 未产出 tflite")
    print(f"TFLite -> {tfl_path}  ({os.path.getsize(tfl_path)/1048576:.2f} MB)")

    # 3) TFLite 复验：随机张量对齐（NHWC 输入）
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tfl_path, num_threads=1)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    print(f"TFLite 输入: {inp['shape']} {inp['dtype']}  输出: "
          f"{[(o['shape'], o['dtype']) for o in it.get_output_details()]}")
    worst = 0.0
    for _ in range(a.samples):
        x = (rng.random((1, 3, size, size)).astype(np.float32) * 255.0)
        ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        it.set_tensor(inp["index"], np.ascontiguousarray(np.transpose(x, (0, 2, 3, 1))))
        it.invoke()
        outs = it.get_output_details()
        assert len(outs) == 1, [o["shape"] for o in outs]
        got = it.get_tensor(outs[0]["index"])
        got = align_layout(got, ref)
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"TFLite 随机张量对齐: max|diff|={worst:.3e}")
    if worst > 1e-2:
        sys.exit("TFLite 数值对齐失败")

    # 4) 真鸟图解码一致性：dec onnx 检测 vs TFLite raw + numpy 解码
    with open(r"E:/code/code tools/data/coco/annotations/instances_val2017.json") as f:
        ann = json.load(f)
    bird_cat = [c["id"] for c in ann["categories"] if c["name"] == "bird"][0]
    imgs = {im["id"]: im for im in ann["images"]}
    gt = {}
    for an in ann["annotations"]:
        if an["category_id"] == bird_cat and not an.get("iscrowd"):
            gt.setdefault(imgs[an["image_id"]]["file_name"], []).append(an["bbox"])
    dec_sess = ort.InferenceSession(
        os.path.join(r"E:/code/code tools/detectors/yolox", name.replace("_raw", "_dec") + ".onnx"),
        so, providers=["CPUExecutionProvider"])
    ok = 0
    total = 0
    for fn in sorted(gt.keys())[:10]:
        bgr = cv2.imread(os.path.join(r"E:/code/code tools/data/coco/val2017", fn))
        if bgr is None:
            continue
        x, ratio = letterbox(bgr, size)
        ref_dets = dec_sess.run(None, {dec_sess.get_inputs()[0].name: x})[0][0]
        bird = ref_dets[:, 4] * ref_dets[:, 5 + cls_idx]
        k = int(np.argmax(bird))
        if float(bird[k]) < 0.3:
            continue  # 无真检出（top 分是噪声锚点），top1 比较无意义，跳过
        total += 1
        rb = ref_dets[k, :4]  # 输入像素尺度（不除 ratio），与 decode_raw 输出同坐标系
        ref_box = (rb[0] - rb[2] / 2, rb[1] - rb[3] / 2, rb[0] + rb[2] / 2, rb[1] + rb[3] / 2)
        it.set_tensor(inp["index"], np.ascontiguousarray(np.transpose(x, (0, 2, 3, 1))))
        it.invoke()
        got_t = it.get_tensor(it.get_output_details()[0]["index"])
        n = ref_dets.shape[0]
        got = align_layout(got_t, np.zeros((1, n, ref_dets.shape[2]), np.float32)).reshape(1, n, ref_dets.shape[2])
        dets = decode_raw(got, size, cls_idx=cls_idx)
        if not dets:
            continue  # total 已计，无检出即失败
        best = max(dets, key=lambda d: d[0])
        i = iou(best, (0, *ref_box))
        ok += int(i >= 0.95)
    print(f"解码一致性: top1 IoU≥0.95 的图 {ok}/{total}")
    if ok < total - 1:
        sys.exit("解码一致性不达标，检查布局")

    meta = {"onnx": a.onnx, "size": size, "tflite": tfl_path,
            "preprocess": "letterbox(114) BGR 0-255 不归一化, 输入 NHWC float32",
            "raw_layout": "[1,N,C] reg4(xy,wh),obj,cls(C-5)；C=85(A 80类) / C=6(B 单类)；"
                          "P3/P4/P5 行主序; "
                          "cx=(raw0+gx)*st, cy=(raw1+gy)*st, w=exp(raw2)*st, h=exp(raw3)*st",
            "bird_class_idx": cls_idx}
    with open(os.path.join(a.out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("完成。")


if __name__ == "__main__":
    sys.exit(main())

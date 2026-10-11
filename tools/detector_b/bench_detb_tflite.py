# -*- coding: utf-8 -*-
r"""B 检测器 int8 tflite 复测：与 fp32 dec-onnx 基线同口径比掉点（README-B-LOCAL §5 闸门）。

喂法：NHWC float32 letterbox(114) BGR 0-255（= YoloDetector.kt L21 口径）。
解码：raw [1,N,6] P3/P4/P5 行主序（export_detector_tflite.decode_raw，含 NMS 0.45），
      框输入尺度 /ratio 回原图。
落盘：检出按 score≥0.20 存 JSON，0.35/0.45 指标离线同表打印（与 rescore_stored_dets 对齐）。

Usage: python bench_detb_tflite.py <model.tflite> <tag> [--skip-region]
产出:  data/detector_b/results/tflite_<tag>.json + tflite_<tag>_eval.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, r"E:\code\observebird\tools")
import bench_detector_a as A  # noqa: E402
from export_detector_tflite import decode_raw  # noqa: E402

RESULTS = r"E:\code\code tools\data\detector_b\results"
STORE_TH = 0.20
THS = (0.35, 0.45)


def make_session(tflite_path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=tflite_path, num_threads=8)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    size = int(inp["shape"][1])
    assert inp["dtype"] == A.np.float32, inp["dtype"]
    assert tuple(inp["shape"]) == (1, size, size, 3), inp["shape"]
    return it, inp["index"], size


def detect(it, inp_idx, size, bgr, score_th=STORE_TH):
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    p = A.np.full((size, size, 3), 114, A.np.float32)
    p[:nh, :nw] = A.cv2.resize(bgr, (nw, nh)).astype(A.np.float32)
    it.set_tensor(inp_idx, p[None])          # NHWC
    it.invoke()
    got = it.get_tensor(it.get_output_details()[0]["index"])
    if got.ndim == 3 and got.shape[1] < got.shape[2]:
        # flatbuffer_direct 把 onnx [1,N,C] 按布局变换成 [1,C,N]（现役 s512 同布局），
        # decode_raw 吃 [1,N,C] → 转回来
        got = A.np.ascontiguousarray(got.transpose(0, 2, 1))
    dets = decode_raw(got, size, score_th=score_th, cls_idx=0)  # B 单类
    return [(s, x1 / r, y1 / r, x2 / r, y2 / r) for s, x1, y1, x2, y2 in dets]


def eval_coco(it, inp_idx, size, gt):
    per = {"tiny<0.25%": [0, 0], "small<1%": [0, 0], "large>=1%": [0, 0]}
    tot = {th: {"gt": 0, "hit": 0, "fp": 0} for th in THS}
    stored, times = {}, []
    for k, (fn, e) in enumerate(sorted(gt.items())):
        bgr = A.cv2.imread(os.path.join(A.COCO_DIR, "val2017", fn))
        if bgr is None:
            continue
        t0 = time.perf_counter()
        dets = detect(it, inp_idx, size, bgr)
        times.append(time.perf_counter() - t0)
        stored[fn] = [[round(s, 4)] + [round(v, 4) for v in box] for s, *box in dets]
        for th in THS:
            dth = [d for d in dets if d[0] >= th]
            hit, fp, matched = A.match_and_score(dth, e["boxes"], e["wh"])
            tot[th]["gt"] += len(e["boxes"])
            tot[th]["hit"] += hit
            tot[th]["fp"] += fp
            for gi, g in enumerate(e["boxes"]):
                frac = g[2] * g[3] / (e["wh"][0] * e["wh"][1])
                b = A.size_bucket(frac)
                per[b][1] += 1
                if matched[gi]:
                    per[b][0] += 1
        if (k + 1) % 50 == 0:
            print(f"  coco {k + 1}/{len(gt)}", flush=True)
    out = {"images": len(stored), "gt": tot[THS[0]]["gt"], "dets": stored,
           "ms_per_img": round(float(A.np.mean(times)) * 1000, 1)}
    for th in THS:
        t, n = tot[th], len(stored)
        out[f"th{th}"] = {
            "recall": round(t["hit"] / max(t["gt"], 1), 4),
            "fp_per_img": round(t["fp"] / max(n, 1), 3),
            "by_size": {b: {"hit": h, "total": m, "recall": round(h / max(m, 1), 4)}
                        for b, (h, m) in per.items()},
        }
    return out


def eval_region(it, inp_idx, size):
    import csv
    eval_csv = r"E:\code\code tools\data\detb_eval\eval_set.csv"
    with open(eval_csv, encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    tot = {th: {"n_img": 0, "n_det": 0, "cnt": 0} for th in THS}
    stored, times = {}, []
    for k, itrow in enumerate(items):
        bgr = A.cv2.imread(itrow["path"])
        if bgr is None:
            continue
        t0 = time.perf_counter()
        dets = detect(it, inp_idx, size, bgr)
        times.append(time.perf_counter() - t0)
        stored[itrow["file"]] = {"region": itrow["region"],
                                 "dets": [[round(s, 4)] + [round(v, 4) for v in box]
                                          for s, *box in dets]}
        for th in THS:
            d = [x for x in dets if x[0] >= th]
            tot[th]["n_img"] += 1
            if d:
                tot[th]["n_det"] += 1
                tot[th]["cnt"] += len(d)
        if (k + 1) % 100 == 0:
            print(f"  region {k + 1}/{len(items)}", flush=True)
    out = {"images": len(stored), "dets": stored,
           "ms_per_img": round(float(A.np.mean(times)) * 1000, 1)}
    for th in THS:
        s, n = tot[th], max(tot[th]["n_img"], 1)
        out[f"th{th}"] = {"det_rate": round(s["n_det"] / n, 4),
                          "dets_per_img": round(s["cnt"] / n, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite")
    ap.add_argument("tag")
    ap.add_argument("--skip-region", action="store_true")
    a = ap.parse_args()

    it, inp_idx, size = make_session(a.tflite)
    print(f"{a.tag}: 输入 {it.get_input_details()[0]['shape']} "
          f"输出 {it.get_output_details()[0]['shape']}")

    gt = A.load_bird_gt()
    print(f"=== [{a.tag}] COCO bird 子集 (int8 tflite) ===")
    coco = eval_coco(it, inp_idx, size, gt)
    json.dump(coco, open(os.path.join(RESULTS, f"tflite_{a.tag}.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    if not a.skip_region:
        print(f"=== [{a.tag}] 四区 400 (int8 tflite) ===")
        reg = eval_region(it, inp_idx, size)
        json.dump(reg, open(os.path.join(RESULTS, f"tflite_{a.tag}_eval.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n===== [{a.tag}] int8 tflite 汇总（对照 fp32 onnx 用 rescore_stored_dets）=====")
    for th in THS:
        c = coco[f"th{th}"]
        line = (f"th{th}: recall {c['recall']:.3f} "
                f"tiny {c['by_size']['tiny<0.25%']['recall']:.3f} "
                f"small {c['by_size']['small<1%']['recall']:.3f} "
                f"FP/图 {c['fp_per_img']:.2f}")
        if not a.skip_region:
            line += f" | 四区 {reg[f'th{th}']['det_rate'] * 100:.1f}% / {reg[f'th{th}']['dets_per_img']:.2f}框"
        print(line)
    print(f"ms/图 coco {coco['ms_per_img']}"
          + (f" 四区 {reg['ms_per_img']}" if not a.skip_region else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

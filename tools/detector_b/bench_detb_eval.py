# -*- coding: utf-8 -*-
r"""四区独立评测集 bench：读冻结清单 detb_eval/eval_set.csv，逐图检测，算检出率。

这是 FP 归因后的关键一步（docs/handoff/detector_b/FP_ISSUE_ARCHIVE.md 处理方案 #3）：
  COCO 的 FP 大部分是"群像图 GT 漏标真鸟"的评测假象；
  这 400 张是**真实观鸟场景**（城市/公园，B 检测器 0 泄漏、无群像偏置），
  在它上面测出的检出率/FP 率，才是贴近真实使用的数字。

与 eval_huabei 的区别：
  * 数据源 = 冻结清单（四区各 100，按 GBIF key 对训练集去重）
  * 不是"华北 200 张"（78% 已被 B 训过，泄漏口径）
  * 有 GT 吗：无 —— 只算「有没有框出鸟」的检出率 + 每图框数分布（FP 观测）
  * 置信度分布也记录下来 —— 供定阈值（0.35 vs 0.45/0.55）用

用法：
  python bench_detb_eval.py <dec.onnx> <tag> [--score-th 0.35]
结果：
  data/detector_b/results/eval_set_<tag>.json + 控制台汇总
"""
import argparse
import csv
import json
import os
import sys
import time

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"E:\code\observebird\tools")
import bench_detector_a as A  # noqa: E402

EVAL_CSV = r"E:\code\code tools\data\detb_eval\eval_set.csv"
RESULTS = r"E:\code\code tools\data\detector_b\results"
THS = (0.35, 0.45, 0.55, 0.65, 0.75)
STORE_TH = 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("tag")
    ap.add_argument("--score-th", type=float, default=0.35,
                    help="主口径（App 的 MIN_DET_SCORE 候选值）")
    a = ap.parse_args()

    if not os.path.exists(EVAL_CSV):
        print("✗ 冻结清单不存在：%s\n  先跑 make_detb_eval_set.py --force" % EVAL_CSV)
        return 1

    with open(EVAL_CSV, encoding="utf-8-sig", newline="") as f:
        items = list(csv.DictReader(f))
    print("冻结清单：%d 张（四区各 %d）"
          % (len(items), len(items) // max(1, len(set(i["region"] for i in items)))))

    # 与 bench_detector_b.BRunner 同款：单类 6 通道解码
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = int(inp.shape[2])
    inp_name = inp.name
    print("模型输入 %s  输出 %s" % (inp.shape, sess.get_outputs()[0].shape))

    out_json = os.path.join(RESULTS, "eval_set_%s.json" % a.tag)
    stored = {}
    times = []
    # 每区/每阈值统计
    reg_stat = {}
    th_stat = {th: {"n_img": 0, "n_det": 0, "cnt": 0, "scores": []} for th in THS}
    all_scores = []          # 全部框分数（画分布用）
    n_done = 0
    t0 = time.time()

    for k, it in enumerate(items):
        bgr = A.cv2.imread(it["path"])
        if bgr is None:
            print("  读图失败：%s" % it["path"])
            continue
        t1 = time.perf_counter()
        x, ratio = A.letterbox(bgr, size)
        out = sess.run(None, {inp_name: x})[0][0]        # [N,6] cx,cy,w,h,obj,cls
        bird = out[:, 4] * out[:, 5]
        m = bird >= STORE_TH
        rows = out[m]
        sc = bird[m]
        dets = []
        for i in range(len(rows)):
            cx, cy, bw, bh = (float(v) for v in rows[i, :4] / ratio)
            dets.append((float(sc[i]), cx - bw / 2, cy - bh / 2,
                         cx + bw / 2, cy + bh / 2))
        dets = A.nms(dets, A.YOLOX_NMS_IOU)
        times.append(time.perf_counter() - t1)

        reg = it["region"]
        rs = reg_stat.setdefault(reg, {th: {"n_img": 0, "n_det": 0} for th in THS})
        stored[it["file"]] = {"region": reg,
                              "dets": [[round(dd[0], 4)] + [round(v, 4) for v in dd[1:]]
                                       for dd in dets]}
        for th in THS:
            d = [x for x in dets if x[0] >= th]
            rs[th]["n_img"] += 1
            th_stat[th]["n_img"] += 1
            if d:
                rs[th]["n_det"] += 1
                th_stat[th]["n_det"] += 1
                th_stat[th]["cnt"] += len(d)
                th_stat[th]["scores"].extend(x[0] for x in d)
        all_scores.extend(x[0] for x in dets)
        n_done += 1
        if (k + 1) % 100 == 0:
            print("  %d/%d  %.1fs" % (k + 1, len(items), time.time() - t0), flush=True)

    # 汇总
    per_region = {}
    for reg, rs in reg_stat.items():
        per_region[reg] = {}
        for th in THS:
            s = rs[th]
            per_region[reg][f"th{th}"] = {
                "det_rate": round(s["n_det"] / max(s["n_img"], 1), 4)}
    per_region["all"] = {}
    for th in THS:
        s = th_stat[th]
        n = max(s["n_img"], 1)
        scores = sorted(s["scores"], reverse=True)
        per_region["all"][f"th{th}"] = {
            "det_rate": round(s["n_det"] / n, 4),
            "dets_per_img": round(s["cnt"] / n, 3),
            "boxes_ge_th": s["n_det"],
        }
    # 置信度分布（所有框，0.2+）：给定阈值提供依据
    buckets = {}
    for lo in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        hi = lo + 0.1
        buckets["%.1f-%.1f" % (lo, hi)] = sum(1 for s in all_scores if lo <= s < hi)

    out = {
        "tag": a.tag, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "set": EVAL_CSV, "images": n_done,
        "score_th_main": a.score_th,
        "no_gt_note": "无 GT：只算检出率/框数分布，不算召回。FP 与检出在此口径下不可分——"
                      "本数据集均为真实鸟类照片，单图多框=过度检出或真多鸟，需人工抽检区分",
        "per_region": per_region,
        "score_hist": buckets,
        "ms_per_img": round(float(A.np.mean(times)) * 1000, 1),
        "dets": stored,
    }
    json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 控制台汇总
    print()
    print("=" * 66)
    print(" 四区独立评测（%d 张，B 检测器 0 泄漏，无群像偏置）" % n_done)
    print("=" * 66)
    print("  %-10s" % "区", "".join("%9s" % ("th%.2f" % th) for th in THS), "  (检出率)")
    for reg in list(reg_stat.keys()) + ["all"]:
        line = "  %-10s" % reg
        for th in THS:
            v = per_region[reg][f"th{th}"]["det_rate"]
            line += "%9.1f%%" % (v * 100)
        print(line)
    print()
    print("  每图框数（all）：", "  ".join(
        "th%.2f=%.2f" % (th, per_region["all"][f"th{th}"]["dets_per_img"])
        for th in THS))
    print("  ms/图：%.1f" % out["ms_per_img"])
    print()
    print("  置信度分布（全部框，0.2+，共 %d 个）：" % len(all_scores))
    for k, v in buckets.items():
        bar = "#" * min(50, v)
        print("    %s %-6d %s" % (k, v, bar))
    th_m = a.score_th
    print()
    print("  主口径 th=%.2f：检出率 %.1f%% | 每图 %.2f 框"
          % (th_m, per_region["all"][f"th{th_m}"]["det_rate"] * 100,
             per_region["all"][f"th{th_m}"]["dets_per_img"]))
    print("  结果: %s" % out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
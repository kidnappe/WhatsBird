# -*- coding: utf-8 -*-
r"""离线扫 NMS IoU —— 不重跑推理，纯拿 ep35 已落盘的 dets 重算 FP/图。

为什么改 NMS 而不是改 score 阈值（ep35 扫描的教训）：
  提阈值治 FP 的同时把 tiny 砍半（0.263→0.115）。因为 FP 的分数**也高**
  （0.55 才滤得掉），说明它们不是"勉强越线的垃圾框"。
  NMS 作用不同：它砍的是**同一只鸟的重复框**——那几只的分数都高，
  用阈值滤不掉，但按 IoU 合并掉后**不伤真阳**。所以 NMS 有机会两头兼得。

输入：results/<tag>.json（bench 落盘，含每图全部框+分数）
输出：data/detector_b/nms_sweep_<tag>.{json,csv}

用法：python sweep_nms_iou.py <tag>            # 默认扫 0.20..0.60
      python sweep_nms_iou.py <tag> --limit 0
"""
import argparse
import json
import math
import os
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"E:\code\observebird\tools")
import bench_detector_a as A  # noqa: E402

RESULTS = r"E:\code\code tools\data\detector_b\results"
OUT_DIR = r"E:\code\code tools\data\detector_b"
THRESH = {"tiny": 0.15, "small": 0.70, "recall": 0.45, "fp_per_img": 0.30}
STORE_TH = 0.20          # bench 落盘阈值
SCORE_TH = 0.35          # 主口径


def nms(dets, iou_th):
    """复用 bench_detector_a.nms（与验收同口径）。"""
    return A.nms(dets, iou_th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ious", default="0.20,0.25,0.30,0.35,0.40,0.45,0.55,0.60")
    a = ap.parse_args()

    p = os.path.join(RESULTS, a.tag + ".json")
    if not os.path.exists(p):
        print("缺结果文件: %s" % p)
        return 1
    d = json.load(open(p, encoding="utf-8"))
    dets_all = d["coco"]["dets"]           # {img: [[score,x1,y1,x2,y2],...]}
    gt = A.load_bird_gt()

    ious = [float(x) for x in a.ious.split(",") if x.strip()]
    print("=== [%s] NMS IoU 离线扫描（score 主口径 %.2f，落盘阈值 %.2f）==="
          % (a.tag, SCORE_TH, STORE_TH))
    print("  %-6s %8s %8s %8s %9s" % ("NMS", "recall", "tiny", "small", "FP/图"))
    print("  " + "-" * 46)

    rows = []
    for iou_th in ious:
        tot_hit = tot_fp = tot_gt = 0
        per = {"tiny<0.25%": [0, 0], "small<1%": [0, 0], "large>=1%": [0, 0]}
        for fn, raw in dets_all.items():
            e = gt.get(fn)
            if e is None:
                continue
            dets = [(r[0], r[1], r[2], r[3], r[4]) for r in raw if r[0] >= STORE_TH]
            kept = nms(dets, iou_th)
            kept = [k for k in kept if k[0] >= SCORE_TH]
            h, f, matched = A.match_and_score(kept, e["boxes"], e["wh"])
            tot_hit += h
            tot_fp += f
            tot_gt += len(e["boxes"])
            for gi, g in enumerate(e["boxes"]):
                b = A.size_bucket(g[2] * g[3] / (e["wh"][0] * e["wh"][1]))
                per[b][1] += 1
                if matched[gi]:
                    per[b][0] += 1
        nimg = len(dets_all)
        recall = tot_hit / max(tot_gt, 1)
        tiny = per["tiny<0.25%"][0] / max(per["tiny<0.25%"][1], 1)
        small = per["small<1%"][0] / max(per["small<1%"][1], 1)
        fpi = tot_fp / max(nimg, 1)
        ok_fp = fpi <= THRESH["fp_per_img"]
        ok_tiny = tiny >= THRESH["tiny"]
        flag = "  <<< 两项均达标!" if (ok_fp and ok_tiny) else ""
        print("  %-6.2f %8.3f %8.3f %8.3f %9.2f%s"
              % (iou_th, recall, tiny, small, fpi, flag))
        rows.append({"nms_iou": iou_th, "recall": recall, "tiny": tiny,
                     "small": small, "fp_per_img": fpi,
                     "ok_fp": ok_fp, "ok_tiny": ok_tiny})

    # 找两头兼得的点
    good = [r for r in rows if r["ok_fp"] and r["ok_tiny"]]
    best = max(good, key=lambda r: r["recall"]) if good else None
    json.dump({"tag": a.tag, "score_th": SCORE_TH, "threshold": THRESH,
               "rows": rows,
               "best_both": best},
              open(os.path.join(OUT_DIR, "nms_sweep_%s.json" % a.tag), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    print("结果: %s" % os.path.join(OUT_DIR, "nms_sweep_%s.json" % a.tag))
    if best:
        print("★ tiny&FP 双达标: NMS=%.2f recall=%.3f tiny=%.3f FP=%.2f"
              % (best["nms_iou"], best["recall"], best["tiny"], best["fp_per_img"]))
    else:
        print("✗ 没有同时满足 tiny≥0.15 且 FP≤0.30 的 NMS 点")
    return 0


if __name__ == "__main__":
    sys.exit(main())
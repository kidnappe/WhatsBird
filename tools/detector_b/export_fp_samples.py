# -*- coding: utf-8 -*-
r"""从 bench 结果里挑出误检最多的图 + 每个误检框的裁剪小图，供人工看"框的是什么"。

这是 FP 归因的第 1 步（只读，不动任何模型）：把 FP 的图导出来并排好，
人工点一下就知道错误模式（脏背景 / 定位歪 / 邻框重叠 / 看着都对），
从而决定是清洗数据、换尺寸，还是改 NMS 就行。

复用 bench 的判定（match_and_score / size_bucket），保证和验收同一把尺子。

用法：
  python export_fp_samples.py ep35                # 默认主口径 0.35，按 FP 数降序取 60 张
  python export_fp_samples.py ep35 --top 60 --iou 0.45
输出：
  data/detector_b/fp_samples/ep35/
     manifest.json         每条 = {img, n_fp, fps:[{idx,score,box,size}]}
     img/<n>_<file>.jpg     整图（红框=FP 框，绿框=命中GT）
     crop/<n>_<file>_f<k>.jpg  单个误检框的放大裁剪
     index.html            网格浏览（键盘翻页）
"""
import argparse
import json
import os
import shutil
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"E:\code\observebird\tools")
import bench_detector_a as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = r"E:\code\code tools\data\detector_b\results"
COCO_DIR = os.path.join(A.TOOLS, "data", "coco")
OUT_ROOT = r"E:\code\code tools\data\detector_b\fp_samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--top", type=int, default=60, help="导多少张（按 FP 数降序）")
    ap.add_argument("--score-th", type=float, default=0.35)
    ap.add_argument("--store-th", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=None,
                    help="若给，先做一次 NMS（离线指定 IoU）再判 FP")
    a = ap.parse_args()

    p = os.path.join(RESULTS, a.tag + ".json")
    if not os.path.exists(p):
        print("缺结果文件: %s" % p)
        return 1
    d = json.load(open(p, encoding="utf-8"))
    dets_all = d["coco"]["dets"]
    gt = A.load_bird_gt()

    out = os.path.join(OUT_ROOT, a.tag)
    for sub in ("img", "crop"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    rows = []
    for fn, raw in dets_all.items():
        e = gt.get(fn)
        if e is None:
            continue
        dets = [(r[0], r[1], r[2], r[3], r[4]) for r in raw if r[0] >= a.store_th]
        if a.iou is not None:
            dets = A.nms(dets, a.iou)
        kept = [k for k in dets if k[0] >= a.score_th]
        h, f, matched = A.match_and_score(kept, e["boxes"], e["wh"])
        # 找出哪些 kept 框是 FP（没匹配上 GT）
        fps = []
        # 重跑一次贪心，标记每个框是否命中
        used_gt = [False] * len(e["boxes"])
        for k in sorted(kept, key=lambda x: -x[0]):
            box4 = (k[1], k[2], k[3], k[4])   # 去掉分数，_iou 只吃 4 个坐标
            best_i, bi = 0.5, -1
            for gi, g in enumerate(e["boxes"]):
                if used_gt[gi]:
                    continue
                gx, gy, gw, gh = g
                inter = _iou(box4, (gx, gy, gx + gw, gy + gh))
                if inter > best_i:
                    best_i, bi = inter, gi
            if bi >= 0:
                used_gt[bi] = True
            else:
                fps.append(k)
        if fps:
            rows.append({"file": fn, "n_fp": len(fps),
                         "n_hit": h,
                         "gt_boxes": [list(g) for g in e["boxes"]],
                         "fps": [{"score": float(k[0]),
                                  "box": [k[1], k[2], k[3], k[4]]}
                                 for k in fps]})

    rows.sort(key=lambda r: -r["n_fp"])
    rows = rows[:a.top]
    print("[%s] FP 图 %d 张（按 FP 数降序取前 %d），共 %d 个误检框"
          % (a.tag, len(rows), a.top, sum(r["n_fp"] for r in rows)))

    manifest = []
    for n, r in enumerate(rows, 1):
        src = os.path.join(COCO_DIR, "val2017", r["file"])
        bgr = cv2.imread(src)
        if bgr is None:
            continue
        h_, w_ = bgr.shape[:2]
        base = "%03d_%s" % (n, os.path.splitext(r["file"])[0])
        # 画框：红=FP，绿=命中GT
        vis = bgr.copy()
        for g in r.get("gt_boxes", []):
            x1, y1, x2, y2 = int(g[0]), int(g[1]), int(g[0] + g[2]), int(g[1] + g[3])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for k, fp in enumerate(r["fps"]):
            x1, y1, x2, y2 = [int(v) for v in fp["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(vis, "FP%.2f" % fp["score"], (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # 单框放大裁剪（外扩 30% 便于看上下文）
            mx, my = (x2 - x1) * 0.3, (y2 - y1) * 0.3
            cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            cx2, cy2 = min(w_, int(x2 + mx)), min(h_, int(y2 + my))
            crop = vis[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            sc = 320.0 / max(crop.shape[1], 1)
            crop = cv2.resize(crop, (320, max(1, int(crop.shape[0] * sc))))
            cv2.imwrite(os.path.join(out, "crop", "%s_f%d.jpg" % (base, k)), crop)
        cv2.imwrite(os.path.join(out, "img", base + ".jpg"), vis)
        manifest.append({"n": n, "file": r["file"], "n_fp": r["n_fp"],
                         "img": "img/%s.jpg" % base,
                         "crops": ["crop/%s_f%d.jpg" % (base, k)
                                   for k in range(len(r["fps"]))]})
        print("  %s  FP=%d" % (base, r["n_fp"]))

    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    _write_index(out, manifest)
    print("\n完成: %s" % out)
    print("用浏览器打开 index.html，或双击 img/ 下任意一张")
    return 0


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def _write_index(out, manifest):
    html = ["<!doctype html><meta charset=utf-8>",
            "<title>FP 误检归因</title>",
            "<style>body{background:#111;color:#ddd;font:14px sans-serif}",
            ".g{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:10px}",
            "img{width:100%;border-radius:6px}.c{margin-top:4px}",
            ".fp{color:#f87171;font-weight:700}</style>",
            "<h3>误检归因 · 点框看它框的是什么</h3>",
            "<div class=g>"]
    for m in manifest:
        html.append("<div><img src='%s'><div class=fp>FP=%d</div>"
                    "<div class=c>%s</div></div>" % (m["img"], m["n_fp"], m["file"]))
    html.append("</div>")
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write("\n".join(html))


if __name__ == "__main__":
    sys.exit(main())
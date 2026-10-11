# -*- coding: utf-8 -*-
r"""从 bench 落盘的 dets（th0.20+ 已 NMS）离线重评分：补任意阈值的 COCO 口径指标。

背景：bench_detector_a.eval_coco 只打印 th0.35/0.25 两个点，但发货阈值候选是
0.45（HANDOFF-DETECTOR-B-POSTTRAIN §3.5）。检测结果已按 0.20 落盘到结果 JSON，
无需重跑推理即可按新阈值重算 recall/tiny/small/FP。

正确性自证：对已有结果按 0.35 重算，应逐项复现 bench 打印值（对不上=口径有 bug）。

Usage:
  python rescore_stored_dets.py <result.json> [<result.json> ...] --ths 0.35 0.45
"""
import argparse
import json
import os
import sys

sys.path.insert(0, r"E:\code\observebird\tools")
import bench_detector_a as A  # noqa: E402


def score_file(path, gt, th):
    """对单个结果 JSON 按 th 重算，返回 (recall, tiny, small, large, fp_per_img, n_img)。"""
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    coco = res["coco"] if "coco" in res else res  # bench_detector_b 落盘无 coco 包裹层
    per = {"tiny<0.25%": [0, 0], "small<1%": [0, 0], "large>=1%": [0, 0]}
    tot = {"gt": 0, "hit": 0, "fp": 0}
    n_img = 0
    for fn, dets in coco["dets"].items():
        e = gt.get(fn)
        if e is None:
            continue
        n_img += 1
        dth = [d for d in dets if d[0] >= th]
        hit, fp, matched = A.match_and_score(dth, e["boxes"], e["wh"])
        tot["gt"] += len(e["boxes"])
        tot["hit"] += hit
        tot["fp"] += fp
        for gi, g in enumerate(e["boxes"]):
            frac = g[2] * g[3] / (e["wh"][0] * e["wh"][1])
            b = A.size_bucket(frac)
            per[b][1] += 1
            if matched[gi]:
                per[b][0] += 1
    return {
        "recall": tot["hit"] / max(tot["gt"], 1),
        "tiny": per["tiny<0.25%"][0] / max(per["tiny<0.25%"][1], 1),
        "small": per["small<1%"][0] / max(per["small<1%"][1], 1),
        "large": per["large>=1%"][0] / max(per["large>=1%"][1], 1),
        "fp_per_img": tot["fp"] / max(n_img, 1),
        "n_img": n_img,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="bench 结果 JSON（含 coco.dets，th0.20 落盘）")
    ap.add_argument("--ths", type=float, nargs="+", default=[0.35, 0.45])
    ap.add_argument("--out", default=None,
                    help="把重评分结果落成 JSON（2026-10-09 新增）。"
                         "此前本脚本只 print，文档里 th0.45 那一整列数字没有任何落盘出处，"
                         "选型表的核心指标因此不可复现——加上这个开关后务必用它。")
    args = ap.parse_args()

    gt = A.load_bird_gt()
    dump = {}
    for th in args.ths:
        print(f"\n===== COCO 口径 score>={th}（离线重评分）=====")
        print(f"{'model':<16}{'recall':>8}{'tiny':>8}{'small':>8}{'large':>8}{'FP/图':>8}")
        dump[f"th{th:g}"] = {}
        for p in args.results:
            r = score_file(p, gt, th)
            name = os.path.splitext(os.path.basename(p))[0]
            print(f"{name:<16}{r['recall']:>8.3f}{r['tiny']:>8.3f}{r['small']:>8.3f}"
                  f"{r['large']:>8.3f}{r['fp_per_img']:>8.2f}   ({r['n_img']} 图)")
            dump[f"th{th:g}"][name] = r
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"source_results": args.results, "ths": args.ths,
                       "protocol": "COCO bird 子集，离线对 th0.20 落盘的 dets 重评分",
                       "scores": dump}, f, ensure_ascii=False, indent=1)
        print(f"\n-> 已落盘 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

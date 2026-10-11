# -*- coding: utf-8 -*-
"""B 模型阈值扫描：找 FP/图≤0.3 与召回的交叉点。

bench 的 stored dets 落在 results/<name>.json 里（落盘阈值 0.2），可以直接离线复盘，
不必重跑推理。产出表：score 阈值 × {recall, tiny, small, FP/图}。

跑法：python sweep_b_th.py bird512_ep5
"""
import json
import os
import sys

sys.path.insert(0, r"E:\code\observebird\tools")
import bench_detector_a as A  # noqa: E402

RES = os.path.join(A.TOOLS, "data", "detector_b", "results")
THS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


def sweep(name):
    path = os.path.join(RES, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        res = json.load(f)

    files = sorted(res["coco"]["dets"].keys())
    # COCO GT 按同一口径重载（bench 用 load_bird_gt，顺序/过滤一致）
    gt = A.load_bird_gt()

    print(f"\n=== [{name}] 阈值扫描（离线复盘，COCO bird 子集）===")
    print(f"{'th':>6}{'recall':>9}{'tiny':>8}{'small':>8}{'large':>8}{'FP/图':>8}{'img检出':>9}")

    for th in THS:
        hit = fp = img_det = 0
        tot = 0
        per = {"tiny<0.25%": [0, 0], "small<1%": [0, 0], "large>=1%": [0, 0]}
        for fn in files:
            e = gt.get(fn)
            if e is None:
                continue
            dets = [(d[0], d[1], d[2], d[3], d[4]) for d in res["coco"]["dets"][fn] if d[0] >= th]
            h, f, matched = A.match_and_score(dets, e["boxes"], e["wh"])
            hit += h
            fp += f
            tot += len(e["boxes"])
            if dets:
                img_det += 1
            for gi, g in enumerate(e["boxes"]):
                b = A.size_bucket(g[2] * g[3] / (e["wh"][0] * e["wh"][1]))
                per[b][1] += 1
                if matched[gi]:
                    per[b][0] += 1

        recall = hit / max(tot, 1)
        fpi = fp / max(len(files), 1)
        print(f"{th:>6.2f}{recall:>9.3f}"
              f"{per['tiny<0.25%'][0] / max(per['tiny<0.25%'][1], 1):>8.3f}"
              f"{per['small<1%'][0] / max(per['small<1%'][1], 1):>8.3f}"
              f"{per['large>=1%'][0] / max(per['large>=1%'][1], 1):>8.3f}"
              f"{fpi:>8.2f}{img_det / max(len(files), 1):>9.3f}")

    print("\n参考：现役 s512 fp32 @0.35 → recall 0.368 / tiny 0.072 / small 0.551 / FP 0.33")
    print("      B 验收线        → tiny ≥0.15 / small ≥0.70 / FP ≤0.30")


if __name__ == "__main__":
    sweep(sys.argv[1] if len(sys.argv) > 1 else "bird512_ep5")
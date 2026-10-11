# -*- coding: utf-8 -*-
"""汇总 crop_regime/results.json：只看部署口径（frac=2.0，即端上 BOX_EXPAND_RATIO=4.0）
下 prob/rank × alpha 的 top1/top5，并给出每区最优，与整图口径的结论对照。

用法：python tools/summarize_crop_prior.py [--frac 2.0]
"""
import argparse
import json
import os

RESULTS = r"E:\code\code tools\data\crop_regime\results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", default="2.0", help="脚本外扩比例；2.0 = 端上 BOX_EXPAND_RATIO 4.0")
    a = ap.parse_args()
    f = a.frac
    j = json.load(open(RESULTS, encoding="utf-8"))
    print(f"ratios={j['ratios']}  det_th={j['det_th']}  prior={j.get('prior')}")
    print(f"取 frac={f}（端上 BOX_EXPAND_RATIO = 2×frac = {float(f)*2:.1f}）\n")

    head = f"{'区':<10}{'口径':<16}{'top1':>8}{'top5':>8}"
    print(head)
    print("-" * len(head))
    summary = {}
    for region, d in j["regions"].items():
        base = d.get(f, {})
        rows = []
        if base:
            rows.append(("-", base))
            print(f"{region:<10}{'无重排':<16}{base['top1']:>7.1f}%{base['top5']:>7.1f}%")
        best = None
        for k, v in d.items():
            if "@" not in k or not k.endswith("|" + f):
                continue
            print(f"{'':<10}{k:<16}{v['top1']:>7.1f}%{v['top5']:>7.1f}%")
            if best is None or v["top1"] > best[1]:
                best = (k, v["top1"], v["top5"])
        if best:
            summary[region] = (base.get("top1") if base else None, best)
    print("\n=== 部署口径下每区最优 ===")
    print(f"{'区':<10}{'无重排':>9}{'最优设置':>18}{'top1':>9}{'增益':>9}")
    for region, (b, best) in summary.items():
        gain = f"{best[1]-b:+.2f}pp" if b is not None else "n/a"
        print(f"{region:<10}{(f'{b:.1f}%' if b else 'n/a'):>9}{best[0]:>18}{best[1]:>8.1f}%{gain:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
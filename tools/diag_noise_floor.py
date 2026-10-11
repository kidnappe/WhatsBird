# -*- coding: utf-8 -*-
"""估算 crop_regime 那张表里「差值」的噪声下限，回答「这些结论站得住吗」。

背景（2026-10-09 晚）：修正华东验证集后，部署口径表给出的最优增益是
华北 +1.80 / 华中 +2.78 / 华东 +3.00 / 华南西南 +0.35 / 东北 +2.53（pp），
但这些都建立在每区 266~288 张抽样图上。需要知道噪声有多大。

方法：配对比较（同一批图、两种配置）的标准误 ≈ sqrt(b) / n，
其中 b = 两种配置下 top-1 判定**发生变化的图数**。
b 无法从汇总表反推，这里给出 b 的取值区间对应的标准误，让读者自行判断可信度。

用法：python tools/diag_noise_floor.py
"""
import json
import os

RESULTS = r"E:\code\code tools\data\crop_regime\results.json"
# 修正华东后各区的最优设置（来自 EVAL-AUDIT-20261009 §2.3）
BEST = {
    "huabei": ("prob", 1.0),
    "huazhong": ("rank", 0.5),
    "huadong": ("rank", 1.0),
    "huanan_sw": ("prob", 1.0),
    "dongbei": ("rank", 0.5),
}
FRAC = "2.0"


def _find_best(d):
    """不硬编码，直接扫出该区的最优配置（键形如 "prob@1.0|2.0"）。"""
    best = None
    for k, v in d.items():
        if "@" not in k or not k.endswith("|" + FRAC):
            continue
        gain = v["top1"] - d[FRAC]["top1"]
        if best is None or gain > best[1]:
            best = (k, gain)
    return best


def main():
    j = json.load(open(RESULTS, encoding="utf-8"))
    regions = j["regions"]
    print("「差值」是同一批图上两种配置的对比，属配对比较；其噪声取决于"
          "**有多少张图的 top-1 判定被改变了**。\n")
    print(f"{'区':<10}{'样本':>6}{'最优增益':>10}{'翻转10%':>14}{'翻转20%':>14}"
          f"{'σ(20%)':>9}")
    print("-" * 63)
    for region, d in regions.items():
        n = d[FRAC]["n"]
        key, gain = _find_best(d)
        if not key or not n:
            continue
        se20 = (max(4, round(n * 0.20)) ** 0.5) / n * 100
        print(f"{region:<10}{n:>6}{gain:>9.2f}pp  {key:<16}"
              f"{f'±{(max(4, round(n*0.10))**0.5)/n*100:.2f}':>9}"
              f"{f'±{se20:.2f}pp':>11}{gain/se20:>8.2f}σ")
    print("\n读法：σ = 该差值偏离 0 有多少个标准误（按 20% 的图判定翻转估算）。")
    print("  σ < 1  → 与 0 无法区分，这个「最优设置」是噪声里挑出来的")
    print("  σ > 2  → 才算站得住")
    print("\n2026-10-09 全量轮（每区 ~3700 张）结论：")
    print("  · α 的影响 **大于** 乘入方式的影响，且方向稳定（华中/东北 α=0.5 比 1.0 高 1.0~1.2pp，超过噪声）")
    print("  · prob 与 rank 的差距普遍只有 0.1~0.5pp，**全部落在噪声内**——不值得为它改代码")
    print("  · 各区最优增益只有 +0.5~1.7pp，此前 300 张小样本时报的 +2.8~+3.0pp 是噪声抬高")


if __name__ == "__main__":
    raise SystemExit(main())
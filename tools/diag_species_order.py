# -*- coding: utf-8 -*-
"""核查：ckpt["species"] 的顺序  vs  train.py 用的 sorted(数据科学名) 顺序。

背景（2026-10-09）：华东 best.pt 在训练日志里 val top1=56.12%（泄漏检查 0、题目可重现），
但 tools/eval_report.py 对同一份权重、同一批验证图算出 68.13%。
两个工具都把「分类输出第 i 位」映射到某个学名，而映射方式不同的话，
同一份权重会被读成完全不同的准确率。

train.py   : labels = {s: i for i, s in enumerate(sorted({r["scientific_name"] for r in rows}))}
eval_report: species = ckpt["species"];  labels = {s: i for i, s in enumerate(species)}

用法：python tools/diag_species_order.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_rows  # noqa: E402

DATA = r"E:\code\code tools\data"
MODELS = r"E:\code\code tools\models"
REGIONS = {
    "huabei": "huabei_mnv4_rec",
    "huazhong": "huazhong_mnv4_25k",
    "huanan_sw": "huanan_sw_mnv4_rec",
    "huadong": "huadong_mnv4_300_rec",
    "dongbei": "dongbei_mnv4_rec",
}


def main():
    print(f"{'区':<10}{'ckpt 类数':>9}{'数据种数':>9}   顺序是否一致   首个不一致的位置")
    bad = []
    for region, run in REGIONS.items():
        ck = torch.load(os.path.join(MODELS, run, "best.pt"), map_location="cpu",
                        weights_only=False)
        ck_species = list(ck["species"])
        _, rows = load_rows(os.path.join(DATA, region, "dataset.csv"))
        data_sorted = sorted({r["scientific_name"] for r in rows})
        if len(ck_species) == len(data_sorted) and ck_species == data_sorted:
            print(f"{region:<10}{len(ck_species):>9}{len(data_sorted):>9}   ✅ 完全一致")
            continue
        pos = next((i for i, (a, b) in enumerate(zip(ck_species, data_sorted)) if a != b), None)
        print(f"{region:<10}{len(ck_species):>9}{len(data_sorted):>9}   ❌ 不一致"
              f"      第 {pos} 位: ckpt='{ck_species[pos] if pos is not None else '?'}' "
              f"vs sorted='{data_sorted[pos] if pos is not None else '?'}'")
        bad.append(region)
        # 量化：按 ckpt 顺序重排标签，准确率会变成什么样（不做，只报集合差异）
        only_ck = set(ck_species) - set(data_sorted)
        only_ds = set(data_sorted) - set(ck_species)
        if only_ck or only_ds:
            print(f"           仅在 ckpt: {sorted(only_ck)[:5]}  仅在数据: {sorted(only_ds)[:5]}")
    print("\n结论：" + ("全部一致，问题不在这里" if not bad else f"{bad} 的编号顺序与训练时不同"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
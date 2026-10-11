# -*- coding: utf-8 -*-
"""诊断：华东 rec 模型训练时用的数据集路径，与当前 dataset.csv 是否同一份。

背景（2026-10-09）：华东重切验证集给出的 image_only 与训练记录差 12pp
（68.13% vs 56.12%），两份切分对不上。若 train_cfg 记录的 dataset 路径
与现在的 data/huadong/dataset.csv 不是同一个文件，解释就成立，且修法明确。

用法：python tools/diag_huadong_split.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import grouped_split, load_rows  # noqa: E402  与训练同一份 split 代码

MODELS = r"E:\code\code tools\models"
CANDIDATES = {
    "huadong": ["huadong_mnv4_300_rec", "huadong_mnv4_300"],
    "huazhong": ["huazhong_mnv4_25k"],
    "dongbei": ["dongbei_mnv4_rec"],
}


def main():
    import torch
    for region, runs in CANDIDATES.items():
        print(f"\n########## {region} ##########")
        for run in runs:
            ck_path = os.path.join(MODELS, run, "best.pt")
            if not os.path.exists(ck_path):
                continue
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            tc = ck.get("train_cfg", {})
            ds = tc.get("dataset")
            cur = os.path.join(r"E:\code\code tools\data", region, "dataset.csv")
            same = "同一个文件 ✅" if os.path.normcase(ds or "") == os.path.normcase(cur) else "**不是同一个文件** ❌"
            print(f"  {run}")
            print(f"    train_cfg.dataset = {ds}")
            print(f"    当前 dataset.csv  = {cur}")
            print(f"    -> {same}")
            print(f"    train_cfg: val_list={tc.get('val_list')} val_frac={tc.get('val_frac')} "
                  f"max_per_species={tc.get('max_per_species')} balance={tc.get('balance')}")
            print(f"    ckpt top1 = {ck.get('top1')}")

            # 用 train_cfg 记录的路径重切，看能否复现 ckpt.top1
            p = ds if ds and os.path.exists(ds) else cur
            try:
                _, rows = load_rows(p)
                tr, va = grouped_split(rows, tc.get("val_frac", 0.2))
                print(f"    按 train_cfg 路径重切: rows={len(rows)} val={len(va)} "
                      f"({len({r['scientific_name'] for r in va})} 种)")
            except Exception as e:  # noqa: BLE001
                print(f"    重切失败: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
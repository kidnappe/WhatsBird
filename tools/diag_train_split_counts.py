# -*- coding: utf-8 -*-
"""对账：训练日志里记录的 train/val 张数  vs  现在重切出来的张数。

背景（2026-10-09）：华东模型训练时 train_cfg.val_list=None（自切且未落盘）。
现在重切出来的验证集跑出 68.13%，而 ckpt.top1 记的是 56.12%，差 12pp。
train.py 在切分后会 log 一行「按 observation 分组划分: train X / val Y」，
这行就是**当时的切分规模**，是判定「两份切分是否一致」的第一手证据。

用法：python tools/diag_train_split_counts.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import grouped_split, load_rows  # noqa: E402

DATA = r"E:\code\code tools\data"
MODELS = r"E:\code\code tools\models"
REGIONS = {
    "huabei": "huabei_mnv4_rec",
    "huazhong": "huazhong_mnv4_25k",
    "huanan_sw": "huanan_sw_mnv4_rec",
    "huadong": "huadong_mnv4_300_rec",
    "dongbei": "dongbei_mnv4_rec",
}
PAT = re.compile(r"按 observation 分组划分:\s*train\s+(\d+)\s*/\s*val\s+(\d+)")
PAT2 = re.compile(r"图像\s+(\d+)\s+张\s*/\s*(\d+)\s+种")
PAT3 = re.compile(r"按冻结验证集划分:\s*train\s+(\d+)\s*/\s*val\s+(\d+)")


def main():
    print(f"{'区':<10}{'训练日志 train/val':>22}{'现在重切 train/val':>22}{'日志图像数':>12}{'现在':>10}  判定")
    for region, run in REGIONS.items():
        log = os.path.join(MODELS, run, "train.log")
        if not os.path.exists(log):
            print(f"{region:<10}  没有 train.log")
            continue
        text = open(log, encoding="utf-8", errors="replace").read()
        m = PAT.search(text) or PAT3.search(text)
        m2 = PAT2.search(text)
        if not m:
            print(f"{region:<10}  日志里没找到切分行（前 300 字: {text[:120]!r}）")
            continue
        tr_log, va_log = int(m.group(1)), int(m.group(2))
        n_log = int(m2.group(1)) if m2 else None
        ds = os.path.join(DATA, region, "dataset.csv")
        _, rows = load_rows(ds)
        tr, va = grouped_split(rows, 0.2)
        same = "✅ 完全一致" if (tr_log, va_log) == (len(tr), len(va)) else "❌ 不一致"
        print(f"{region:<10}{f'{tr_log} / {va_log}':>22}{f'{len(tr)} / {len(va)}':>22}"
              f"{(n_log if n_log else '-'):>12}{len(rows):>10}  {same}")
        if n_log is not None and n_log != len(rows):
            print(f"           ⚠️ 日志的图像数 {n_log} ≠ 现在 {len(rows)} → **数据集在训练后变过**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""检查已有的 data/inat_eval 与 data/national_eval244 两份外部评测集，能否给**区域分类器**用。

背景（2026-10-09）：华东分类器的干净验证集重建失败（两份切分差 12pp，见
docs/handoff/root/HANDOFF-20261009.md §3.1 坑 3），而 detb_eval 那 400 张经
tools/check_eval_set_clean.py 验证**100% 落在各区分类器训练集内**、完全不可用。
于是去查盘上已有的另两份"外部评测集"是否干净。

判据：按 GBIF occurrence key（编在文件名末段）与五区 dataset.csv 求交集，
      交集为 0 才算对该区零泄漏；同时报告覆盖的种数（决定能否当 150 类验收）。
用法：python tools/check_inat_eval_clean.py
"""
import csv
import os
import re
import sys
from collections import defaultdict

DATA = r"E:\code\code tools\data"
REGIONS = ["huabei", "huazhong", "huanan_sw", "huadong", "dongbei"]
KEY_RE = re.compile(r"_(\d{6,})(?:\.jpg)?$")


def key_of(name):
    m = KEY_RE.search(os.path.splitext(name)[0])
    return m.group(1) if m else None


def load_manifest(d):
    for name in ("manifest.csv", "labels.csv", "eval_set.csv"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                return name, list(csv.DictReader(f))
    return None, []


def main():
    # 各区分类器训练集的 key 全集
    train = {}
    for r in REGIONS:
        p = os.path.join(DATA, r, "dataset.csv")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            train[r] = {key_of(row["file"]) for row in csv.DictReader(f)}
        train[r].discard(None)

    for dname in ("inat_eval", "national_eval244"):
        d = os.path.join(DATA, dname)
        if not os.path.isdir(d):
            print(f"== {dname}: 不存在")
            continue
        src, rows = load_manifest(d)
        if not rows:
            print(f"== {dname}: 无 manifest/labels 可读")
            continue
        fcol = "file" if "file" in rows[0] else list(rows[0])[0]
        scol = "scientific_name" if "scientific_name" in rows[0] else None
        keys = {key_of(r[fcol]) for r in rows}
        keys.discard(None)
        species = {r[scol] for r in rows} if scol else set()
        print(f"\n== {dname}  ({src})  {len(rows)} 张 / {len(keys)} key / {len(species)} 种")
        for r in REGIONS:
            hit = len(keys & train[r])
            print(f"   vs {r:<10} 训练集({len(train[r])}): 重叠 {hit:>4}"
                  f"  {'✅ 零泄漏' if hit == 0 else '❌'}")
        if len(species) < 150:
            print(f"   ⚠️ 只覆盖 {len(species)} 种，不足以当 150 类验收集（需 150）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
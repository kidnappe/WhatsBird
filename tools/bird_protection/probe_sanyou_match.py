# -*- coding: utf-8 -*-
"""探针：骨架 384 种 ∩ 三有名录（2023）鸟类 1028 种，看匹配率与漏网。

复用 build_protection_asset.py 的 norm() / ALIASES / load_skeleton()，
保证与 protection.json 的口径完全一致（同名不同种一律不猜）。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_protection_asset import ALIASES, load_skeleton, norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BIRDS = ROOT / "tools/bird_protection/raw/sanyou_2023_birds.csv"


def load_sanyou():
    out = {}
    with BIRDS.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[norm(row["学名"])] = (row["中文名"], row["学名"])
    return out


def main():
    skeleton = load_skeleton()
    sanyou = load_sanyou()
    print("三有名录鸟类: %d 种" % len(sanyou))
    print("骨架: %d 种\n" % len(skeleton))

    hit, via_alias, miss = [], [], []
    for sci, zh in skeleton:
        if norm(sci) in sanyou:
            hit.append((sci, zh, ""))
        elif sci in ALIASES and norm(ALIASES[sci]) in sanyou:
            hit.append((sci, zh, ALIASES[sci]))
            via_alias.append((sci, zh, ALIASES[sci]))
        else:
            miss.append((sci, zh))

    print("✅ 命中三有: %d  (其中走别名 %d)" % (len(hit), len(via_alias)))
    print("❌ 不在三有: %d\n" % len(miss))

    if via_alias:
        print("走别名的：")
        for sci, zh, al in via_alias:
            print("   %-30s %-10s -> %s" % (sci, zh, al))
        print()

    print("不在三有名录的 %d 种（前 40）：" % len(miss))
    for i, (sci, zh) in enumerate(miss[:40]):
        print("   %-32s %s" % (sci, zh))
    if len(miss) > 40:
        print("   ... 另 %d 种" % (len(miss) - 40))
    return 0


if __name__ == "__main__":
    sys.exit(main())

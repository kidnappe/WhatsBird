# -*- coding: utf-8 -*-
"""探针：中国鸟类名录 v10.0 CSV 与 384 种骨架的学名匹配率。

只报告，不写产物。跑完看 unmatched 决定要不要做异名映射。
数据源见 docs/legal/LICENSES.md §D。
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "tools/bird_protection/raw/china_bird_list_v10.csv"
LABELS = ROOT / "android/app/src/main/assets/models/labels_national_v2.txt"


def norm(sci: str) -> str:
    """学名归一：去尾随标点/方括号内容/多余空白，小写。"""
    s = sci.strip().lower()
    s = re.sub(r"\[[^\]]*\]", "", s)          # 去掉 [et] 之类附注
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,;")


def load_csv():
    """返回 学名 -> (zh, iucn, cn)。只收有学名的行。"""
    out = {}
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            sci = row[1].strip()
            if not sci or " " not in sci:      # 只有拉丁双名才是物种行
                continue
            out[norm(sci)] = (row[2].strip(), row[4].strip(), row[5].strip())
    return out


def main():
    src = load_csv()
    print(f"CSV 物种行: {len(src)}")

    base = []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        base.append((p[0].strip(), p[1].strip() if len(p) > 1 else ""))

    print(f"骨架种数: {len(base)}")

    hit = miss = 0
    misses = []
    by_iucn, by_cn = {}, {}
    for sci, zh in base:
        hit_row = src.get(norm(sci))
        if hit_row is None:
            miss += 1
            misses.append((sci, zh))
            continue
        hit += 1
        _, iu, cn = hit_row
        if iu:
            by_iucn[iu] = by_iucn.get(iu, 0) + 1
        if cn and cn != "-":
            by_cn[cn] = by_cn.get(cn, 0) + 1

    print(f"\n命中 {hit} / {len(base)}   未命中 {miss}  ({miss/len(base)*100:.1f}%)")
    print("IUCN 分布:", dict(sorted(by_iucn.items(), key=lambda kv: -kv[1])))
    print("行政保护分布:", dict(sorted(by_cn.items(), key=lambda kv: -kv[1])))
    print("\n--- 未命中清单 ---")
    for sci, zh in misses:
        print(f"  {sci}\t{zh}")


if __name__ == "__main__":
    sys.exit(main())
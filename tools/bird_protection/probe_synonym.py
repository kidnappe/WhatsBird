# -*- coding: utf-8 -*-
"""异名候选探针：对未命中的种，按「种加词」和「中文名」在 CSV 里找候选行。

输出的别名表要人眼过一遍再写进 build_protection_asset.py，不允许凭记忆填。
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "tools/bird_protection/raw/china_bird_list_v10.csv"
LABELS = ROOT / "android/app/src/main/assets/models/labels_national_v2.txt"

UNMATCHED = [
    "Anarhynchus alexandrinus", "Anarhynchus leschenaultii", "Anthus japonicus",
    "Ardea coromanda", "Astur gentilis", "Botaurus sinensis", "Cairina moschata",
    "Cygnus atratus", "Lophospiza trivirgata", "Nucifraga hemispila",
    "Paradoxornis gularis", "Paradoxornis heudei", "Periparus venustulus",
    "Saundersilarus saundersi", "Suthora alphonsiana", "Suthora webbiana",
    "Thinornis dubius",
]


def load_rows():
    rows = []
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            sci = row[1].strip()
            if not sci or " " not in sci:
                continue
            rows.append((sci, row[2].strip(), row[4].strip(), row[5].strip()))
    return rows


def main():
    rows = load_rows()
    labels = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = line.split("\t")
            labels[p[0].strip()] = p[1].strip() if len(p) > 1 else ""

    for sci in UNMATCHED:
        zh = labels.get(sci, "")
        epithet = sci.split()[-1].lower()
        # 属名可能不同（属拆分），用种加词 + 中文名双条件找候选
        by_zh = [(r[0], r[1], r[2], r[3]) for r in rows if r[1] == zh]
        by_ep = [(r[0], r[1], r[2], r[3]) for r in rows
                 if r[0].split()[-1].lower() == epithet]
        print(f"\n### {sci}  [{zh}]")
        if by_zh:
            print(f"   中文名命中 {len(by_zh)}:")
            for r in by_zh[:6]:
                mark = "  <== 同种加词" if r[0].split()[-1].lower() == epithet else ""
                print(f"     {r[0]:<32} {r[1]:<10} iucn={r[2]:<4} 保护={r[3]}{mark}")
        if not by_zh:
            print("   中文名未命中")
        if not by_ep:
            print(f"   种加词 '{epithet}' 未命中")
        elif not by_zh:
            print(f"   种加词命中 {len(by_ep)}:")
            for r in by_ep[:6]:
                print(f"     {r[0]:<32} {r[1]:<10} iucn={r[2]:<4} 保护={r[3]}")


if __name__ == "__main__":
    sys.exit(main())
# -*- coding: utf-8 -*-
"""补漏探针：为什么 5 个「理应精确命中」的种没匹配上。"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "tools/bird_protection/raw/china_bird_list_v10.csv"

TARGETS = ["Alcippe davidi", "Alcippe hueti", "Anser serrirostris",
           "Parus cinereus", "Saxicola stejnegeri", "Ardea coromanda"]


def main():
    rows = []
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) >= 6:
                rows.append(r)

    for t in TARGETS:
        zh = ""
        ep = t.split()[-1].lower()
        print(f"\n### {t}")
        exact = [r for r in rows if r[1].strip().lower() == t.lower()]
        for r in exact:
            print(f"  精确行: {r}")
        byep = [r for r in rows if len(r[1].split()) >= 2 and r[1].split()[-1].lower() == ep]
        for r in byep[:8]:
            print(f"  同种加词: {r[1]:<30} {r[2]:<10} iucn={r[4]:<4} 保护={r[5]}")

    print("\n--- CSV 中含 'Parus' / 'Alcippe' / 'Schoeniparus' / 'Anser' 的行 ---")
    for r in rows:
        s = r[1].strip()
        if s.split()[0] in ("Parus", "Alcippe", "Schoeniparus", "Anser", "Bubulcus"):
            print(f"  {s:<30} {r[2]:<10} iucn={r[4]:<4} 保护={r[5]}")


if __name__ == "__main__":
    sys.exit(main())
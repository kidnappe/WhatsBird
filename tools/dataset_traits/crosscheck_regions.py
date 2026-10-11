# -*- coding: utf-8 -*-
"""区域前缀扫描：P(旅鸟) 的大面积不一致，到底是数据集错了，还是我把 bbox 归错动物地理区？

对每个候选区划前缀(I/II/III/VI/VII)各算一遍单值代码 vs GBIF 实测物候型的命中率，
并顺带看那些被标 P 却在 6-7 月零记录的物种在各区的代码，判断有没有哪个区把它标成 W/R。
"""
import csv
import re
import sys
from collections import defaultdict, Counter
import openpyxl

XLSX, CSV = sys.argv[1], sys.argv[2]
BREED, WINTER, PRESENT = {6, 7}, {12, 1}, 0.05

wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ws = wb['Chinesebirdsdata']
rows = [r for r in ws.iter_rows(values_only=True)][2:]
wb.close()
rows = [r for r in rows if r[4]]
REG_RE = re.compile(r'^([IVX]+)([A-Z]?)\(([^)]*)\)')


def parse_regions(s):
    out = defaultdict(set)
    for part in str(s or '').split(','):
        m = REG_RE.match(part.strip())
        if m:
            out[m.group(1)] |= {x.strip() for x in m.group(3).split(',') if x.strip()}
    return out


traits = {str(r[4]).strip(): {'cn': str(r[2]).strip(), 'union': str(r[25] or '').strip(),
                              'regions': parse_regions(r[26])} for r in rows}
months = defaultdict(list)
with open(CSV, encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        n, m = (row.get('scientific_name') or '').strip(), (row.get('month') or '').strip()
        if n and m.isdigit():
            months[n].append(int(m))

matched = [sp for sp in sorted(set(months) & set(traits)) if len(months[sp]) >= 30]
EXPECT = {'R': {'R型', 'S型', 'W型'}, 'S': {'S型', 'R型'}, 'W': {'W型'}, 'P': {'P型'}}


def gtype(ms):
    c, n = Counter(ms), len(ms)
    b = sum(c[m] for m in BREED) / n >= PRESENT
    w = sum(c[m] for m in WINTER) / n >= PRESENT
    return ('R型' if b and w else 'S型' if b else 'W型' if w else 'P型'), \
           round(sum(c[m] for m in BREED) / n, 3), round(sum(c[m] for m in WINTER) / n, 3)


print("=== 各候选动物地理区前缀的单值代码命中率 ===")
print(f"{'区':<5s}{'R':>14s}{'S':>14s}{'W':>14s}{'P':>14s}{'  可判物种':>10s}")
for reg in ('I', 'II', 'III', 'IV', 'V', 'VI', 'VII'):
    conf = defaultdict(Counter)
    for sp in matched:
        codes = traits[sp]['regions'].get(reg, set())
        if len(codes) != 1 or not codes:
            continue
        code = codes.pop()
        if code not in EXPECT:
            continue
        conf[code][gtype(months[sp])[0]] += 1
    cells = ""
    tot_n = 0
    for code in ('R', 'S', 'W', 'P'):
        r = conf[code]
        n = sum(r.values())
        ok = sum(v for k, v in r.items() if k in EXPECT[code])
        tot_n += n
        cells += f"{ok}/{n}".rjust(14) if n else "—".rjust(14)
    print(f"{reg:<5s}{cells}{tot_n:>10d}")

print("\n=== union(全国口径) 单值代码命中率，做对照 ===")
conf = defaultdict(Counter)
for sp in matched:
    codes = set(traits[sp]['union'])
    if len(codes) != 1 or not codes:
        continue
    code = codes.pop()
    if code not in EXPECT:
        continue
    conf[code][gtype(months[sp])[0]] += 1
for code in ('R', 'S', 'W', 'P'):
    r = conf[code]
    n = sum(r.values())
    ok = sum(v for k, v in r.items() if k in EXPECT[code])
    print(f"  {code}: {ok}/{n} = {ok/n*100:.1f}%  分布={dict(r)}" if n else f"  {code}: —")

print("\n=== 重点：繁殖季(6-7月)零记录、却被标为旅鸟(P)的物种，在别的区是什么码？===")
print(f"{'中文名':<12s}{'拉丁名':<26s}{'union':>7s}  " + "".join(f"{r:>8s}" for r in ('I', 'II', 'III', 'VI', 'VII')) + "   冬占比")
cnt = 0
for sp in matched:
    ms = months[sp]
    gt, br, wr = gtype(ms)
    if br > 0 or wr < 0.1:
        continue
    t = traits[sp]
    iic = t['regions'].get('II', set())
    if iic != {'P'}:
        continue
    cells = "".join(f"{''.join(sorted(t['regions'].get(r, set()))) or '-':>8s}" for r in ('I', 'II', 'III', 'VI', 'VII'))
    print(f"{t['cn']:<12s}{sp:<26s}{t['union']:>7s}  {cells}   {wr}")
    cnt += 1
print(f"合计 {cnt} 种：华北 bbox 内 6-7 月零记录（不繁殖）且有冬季记录，但区划 II 标为旅鸟 P")

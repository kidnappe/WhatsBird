# -*- coding: utf-8 -*-
"""交叉校验 v2：把「居留型」当成繁殖状态来检验，而不是「哪季记录最多」。

关键语义：中国鸟类学里 留鸟R/夏候鸟S/冬候鸟W/旅鸟P 说的是**在该区的繁殖季身份**，
不是「哪季记录多」。而 GBIF 记录的季节分布被观测/采集强度主导。
所以正确的判别量是**繁殖窗口(6-7月)有没有记录**，不是峰值季节。

用法: python crosscheck_v2.py <xlsx> <dataset.csv> [区域前缀]
"""
import csv
import re
import sys
from collections import defaultdict, Counter
import openpyxl

XLSX, CSV = sys.argv[1], sys.argv[2]
REGION = sys.argv[3] if len(sys.argv) > 3 else 'II'
BREED, WINTER = {6, 7}, {12, 1}
PRESENT = 0.05          # 该窗口占比 >=5% 记为「有记录」

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

months, years, dsk = defaultdict(list), Counter(), Counter()
with open(CSV, encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        n = (row.get('scientific_name') or '').strip()
        m = (row.get('month') or '').strip()
        if n and m.isdigit():
            months[n].append(int(m))
        y = (row.get('year') or '').strip()
        years[y[:4] if len(y) >= 4 else y] += 1
        dsk[(row.get('dataset_key') or '')[:12]] += 1

print("=== GBIF 数据构成（先看它是什么口径的数据）===")
print("年份 top10:", years.most_common(10))
print("dataset_key top5:", dsk.most_common(5))

matched = sorted(set(months) & set(traits))
print(f"\n命中物种 {len(matched)}；基线季节占比见上一脚本（春37.6 冬30.3 夏18.3 秋13.7）\n")


def profile(ms):
    c = Counter(ms)
    n = len(ms)
    b = sum(c[m] for m in BREED) / n
    w = sum(c[m] for m in WINTER) / n
    return (b >= PRESENT), (w >= PRESENT), b, w, n


def gbif_type(b, w):
    if b and w:
        return 'R型(夏冬都在)'
    if b and not w:
        return 'S型(夏在冬缺)'
    if not b and w:
        return 'W型(冬在夏缺)'
    return 'P型(只在过渡季)'


EXPECT = {'R': {'R型(夏冬都在)', 'S型(夏在冬缺)', 'W型(冬在夏缺)'},   # 留鸟应四季有，夏冬至少不空
          'S': {'S型(夏在冬缺)', 'R型(夏冬都在)'},
          'W': {'W型(冬在夏缺)'},
          'P': {'P型(只在过渡季)'},
          'V': set()}

for src in ('union', '区域' + REGION):
    conf = defaultdict(Counter)
    det = []
    for sp in matched:
        ms = months[sp]
        if len(ms) < 30:
            continue
        b, w, br, wr, n = profile(ms)
        t = traits[sp]
        codes = set(t['union']) if src == 'union' else t['regions'].get(REGION, set())
        if len(codes) != 1 or not codes:
            continue
        code = codes.pop()
        if code not in EXPECT:
            continue
        gt = gbif_type(b, w)
        conf[code][gt] += 1
        det.append((sp, t['cn'], code, gt, round(br, 3), round(wr, 3), n))

    print(f"=== 用 {src} 的单值代码 vs GBIF 实测物候型 ===")
    types = ['R型(夏冬都在)', 'S型(夏在冬缺)', 'W型(冬在夏缺)', 'P型(只在过渡季)']
    print(f"  代码 \\ 实测   " + "".join(f"{t[:8]:>11s}" for t in types) + "   命中率")
    for code in ('R', 'S', 'W', 'P'):
        if not conf[code]:
            continue
        row = conf[code]
        tot = sum(row.values())
        ok = sum(v for k, v in row.items() if k in EXPECT[code])
        cells = "".join(f"{row.get(t, 0):>11d}" for t in types)
        print(f"  {code:<12s}{cells}   {ok}/{tot} = {ok/tot*100:5.1f}%")
    bad = [d for d in det if gbif_type(d[4] >= PRESENT, d[5] >= PRESENT) not in EXPECT[d[2]]]
    print(f"  反例 {len(bad)} 个，样例:")
    for sp, cn, code, gt, br, wr, n in sorted(bad, key=lambda x: -x[6])[:8]:
        print(f"    {cn:<12s} {sp:<26s} 码={code} 实测={gt:<14s} 繁殖季占比={br:<6} 冬季占比={wr:<6} n={n}")
    print()

# -*- coding: utf-8 -*-
"""交叉校验：生活史数据集的「迁徙状态（居留型）」 vs 项目 GBIF 记录的实际月度物候。

对照组：把全部物种合起来的月份分布当基线，抵消「观鸟季节性」这个观测者偏差。
对每个物种算 seasonality index = 该物种某季占比 / 基线该季占比。
然后检验：代码预测的季节，是不是该物种指数最高的季节。

用法: python crosscheck_migrant_status.py <xlsx> <dataset.csv> [区域前缀, 默认 II]
"""
import csv
import re
import sys
from collections import defaultdict, Counter
import openpyxl

XLSX = sys.argv[1]
CSV = sys.argv[2]
REGION = sys.argv[3] if len(sys.argv) > 3 else 'II'

SEASONS = {'春(3-5)': {3, 4, 5}, '夏(6-8)': {6, 7, 8}, '秋(9-11)': {9, 10, 11}, '冬(12-2)': {12, 1, 2}}
SEASON_OF = {m: s for s, ms in SEASONS.items() for m in ms}

# ---------- 1. 读生活史数据集 ----------
wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
ws = wb['Chinesebirdsdata']
rows = [r for r in ws.iter_rows(values_only=True)][2:]
wb.close()
rows = [r for r in rows if r[4]]

REG_RE = re.compile(r'^([IVX]+)([A-Z]?)\(([^)]*)\)')


def parse_regions(s):
    """'IIIA(S),IVB(P)' -> {'III': {'S'}, 'IV': {'P'}}"""
    out = defaultdict(set)
    for part in str(s or '').split(','):
        m = REG_RE.match(part.strip())
        if m:
            out[m.group(1)] |= {x.strip() for x in m.group(3).split(',') if x.strip()}
    return out


traits = {}
for r in rows:
    traits[str(r[4]).strip()] = {
        'cn': str(r[2]).strip(),
        'endemic': str(r[5]).strip(),
        'union': str(r[25] or '').strip(),
        'regions': parse_regions(r[26]),
        'nprov': str(r[27] or '').strip(),
    }

# ---------- 2. 读 GBIF 区域数据 ----------
months = defaultdict(list)
n_rows = 0
with open(CSV, encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        n = (row.get('scientific_name') or '').strip()
        m = (row.get('month') or '').strip()
        if n and m.isdigit():
            months[n].append(int(m))
            n_rows += 1

pooled = Counter(SEASON_OF[m] for ms in months.values() for m in ms)
tot = sum(pooled.values())
base = {s: pooled[s] / tot for s in SEASONS}

# ---------- 3. 校验 ----------
def indices(ms):
    c = Counter(SEASON_OF[m] for m in ms)
    n = sum(c.values())
    return {s: (c[s] / n) / base[s] if base[s] else 0 for s in SEASONS}


def predict(codes, union):
    """从代码推预测季节；多值/无值返回 None"""
    codes = {c for c in codes if c}
    if len(codes) != 1:
        return None
    c = codes.pop()
    return {'R': None, 'S': '夏(6-8)', 'W': '冬(12-2)', 'P': None, 'V': None}.get(c, None), c


print(f"GBIF 记录 {n_rows} 行，有月份 {tot} 条；区域前缀 = {REGION}")
print("基线季节占比（全体物种合起来，含观鸟季节性偏差）: " +
      "  ".join(f"{s}={base[s]*100:.1f}%" for s in SEASONS))

matched = sorted(set(months) & set(traits))
print(f"物种：GBIF {len(months)} / 生活史 {len(traits)} / 命中 {len(matched)}\n")

stat = defaultdict(lambda: [0, 0])      # (来源, 代码) -> [测对数, 命中数]
residents = []
examples = []
for sp in matched:
    ms = months[sp]
    if len(ms) < 30:                    # 样本太少的物种跳过，避免噪声
        continue
    idx = indices(ms)
    t = traits[sp]
    for src, codes in (('union', set(t['union'])), ('区域' + REGION, t['regions'].get(REGION, set()))):
        if not codes:
            continue
        if codes == {'R'}:
            # 留鸟：预测四季都该有（指数不偏）
            stat[(src, 'R')][0] += 1
            spread = max(idx.values()) - min(idx.values())
            if spread < 1.0:
                stat[(src, 'R')][1] += 1
            residents.append((sp, t['cn'], round(idx['夏(6-8)'], 2), round(idx['冬(12-2)'], 2), len(ms), src))
            continue
        p = predict(codes, t['union'])
        if not p or not p[0]:
            continue
        want, code = p
        stat[(src, code)][0] += 1
        best = max(idx, key=idx.get)
        ok = (best == want)
        if ok:
            stat[(src, code)][1] += 1
        else:
            examples.append((sp, t['cn'], src, ''.join(sorted(codes)), want, best,
                             round(idx[want], 2), round(idx[best], 2), len(ms)))

print("=== 一致率（代码预测的季节 == 实测指数最高的季节）===")
for (src, code), (n, ok) in sorted(stat.items()):
    kind = '四季均衡' if code == 'R' else '峰值季节'
    print(f"  {src:<10s} {code}  n={n:>3d}  一致={ok:>3d}  {ok/n*100:5.1f}%   ({kind})")

print("\n=== 反例（前 15，按样本量降序）===")
for e in sorted(examples, key=lambda x: -x[7])[:15]:
    sp, cn, src, code, want, best, wi, bi, n = e
    print(f"  {cn:<12s} {sp:<28s} {src:<8s} 码={code:<4s} 预测={want} 实测峰值={best} "
          f"(预测季指数{wi} vs 峰值季指数{bi}) n={n}")

print("\n=== 标为留鸟(R) 的物种：实测季节均衡度（越接近 0 越像留鸟）===")
for r in sorted(residents, key=lambda x: -(x[2] + x[3])):
    sp, cn, s_idx, w_idx, n, src = r
    print(f"  {cn:<12s} {sp:<28s} 夏指数={s_idx:<5} 冬指数={w_idx:<5} n={n:<5} ({src})")

# ---------- 4. 代码本身的信息量 ----------
print("\n=== 迁徙状态代码分布（union 层）===")
cnt = Counter(t['union'] for t in traits.values())
for k, v in cnt.most_common(14):
    print(f"  {k or '(空)':<8s} {v}")
uniq = sum(v for k, v in cnt.items() if len({c for c in k}) == 1)
print(f"  单值代码（可直接当月过滤）: {uniq}/{len(traits)} = {uniq/len(traits)*100:.1f}%")

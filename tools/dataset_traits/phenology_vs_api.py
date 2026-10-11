# -*- coding: utf-8 -*-
"""量化：项目 GBIF 语料（每物种截断 min(300,可用)）的月度物候，与 GBIF 全量 facet 差多少？

GBIF 的 occurrence/search 支持 facet=month&limit=0 —— 一次请求直接拿该物种在 bbox 内的
**全量、不截断**月份直方图。所以「截断语料能不能当物候先验」这件事可以精确验证：
对每个命中物种，比较 本语料月份占比 vs API 全量月份占比（12 维），看相关性和判定翻转。
结果缓存到 phenology_cache.json。
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

BBOX = (36, 43, 110, 120)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phenology_cache.json')
CSV = r'E:\code\code tools\data\huabei\dataset.csv'
BREED, PRESENT = {6, 7}, 0.05


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ObserveBird-research/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


cache = json.load(open(CACHE, encoding='utf-8')) if os.path.exists(CACHE) else {}
months = defaultdict(list)
with open(CSV, encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        n = (row.get('scientific_name') or '').strip()
        m = (row.get('month') or '').strip()
        if n and m.isdigit():
            months[n].append(int(m))

todo = [s for s in sorted(months) if len(months[s]) >= 30]
print(f"待查物种 {len(todo)}（本语料 >=30 条），已有缓存 {len(cache)}")
for i, sp in enumerate(todo, 1):
    if sp in cache:
        continue
    key = get('https://api.gbif.org/v1/species/match?name=' + urllib.parse.quote(sp)).get('usageKey')
    if not key:
        cache[sp] = {'error': 'no usageKey'}
        continue
    url = (f'https://api.gbif.org/v1/occurrence/search?taxonKey={key}'
           f'&decimalLatitude={BBOX[0]},{BBOX[1]}&decimalLongitude={BBOX[2]},{BBOX[3]}'
           f'&facet=month&facetLimit=12&limit=0&hasCoordinate=true')
    r = get(url)
    fac = next((f['counts'] for f in r.get('facets', []) if f['field'] == 'MONTH'), [])
    cache[sp] = {'key': key, 'total': r.get('count'),
                 'months': {c['name']: c['count'] for c in fac}}
    if i % 20 == 0:
        print(f"  ... {i}/{len(todo)}")
        json.dump(cache, open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False)
    time.sleep(0.12)
json.dump(cache, open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False)


def props(counter, total):
    return {m: counter.get(m, 0) / total for m in range(1, 13)} if total else {}


def pearson(a, b):
    n = 12
    ma, mb = sum(a.values()) / n, sum(b.values()) / n
    num = sum((a[m] - ma) * (b[m] - mb) for m in range(1, 13))
    da = sum((a[m] - ma) ** 2 for m in range(1, 13)) ** 0.5
    db = sum((b[m] - mb) ** 2 for m in range(1, 13)) ** 0.5
    return num / (da * db) if da and db else float('nan')


rows, flips, nodata = [], [], []
for sp in todo:
    c = cache.get(sp) or {}
    if 'months' not in c:
        nodata.append(sp)
        continue
    loc = Counter(months[sp])
    a = props(loc, sum(loc.values()))
    b = props({int(k): v for k, v in c['months'].items()}, sum(c['months'].values()))
    cor = pearson(a, b)
    br_loc = sum(a[m] for m in BREED)
    br_api = sum(b[m] for m in BREED)
    flip = (br_loc >= PRESENT) != (br_api >= PRESENT)
    rows.append((sp, cor, br_loc, br_api, sum(loc.values()), c['total'] or 0, flip))
    if flip:
        flips.append(rows[-1])

cors = sorted(r[1] for r in rows if r[1] == r[1])
print(f"\n=== 本语料(截断) vs GBIF 全量 facet：12维月份占比相关性（{len(cors)} 物种）===")
print(f"  中位 r = {cors[len(cors)//2]:.3f}   p10 = {cors[len(cors)//10]:.3f}   p90 = {cors[len(cors)*9//10]:.3f}   min = {cors[0]:.3f}")
print(f"  r >= 0.9 的物种: {sum(1 for c in cors if c >= 0.9)}/{len(cors)}")
print(f"  r <  0.7 的物种: {sum(1 for c in cors if c < 0.7)}/{len(cors)}")
print(f"\n=== 「6-7月有记录(繁殖季占比>=5%)」这个判定的翻转 ===")
print(f"  翻转 {len(flips)}/{len(rows)} = {len(flips)/len(rows)*100:.1f}%")
for sp, cor, bl, ba, nl, na, _ in sorted(flips, key=lambda x: -x[4])[:12]:
    print(f"  {sp:<28s} r={cor:6.3f}  语料繁殖季={bl:.3f}  全量繁殖季={ba:.3f}  语料n={nl:<5} 全量n={na}")
if nodata:
    print(f"\n无 API 数据 {len(nodata)} 个: {nodata[:10]}")
print(f"\n缓存: {CACHE}")

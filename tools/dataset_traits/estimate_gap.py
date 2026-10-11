# -*- coding: utf-8 -*-
"""估算「补数据」会给各区训练集带来多大膨胀、要多久。

口径：
  * 供给 = iNat 在该区 bbox 内的研究级带照片记录数（用 total_results，1 次请求/种）
  * 目标 = 每类想达到的张数（默认 90，= 华北现状中位数；另外给出 150 / 300 两档参照）
  * 缺口 = max(0, 目标 - 该种现有张数)，但不超过供给
  * 下载速率取项目自己实测过的 250 张/分（HANDOFF §6）
"""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = r'E:\code\code tools\data'
REGIONS = {                     # lat_min, lat_max, lon_min, lon_max
    'huabei':    (36, 43, 110, 120),
    'huazhong':  (31, 37, 105, 117),
    'huadong':   (27, 35, 117, 123),
    'huanan_sw': (20, 31, 97, 117),
}
TARGETS = (90, 150, 300)
RATE_PER_MIN = 250              # 张/分，项目实测（GBIF 图片期）


def bbox_param(r):
    a, b, c, d = REGIONS[r]
    return f"swlat={a}&nelat={b}&swlng={c}&nelng={d}"


def inat_total(name, region):
    u = ("https://api.inaturalist.org/v1/observations?quality_grade=research&photos=true"
         f"&per_page=0&taxon_name={urllib.parse.quote(name)}&{bbox_param(region)}")
    for i in range(3):
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'ObserveBird-est/1.0'})
            with urllib.request.urlopen(req, timeout=60) as f:
                return json.load(f).get('total_results', 0)
        except Exception:
            if i == 2:
                return -1
            time.sleep(1.5 * (i + 1))


def load(region):
    p = os.path.join(BASE, region, 'gbif_availability.csv')
    out = []
    for r in csv.DictReader(open(p, encoding='utf-8-sig', newline='')):
        try:
            n = int(float(r.get('gbif_records') or 0))
        except ValueError:
            n = 0
        out.append((r['scientific_name'], r['common_name'], n))
    return out


# 现有已下载张数（dataset.csv 行数按种统计）
def downloaded(region):
    p = os.path.join(BASE, region, 'dataset.csv')
    c = {}
    if not os.path.exists(p):
        return c
    for row in csv.DictReader(open(p, encoding='utf-8-sig', newline='')):
        s = (row.get('scientific_name') or '').strip()
        if s:
            c[s] = c.get(s, 0) + 1
    return c


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    summary = {}
    for region in REGIONS:
        if only and region != only:
            continue
        rows = load(region)
        have = downloaded(region)
        # 只算「缺口种」：可下载量低于最大目标，或者实际下载为 0
        gap = [(s, cn, g, have.get(s, 0)) for s, cn, g in rows if g < max(TARGETS) or have.get(s, 0) == 0]
        print(f"\n=== {region}：名单 {len(rows)} 种，其中需要补的 {len(gap)} 种"
              f"（现有总计 {sum(have.values())} 张）===")
        per_target = {t: 0 for t in TARGETS}
        t0 = time.time()
        for i, (s, cn, g, cur) in enumerate(gap, 1):
            supply = inat_total(s, region)
            for t in TARGETS:
                need = max(0, t - cur)
                take = min(need, max(0, supply))
                per_target[t] += take
            if i % 25 == 0:
                print(f"    ... {i}/{len(gap)}  ({time.time()-t0:.0f}s)")
            time.sleep(0.15)
        print(f"  可补张数（按目标每类张数）:")
        for t in TARGETS:
            n = per_target[t]
            print(f"    目标 {t:>3d} 张/类 → 新增 {n:>7d} 张   "
                  f"下载约 {n/RATE_PER_MIN:>7.1f} 分钟")
        summary[region] = per_target
    print("\n=== 汇总 ===")
    for t in TARGETS:
        tot = sum(v[t] for v in summary.values())
        print(f"  目标 {t:>3d} 张/类：四区合计新增 {tot:>7d} 张，"
              f"下载约 {tot/RATE_PER_MIN/60:.1f} 小时")
    json.dump(summary, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'gap_estimate.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()

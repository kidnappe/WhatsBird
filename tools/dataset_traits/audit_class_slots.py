# -*- coding: utf-8 -*-
"""审计各区的「类别名额」花得值不值（对应 CLEANING-GUIDE 之外的一条独立检查）。

背景：每个区的类目名单是死配额（--n 150）。所以
  * 名单里多一个不该有的种 → 不污染，只是挤掉一个名额
  * 名单里少一个/废一个种   → 这个种永远出不来，模型还会自信地给近似错答案
本脚本查第二种：找出「占着名额却没有训练数据」的类，并区分原因：
  (a) 名字在 GBIF 名录里不存在 / 只能匹配到属 → 被 build_dataset_gbif 的 rank 校验挡掉
  (b) 名字没问题，但该 bbox 内确实几乎没有记录

用法: python audit_class_slots.py [--region huabei ...]
"""
import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter

BASE = r'E:\code\code tools\data'
REGIONS = {
    'huabei':    (36, 43, 110, 120),
    'huazhong':  (31, 37, 105, 117),
    'huadong':   (27, 35, 117, 123),
    'huanan_sw': (20, 31, 97, 117),
}
LOW = 30        # 低于此数视为「基本学不到东西」


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ObserveBird-audit/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def num(row):
    try:
        return int(float(row.get('gbif_records') or 0))
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', action='append', help='默认全部')
    a = ap.parse_args()
    regs = a.region or list(REGIONS)

    for reg in regs:
        p = os.path.join(BASE, reg, 'gbif_availability.csv')
        if not os.path.exists(p):
            print(f'\n### {reg}: 缺 gbif_availability.csv，跳过')
            continue
        rows = list(csv.DictReader(open(p, encoding='utf-8-sig', newline='')))
        vals = sorted(num(r) for r in rows)
        zero = [r for r in rows if num(r) == 0]
        low = [r for r in rows if 0 < num(r) < LOW]
        print(f'\n### {reg}  名单 {len(rows)} 种   可下载量 中位={vals[len(vals)//2]} 最小={vals[0]} 最大={vals[-1]}')
        print(f'    0 条（名额全废）= {len(zero)}    <{LOW} 条（基本学不到）= {len(low)}')
        for r in low[:10]:
            print(f'      低量  {r["scientific_name"]:<30s} {r["common_name"]:<12s} n={num(r)}')

        if not zero:
            continue
        lat0, lat1, lon0, lon1 = REGIONS.get(reg, (0, 0, 0, 0))
        BB = f'decimalLatitude={lat0},{lat1}&decimalLongitude={lon0},{lon1}'
        print(f'    --- 0 条种的归因（GBIF 名录匹配 + 该 bbox 实际记录数）---')
        for r in zero:
            sp = r['scientific_name']
            m = get('https://api.gbif.org/v1/species/match?name=' + urllib.parse.quote(sp))
            mt, rank = m.get('matchType'), m.get('rank')
            key = m.get('usageKey')
            note = ''
            if mt == 'NONE' or not key:
                note = 'GBIF 名录里没有这个名字 → 必然 0 条'
            elif rank != 'SPECIES':
                note = f'只匹配到【{rank}】→ 被 rank 校验挡掉 → 必然 0 条'
            else:
                b = get(f'https://api.gbif.org/v1/occurrence/search?taxonKey={key}&{BB}&limit=0')['count']
                if b >= 100:
                    note = (f'❗**假 0**：名字正常(SPECIES)且该 bbox 有 {b} 条记录 '
                            f'→ 预取阶段查询失败被当成"没有数据"缓存了，重跑管线不会自愈，须重查')
                else:
                    note = f'名字正常(SPECIES)，该 bbox 仅 {b} 条 → 确实是数据稀疏'
            print(f'      {r["common_name"]:<12s} {sp:<26s} {str(mt):<10s} {str(rank):<8s} {note}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""抽样看各特征列取值格式，并与项目现有物种清单（GBIF dataset.csv）做名称交集检查。"""
import csv
import openpyxl

xlsx = r'E:\code\observebird\tools\dataset_traits\supplFile_art_20220117115032.xlsx'
wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
ws = wb['Chinesebirdsdata']
rows = [r for r in ws.iter_rows(values_only=True)][2:]
rows = [r for r in rows if r[4]]
wb.close()

cols = {'食性': 18, '窝卵数': 19, '卵大小': 20, '巣型': 22, '巢址': 23, '集群': 24, '迁徙': 25, '分布区': 26, '省份数': 27, '特有种': 5}
for name, i in cols.items():
    vals = [str(r[i]).strip() for r in rows if r[i] is not None]
    uniq = sorted(set(vals))
    print(f"{name}: 唯一值 {len(uniq)} 个 -> {uniq[:12]}{' ...' if len(uniq) > 12 else ''}")

# 与项目 GBIF dataset.csv 的拉丁名交集
import os
gbif = r'E:\code\code tools\data\huabei\dataset.csv'
if os.path.exists(gbif):
    names = set()
    with open(gbif, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            n = (row.get('scientific_name') or '').strip()
            if n:
                names.add(n)
    mine = {str(r[4]).strip(): str(r[2]).strip() for r in rows}
    hit = names & set(mine)
    miss = names - set(mine)
    print(f"\nGBIF huabei 物种数 {len(names)}; 命中生活史数据集 {len(hit)}; 未命中 {len(miss)}")
    print("未命中样例:", sorted(miss)[:15])
    # 展示几个命中物种的重排关键字段
    print("\n命中样例（迁徙状态 / 分布省份数 / 巢址）:")
    idx = {str(r[4]).strip(): r for r in rows}
    for n in sorted(hit)[:8]:
        r = idx[n]
        print(f"  {n:<28s} {r[2]:<10s} 迁徙={r[25]}  省份数={r[27]}  巢址={r[23]}  巣型={r[22]}")
else:
    print("GBIF dataset.csv 不存在:", gbif)

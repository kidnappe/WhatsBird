# -*- coding: utf-8 -*-
r"""把 national 训练注册进 train_dashboard.py 的 REGIONS（幂等，可重复跑）。"""
p = r"E:\code\observebird\tools\dashboard\train_dashboard.py"
src = open(p, encoding="utf-8").read()
if '"national": {"model"' in src:
    print("national 条目已存在，跳过")
    raise SystemExit(0)

NAT = (
    '    "national": {"model": r"E:\\code\\code tools\\models\\national_mnv4_v1",\n'
    '                  "data": r"E:\\code\\code tools\\data\\national",\n'
    '                  "logdir": r"E:\\code\\code tools\\data\\national",\n'
    '                  "target": None, "baseline": 0, "label": "全国兜底 373（自训，授权数据）",\n'
    '                  "species_csv": r"E:\\code\\observebird\\tools\\national_clf\\species_national_373.csv"},\n'
)
anchor = 'species_huadong_150.csv'
lines = src.splitlines(keepends=True)
out, inserted = [], False
for ln in lines:
    out.append(ln)
    if anchor in ln and not inserted:
        out.append(NAT)
        inserted = True
assert inserted, "插入点未找到"
open(p, "w", encoding="utf-8", newline="").write("".join(out))
print("national 条目已插入")

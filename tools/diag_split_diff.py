# -*- coding: utf-8 -*-
"""找出 eval_report.py 与 train.py 两份 grouped_split 在华东上的实际差异。

背景（2026-10-09）：用 train.py 的代码判 华东 best.pt 得 56.16%（与训练日志一致），
而 tools/eval_report.py 对同一份权重、同一批图得 68.13%。
切分张数相同（都是 6905），题号顺序也一致，所以怀疑两份 grouped_split 切出的
**成员**不同。本脚本逐项比对：张数、交集、只在一边的分别多少，并给出样本。

用法：python tools/diag_split_diff.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T                                   # noqa: E402
import eval_report as E                             # noqa: E402

DATA = r"E:\code\code tools\data"
REGION = "huadong"


def main():
    ds = os.path.join(DATA, REGION, "dataset.csv")
    img_dir = os.path.join(os.path.dirname(ds), "images")

    tr_img_dir, tr_rows = T.load_rows(ds)
    _, va_train = T.grouped_split(tr_rows, 0.2)

    # eval_report 的读法（原样照抄）
    er_rows = []
    with open(ds, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if os.path.exists(os.path.join(img_dir, r["file"])):
                er_rows.append(r)
    _, va_er = E.grouped_split(er_rows, 0.2)

    ftr = {r["file"] for r in va_train}
    fer = {r["file"] for r in va_er}
    print(f"train.py   : rows={len(tr_rows):6}  val={len(va_train):5}  唯一文件={len(ftr):5}")
    print(f"eval_report: rows={len(er_rows):6}  val={len(va_er):5}  唯一文件={len(fer):5}")
    print(f"交集={len(ftr & fer)}   只在 train 侧={len(ftr - fer)}   只在 eval 侧={len(fer - ftr)}")
    if ftr != fer:
        only_tr = sorted(ftr - fer)[:5]
        only_er = sorted(fer - ftr)[:5]
        print("\n样例：")
        print(f"  只在 train 侧 val: {only_tr}")
        print(f"  只在 eval  侧 val: {only_er}")
        # 这些只在 eval 侧的图，是不是其实在 train 侧的训练集里？（= 泄漏）
        tr_set = {r["file"] for r in T.grouped_split(tr_rows, 0.2)[0]}
        leak = sum(1 for f_ in fer if f_ in tr_set)
        print(f"\n⚠️ eval 侧判为 val、却出现在 train 侧 train 集里的图: {leak} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
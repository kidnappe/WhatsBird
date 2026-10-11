# -*- coding: utf-8 -*-
"""检查 detb_eval/eval_set.csv（检测器四区独立评测集）对**分类器**是否也零泄漏。

背景（2026-10-09）：该评测集当初只对**检测器 B** 的训练/验证集做了 GBIF key 去重
（见 tools/make_detb_eval_set.py），**没有对五个区域分类器的训练集去重**。
华东的分类器验证集重建失败（见 HANDOFF-20261009 §3.1 坑 3），本脚本用来判断
「能否借用这份 400 张给分类器当干净验证集」。

判据：文件名形如 `学名_GBIFkey.jpg` / `学名_GBIFkey_记录号.jpg`，取末段数字作 GBIF key。
用法：python tools/check_eval_set_clean.py [--regions huadong ...]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

DATA = r"E:\code\code tools\data"
EVAL_CSV = os.path.join(DATA, "detb_eval", "eval_set.csv")


def gbif_key_of(filename: str):
    """从文件名解析 GBIF occurrence key：取最后一段纯数字。"""
    stem = os.path.splitext(filename)[0]
    for part in reversed(stem.split("_")):
        if part.isdigit() and len(part) >= 6:
            return part
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["huanan_sw", "huazhong", "huadong", "dongbei"])
    a = ap.parse_args()

    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        eval_rows = list(csv.DictReader(f))
    by_region = defaultdict(list)
    for r in eval_rows:
        by_region[r["region"]].append(r)
    print(f"eval_set.csv 共 {len(eval_rows)} 张；字段 {list(eval_rows[0].keys())}")

    ok = True
    for region in a.regions:
        ds = os.path.join(DATA, region, "dataset.csv")
        if not os.path.exists(ds):
            print(f"{region}: 没有 dataset.csv，跳过")
            continue
        with open(ds, encoding="utf-8") as f:
            train_keys = {gbif_key_of(r["file"]) for r in csv.DictReader(f)}
        train_keys.discard(None)
        rows = by_region.get(region, [])
        keys = [gbif_key_of(r["file"]) for r in rows]
        nokey = sum(1 for k in keys if k is None)
        hit = sum(1 for k in keys if k is not None and k in train_keys)
        n = len(rows)
        flag = "✅ 零泄漏，可借用" if hit == 0 and nokey == 0 else "❌ 有重叠/缺 key"
        if hit or nokey:
            ok = False
        print(f"  {region:<10} {n:>3} 张 | 解析不出 key {nokey} | 落在分类器训练集内 {hit}  {flag}")

    print("\n结论：" + ("四区均可借用为分类器独立验证集" if ok else "存在重叠，不可直接借用"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
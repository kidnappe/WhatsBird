# -*- coding: utf-8 -*-
r"""合并五区数据集 + 缺口种补抓 → 全国兜底训练集（2026-10-07 用户拍板：短期忽略授权，
合成五区为主路）。license/attribution 列原样保留——将来商用化时可直接按行过滤，数据不浪费。

类别下限：样本 <20 张的种不进训练（373 类强不均衡下的噪声类），落 <out>/dropped_classes.csv。

Usage: python build_national_merged.py --out "E:\code\code tools\data\national"
产出: <out>/dataset.csv（file 列为绝对路径，tools/train.py 已支持）
"""
import argparse
import csv
import os
import sys
from collections import Counter

# stdout 管道默认 GBK，种中文名里的「䴓」等扩展区字符会 UnicodeEncodeError 炸掉无人值守链
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATA = r"E:\code\code tools\data"
GAP_CSV = os.path.join(DATA, "national_gap", "dataset.csv")
REGIONS = ["huabei", "huazhong", "huanan_sw", "huadong", "dongbei"]
MIN_CLASS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "national"))
    ap.add_argument("--no-gap", action="store_true",
                    help="跳过缺口种补抓数据（两段式兜底：141 种由存量 cn373 兜底，"
                         "补抓数据留给全国 v2）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for region in REGIONS:
        src = os.path.join(DATA, region, "dataset.csv")
        img_dir = os.path.join(DATA, region, "images")
        for r in csv.DictReader(open(src, encoding="utf-8")):
            abs_path = os.path.join(img_dir, r["file"])
            if not os.path.exists(abs_path):
                continue
            rows.append({"file": abs_path, "scientific_name": r["scientific_name"],
                         "occurrence_key": r.get("occurrence_key", ""),
                         "license": r.get("license", ""),
                         "attribution": r.get("attribution", "")})
        print(f"  {region}: 累计 {len(rows)}")

    if os.path.exists(GAP_CSV) and not a.no_gap:
        gap_img = os.path.join(os.path.dirname(GAP_CSV), "images")
        n0 = len(rows)
        for r in csv.DictReader(open(GAP_CSV, encoding="utf-8")):
            abs_path = os.path.join(gap_img, r["file"])
            if not os.path.exists(abs_path):
                continue
            rows.append({"file": abs_path, "scientific_name": r["scientific_name"],
                         "occurrence_key": r.get("occurrence_key", ""),
                         "license": r.get("license", ""),
                         "attribution": r.get("attribution", "")})
        print(f"  缺口种补抓: +{len(rows) - n0}")

    cnt = Counter(r["scientific_name"] for r in rows)
    dropped = sorted(s for s, c in cnt.items() if c < MIN_CLASS)
    keep = [r for r in rows if cnt[r["scientific_name"]] >= MIN_CLASS]
    out_csv = os.path.join(a.out, "dataset.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "scientific_name", "occurrence_key",
                                          "license", "attribution"])
        w.writeheader()
        w.writerows(keep)
    with open(os.path.join(a.out, "dropped_classes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scientific_name", "n_images"])
        for s in dropped:
            w.writerow([s, cnt[s]])
    n_cls = len({r["scientific_name"] for r in keep})
    print(f"合并完成: {len(keep)} 张 / {n_cls} 类（弃 {len(dropped)} 个 <{MIN_CLASS} 张的类）")
    print(f"→ {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

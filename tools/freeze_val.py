# -*- coding: utf-8 -*-
"""冻结验证集：把"当前 dataset.csv 按 train.py 的 split 逻辑切出的 val"固化成文件。

为什么要它：dataset.csv 会涨（扩量、iNat/ Commons 补图、清洗删行），而 train.py 的
grouped_split 是按当时全量重切的 —— 同一份 checkpoint 换个时间点评测，val 集就换了题。
实测踩过：华中 17k 模型对扩到 25k 的 dataset.csv 评测报 60.25%，而它自己的记录是 57.75%，
差额不是变强，是训练时见过的图被重新切进了"验证集"（标签泄漏）。

产物（默认与 dataset.csv 同目录）：
- frozen_val.csv      file,scientific_name,occurrence_key（0 号种子的 val 观察集）
- frozen_val.meta.json 生成时间/行数/指纹，供复验

用法：
    python tools/freeze_val.py --region huazhong          # 生成（已存在则只报漂移）
    python tools/freeze_val.py --region huazhong --force  # 重切并覆盖
    python tools/freeze_val.py --region huazhong --verify # 只校验冻结集是否还能用上
之后训练/评测都带上：
    python tools/train.py  --dataset ... --val-list <data>/frozen_val.csv
    python tools/eval_report.py --ckpt ... --dataset ... --val-list <data>/frozen_val.csv
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import grouped_split, load_rows  # noqa: E402  复用同一份 split，杜绝口径漂移

DATA = r"E:\code\code tools\data"


def fingerprint(files):
    return hashlib.sha1("\n".join(sorted(files)).encode("utf-8")).hexdigest()[:12]


def read_frozen(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["file"] for r in rows}, rows


def write_frozen(out_csv, rows, meta):
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "scientific_name", "occurrence_key"])
        for r in rows:
            w.writerow([r["file"], r["scientific_name"], r.get("occurrence_key", "")])
    with open(out_csv + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--dataset", default=None, help="默认 <data>/<region>/dataset.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true", help="只查已有冻结集的可复用性")
    a = ap.parse_args()

    dataset = a.dataset or os.path.join(DATA, a.region, "dataset.csv")
    img_dir = os.path.join(os.path.dirname(dataset), "images")
    out_csv = os.path.join(os.path.dirname(dataset), "frozen_val.csv")

    if a.verify:
        if not os.path.exists(out_csv):
            sys.exit(f"没有冻结集：{out_csv}")
        files, rows = read_frozen(out_csv)
        missing = [f for f in files if not os.path.exists(os.path.join(img_dir, f))]
        print(f"冻结集 {len(files)} 张 / {len({r['scientific_name'] for r in rows})} 种，"
              f"图片缺失 {len(missing)} 张")
        if missing:
            print("  缺失示例:", missing[:5])
        sys.exit(1 if missing else 0)

    _, current_rows = load_rows(dataset)
    print(f"dataset.csv 现有 {len(current_rows)} 张（图片在盘的）")
    _, fresh_val = grouped_split(current_rows, a.val_frac, a.seed)

    if os.path.exists(out_csv) and not a.force:
        files, rows = read_frozen(out_csv)
        now = {r["file"] for r in fresh_val}
        meta = json.load(open(out_csv + ".meta.json", encoding="utf-8"))
        print(f"已有冻结集 {len(files)} 张（生成于 {meta['generated_at']}，"
              f"当时 dataset {meta['dataset_rows_with_image']} 行，指纹 {meta['fingerprint']}）")
        print(f"  与当前重切结果重合 {len(files & now)} / 冻结集独有 {len(files - now)} / "
              f"新切独有 {len(now - files)}")
        print("  想覆盖加 --force；训练/评测继续用 --val-list 指它")
        return

    files = {r["file"] for r in fresh_val}
    write_frozen(out_csv, fresh_val, {
        "region": a.region,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "dataset_rows_with_image": len(current_rows),
        "n_val": len(fresh_val),
        "val_frac": a.val_frac,
        "seed": a.seed,
        "species": len({r["scientific_name"] for r in fresh_val}),
        "fingerprint": fingerprint(files),
    })
    print(f"-> {out_csv}  {len(fresh_val)} 张 / "
          f"{len({r['scientific_name'] for r in fresh_val})} 种  "
          f"指纹 {fingerprint(files)}")


if __name__ == "__main__":
    main()

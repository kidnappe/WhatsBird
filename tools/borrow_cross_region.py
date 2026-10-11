"""Borrow tail-species images from sibling regional datasets.

The regional models share the 373-species universe, so a species that is data-
poor in one region's bbox is often data-rich in another. This copies rows (and
their image files) from sibling datasets into the target region for species
below a floor, capped at --cap total images per species. Dedupe by
occurrence_key against the target's existing rows and between sources; filenames
get a region prefix to avoid collisions.

Borrowed rows keep their license/attribution (compliance) and take the TARGET
region's birdnet_index (class mapping). dataset_key is set to "<src>_borrow"
so per-source counters in the dashboard are unaffected; rows coming from the
reserve pool (tools/reserve_pool.py, <data>/reserve/*.csv, borrowed but off-list
species whose images stay in place) get "<region>_reserve" instead.

Usage:
  python tools/borrow_cross_region.py --target "E:\\...\\data\\huazhong" \
      --species species_huazhong_150.csv \
      --source "E:\\...\\data\\huabei" --source "E:\\...\\data\\huanan_sw" \
      --floor 150 --cap 150 --dry-run
"""
import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter

from build_dataset_gbif import write_json_atomic
from reserve_pool import resolve_image  # 候补池行的图片在原地区目录，路径由池格式的作者解释

COLS = ["file", "scientific_name", "common_name", "birdnet_index", "gbif_key",
        "occurrence_key", "lat", "lon", "year", "month", "license",
        "dataset_key", "attribution"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="目标区域数据目录")
    ap.add_argument("--species", required=True, help="目标区域的物种清单 CSV")
    ap.add_argument("--source", action="append", required=True,
                    help="借出区域数据目录（可多次）")
    ap.add_argument("--floor", type=int, default=150,
                    help="物种已有图片低于此数才借")
    ap.add_argument("--cap", type=int, default=150,
                    help="借用后每物种图片总数上限")
    ap.add_argument("--include-empty", action="store_true",
                    help="连一张图都没有的薄种也一起补（默认不补，保持历史行为可复现）")
    ap.add_argument("--reserve-dir", default=None,
                    help="候补池目录，默认 <data>/reserve（见 tools/reserve_pool.py）")
    ap.add_argument("--no-reserve", action="store_true",
                    help="只从 --source 区域借，不看候补池")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印借用计划，不复制图片、不写 dataset.csv")
    a = ap.parse_args()

    # 目标区现状
    ds_path = os.path.join(a.target, "dataset.csv")
    img_dir = os.path.join(a.target, "images")
    have, done_keys = {}, set()
    for r in csv.DictReader(open(ds_path, encoding="utf-8")):
        have[r["scientific_name"]] = have.get(r["scientific_name"], 0) + 1
        for k in (r.get("occurrence_key", ""), r.get("gbif_key", "")):
            if k:
                done_keys.add(k)
                # 借入行的 occurrence_key 落盘带 "{tag}-" 前缀，池里是裸数字 key。
                # 不剥前缀的话重跑对不上，会二次追加同一批图。GBIF key 只含数字。
                if "-" in k:
                    done_keys.add(k.split("-", 1)[1])
    # 目标区的类映射（borrow 行统一用目标区的 birdnet_index）
    idx_of = {r["scientific_name"]: r.get("birdnet_index", "")
              for r in csv.DictReader(open(a.species, encoding="utf-8-sig"))}
    zh_of = {r["scientific_name"]: r.get("common_name", "")
             for r in csv.DictReader(open(a.species, encoding="utf-8-sig"))}

    print(f"目标: {a.target}（{sum(have.values())} 行 / {len(have)} 种）")
    print(f"借用规则: 已有 < {a.floor} 张的种，补到 {a.cap} 张\n")

    # 借出池：sci -> [(tag, row, img_path, kind)]。kind 只影响落盘时的 dataset_key 溯源。
    pool = {}

    def add(tag, r, img, kind):
        pool.setdefault(r["scientific_name"], []).append((tag, r, img, kind))

    for src in a.source:
        tag = os.path.basename(os.path.normpath(src))  # huabei / huanan_sw
        src_ds = os.path.join(src, "dataset.csv")
        n_src = 0
        for r in csv.DictReader(open(src_ds, encoding="utf-8")):
            sci = r["scientific_name"]
            img = os.path.join(src, "images", r["file"])
            if not os.path.exists(img):
                continue
            add(tag, r, img, "dataset")
            n_src += 1
        print(f"借出池 {tag}: {n_src} 行（与目标清单交集 "
              f"{sum(1 for s in pool if s in idx_of)} 种）")

    # 候补池：种已从该区名单里掉下去、图仍在其 data/<region>/images/ 原地
    reserve_dir = a.reserve_dir or os.path.join(
        os.path.dirname(os.path.normpath(a.target)), "reserve")
    if not a.no_reserve and os.path.isdir(reserve_dir):
        n_res = 0
        for name in sorted(os.listdir(reserve_dir)):
            if not name.endswith(".csv"):
                continue
            with open(os.path.join(reserve_dir, name), encoding="utf-8-sig",
                      newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "region" not in reader.fieldnames:
                    continue  # 不是池文件（别人的 csv 混进来了）
                for r in reader:
                    img = resolve_image(r, reserve_dir)
                    if not r.get("scientific_name") or img is None:
                        continue
                    add(r["region"], r, img, "reserve")
                    n_res += 1
        print(f"候补池 {reserve_dir}: {n_res} 行")

    # 逐物种决定借用量。默认只补"已有图但不足 floor"的种 —— 华中 25k 就是这么借出来的，
    # 默认行为变了那批数字就不可复现了。一张图都没有的薄种要补，显式加 --include-empty。
    todo = sorted(set(idx_of) | set(have)) if a.include_empty else sorted(have)
    plan, total_borrow = {}, 0
    for sci in sorted(todo, key=lambda s: have.get(s, 0)):
        cnt = have.get(sci, 0)
        if cnt >= a.floor or sci not in idx_of:
            continue
        want = min(a.cap, a.floor) - cnt
        got = 0
        picks = []
        for tag, r, img, kind in pool.get(sci, []):
            if got >= want:
                break
            ok = r.get("occurrence_key", "") and r.get("gbif_key", "")
            if not ok:
                continue
            if r.get("occurrence_key", "") in done_keys:
                continue
            picks.append((tag, r, img, kind))
            done_keys.add(r.get("occurrence_key", ""))
            got += 1
        if picks:
            plan[sci] = picks
            total_borrow += len(picks)
    print(f"\n借用计划: {len(plan)} 个尾部种，共 +{total_borrow} 张")
    dist = Counter(min(a.cap, have.get(s, 0) + len(p)) for s, p in plan.items())
    print("借后各物种张数分布:", dict(sorted(dist.items())))

    if a.dry_run:
        for sci, picks in sorted(plan.items(), key=lambda x: len(x[1]))[:10]:
            print(f"  {sci:<26} {have.get(sci, 0):>4} -> +{len(picks)}")
        return

    # 落盘：拷图片 + 追加行
    fh = open(ds_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    copied = failed = 0
    for sci, picks in plan.items():
        for tag, r, img, kind in picks:
            new_name = f"{tag}_{r['file']}"
            dst = os.path.join(img_dir, new_name)
            try:
                shutil.copyfile(img, dst)
            except OSError:
                failed += 1
                continue
            w.writerow([new_name, sci, zh_of.get(sci, ""),
                        idx_of.get(sci, ""), r.get("gbif_key", ""),
                        f"{tag}-{r.get('occurrence_key', '')}",
                        r.get("lat", ""), r.get("lon", ""),
                        r.get("year", ""), r.get("month", ""),
                        r.get("license", ""),
                        f"{tag}_{'reserve' if kind == 'reserve' else 'borrow'}",
                        r.get("attribution", "")])
            copied += 1
        fh.flush()
    fh.close()
    rows = sum(1 for _ in open(ds_path, encoding="utf-8")) - 1
    print(f"\n完成: 拷贝 {copied} 张（失败 {failed}），dataset.csv -> {rows} 行")
    write_json_atomic(os.path.join(a.target, "borrow_report.json"),
                      {"borrowed": copied, "failed": failed,
                       "species": len(plan), "rows_after": rows,
                       "floor": a.floor, "cap": a.cap,
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()

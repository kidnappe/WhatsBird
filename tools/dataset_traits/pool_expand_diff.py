# -*- coding: utf-8 -*-
"""扩池推演：候选池 373 → 全部（约 1445 种）后，四个区的名单会变成什么样、要下多少图。

依赖：`build_name_map.py` 产出的 species_name_map.json（把名字对上 geomodel 索引）。
特性：
  * 推理次数 = 探针点数 × 48，**与候选池大小无关** —— 所以扩池的计算成本 ≈ 0
  * 探针点直接 import pick_region_species（单一来源，避免两处漂移）
  * 不只报差异，还回答"新进的种要下多少图"：先看**兄弟区已有的图能不能借**
    （borrow_cross_region.py 的机制），只算补不齐的那部分

用法:
  python pool_expand_diff.py                 # 跑四个区，写 pool_expand_diff.csv
  python pool_expand_diff.py --n 150         # 每区名额（默认 150）
  python pool_expand_diff.py --out-new-only  # 只导出新名单 CSV
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)                       # <repo>/tools
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)

GEOMODEL_CANDIDATES = [r"E:\code\code tools\geomodel",
                       os.path.join(TOOLS, "geomodel")]
DATA_ROOT = r"E:\code\code tools\data"
NAME_MAP = os.path.join(HERE, "species_name_map.json")
OUT_CSV = os.path.join(HERE, "pool_expand_diff.csv")

import pick_region_species as prs                   # 复用它的探针点定义


def find_geomodel():
    for d in GEOMODEL_CANDIDATES:
        if os.path.isfile(os.path.join(
                d, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx")):
            return d
    sys.exit("找不到 geomodel（--geomodel-dir 或 tools/geomodel/）")


def load_candidates():
    """候选 = 名字对照表里所有能对上 geomodel 索引的种。

    返回 {geomodel_index: {name, gbif_name, zh, src}}
    """
    if not os.path.exists(NAME_MAP):
        sys.exit(f"缺 {NAME_MAP} —— 先跑 build_name_map.py")
    m = json.load(open(NAME_MAP, encoding="utf-8"))
    cands = {}
    for name, v in m.items():
        gi = v.get("geomodel_index")
        if gi is None or gi == "":
            continue
        gi = int(gi)
        cands.setdefault(gi, {"name": name, "gbif_name": v.get("gbif_name") or name,
                              "zh": v.get("zh") or "", "src": v.get("src") or ""})
    return cands


def region_images(region: str):
    """该区 dataset.csv 里每种的图片数（已下）。"""
    out = defaultdict(int)
    p = os.path.join(DATA_ROOT, region, "dataset.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                s = (row.get("scientific_name") or "").strip()
                if s:
                    out[s] += 1
    return out


def old_list(region: str):
    """该区**当前实际在用的**名单。三个来源依次尝试：

      1) data/<r>/species_<r>_150.csv       —— 华东/华中是这个
      2) data/<r>/gbif_availability.csv     —— 下载器写的，就是当时的下载输入（150 行）
      3) dist/observebird_<r>_pack/tools/species/species_<r>_150.csv —— 外包包里的

    踩过的坑：原来只试 ①，于是华北/华南/东北三个区"旧名单 = 0"，差异表全空 ——
    看起来像"整份名单都换了"，其实只是没读到文件。
    """
    for p in (os.path.join(DATA_ROOT, region, f"species_{region}_150.csv"),
              os.path.join(DATA_ROOT, region, "gbif_availability.csv"),
              os.path.join(os.path.dirname(TOOLS), "dist",
                           f"observebird_{region}_pack", "tools", "species",
                           f"species_{region}_150.csv")):
        if not os.path.exists(p):
            continue
        try:
            out = [r["scientific_name"].strip()
                   for r in csv.DictReader(open(p, encoding="utf-8-sig", newline=""))
                   if r.get("scientific_name")]
            if out:
                return out
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--floor", type=int, default=90,
                    help="每类认为「够用」的图片数（用于算还需补多少）")
    ap.add_argument("--out-new-only", action="store_true")
    a = ap.parse_args()

    gm = find_geomodel()
    cands = load_candidates()
    zh_of = {v["name"]: v["zh"] for v in cands.values()}   # 名字 -> 中文（避免逐条线性查表）
    print(f"候选池（能对上 geomodel）: {len(cands)} 种", flush=True)

    # geomodel 标签：索引 -> 学名（用于给选中的种回填 GBIF 名）
    lab = open(os.path.join(gm, "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"),
               encoding="utf-8", errors="replace").read().splitlines()
    lab_sci = []
    for l in lab:
        p = l.split("\t")
        lab_sci.append(p[1].strip() if len(p) > 1 else "?")
    print(f"geomodel 标签: {len(lab_sci)}", flush=True)

    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(
        os.path.join(gm, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx"),
        so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    # 兄弟区图片库存：某物种在**别的区**一共有多少张（可直接借）
    all_region_imgs = {r: region_images(r) for r in prs.REGIONS if r != "national"}

    rows, summary = [], []
    for region in [r for r in prs.REGIONS if r != "national"]:
        probes = prs.REGIONS[region]
        print(f"\n=== {region}  探针 {len(probes)} 个 × 48 周 = {len(probes)*48} 次推理 ===",
              flush=True)
        acc = np.zeros(len(lab_sci))
        for lat, lon in probes:
            for week in range(1, 49):
                acc += sess.run(None, {iname: np.array([[lat, lon, week]],
                                                       dtype=np.float32)})[0][0]
        acc /= (len(probes) * 48)
        scored = sorted(((float(acc[gi]), gi, v) for gi, v in cands.items()),
                        reverse=True, key=lambda x: x[0])
        new = [s[2]["name"] for s in scored[: a.n]]
        old = old_list(region) or []
        keep = [s for s in new if s in old]
        drop = [s for s in old if s not in new]
        enter = [s for s in new if s not in old]
        own = region_images(region)

        borrow = defaultdict(int)
        need_total = 0
        for s in enter:
            b = sum(all_region_imgs[r].get(s, 0) for r in all_region_imgs)
            borrow[s] = b
            need_total += max(0, a.floor - min(a.floor, b))
        sib_ok = sum(1 for s in enter if borrow[s] >= a.floor)
        sib_part = sum(1 for s in enter if 0 < borrow[s] < a.floor)
        sib_none = sum(1 for s in enter if borrow[s] == 0)

        # 缺口：新名单里"自己一份都没有"的种
        missing_own = [s for s in new if own.get(s, 0) == 0]
        print(f"  旧 {len(old)} → 新 {len(new)}   留下 {len(keep)}  掉出 {len(drop)}  新进 {len(enter)}")
        print(f"  新进的 {len(enter)} 个：兄弟区能借够 {sib_ok} 个 / 能借到一部分 {sib_part} 个 / 完全没得借 {sib_none} 个")
        print(f"  → 补齐这 {len(enter)} 个到每类 {a.floor} 张，还需新下 ≈ {need_total} 张")
        print(f"  新名单里自己一份图都没有的种: {len(missing_own)} 个")
        best, worst = scored[0][0], scored[a.n - 1][0]
        print(f"  新名单分数区间 {worst:.4f} ~ {best:.4f}"
              f"（旧名单第 {a.n} 名的分数决定挤掉谁）")
        summary.append({"region": region, "old": len(old), "new": len(new),
                        "keep": len(keep), "drop": len(drop), "enter": len(enter),
                        "borrow_ok": sib_ok, "borrow_part": sib_part,
                        "borrow_none": sib_none, "need_download": need_total,
                        "missing_own": len(missing_own)})
        for s in enter:
            rows.append({"region": region, "species": s, "zh": zh_of.get(s, ""),
                         "sibling_images": borrow[s],
                         "need_download": max(0, a.floor - min(a.floor, borrow[s]))})
        if a.out_new_only:
            p = os.path.join(HERE, f"species_{region}_new{a.n}.csv")
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["rank", "scientific_name", "zh", "geomodel_mean", "gbif_name"])
                for i, (sc, gi, v) in enumerate(scored[: a.n], 1):
                    w.writerow([i, v["name"], v["zh"], round(sc, 6), v["gbif_name"]])
            print(f"  -> {p}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["region", "species", "zh",
                                          "sibling_images", "need_download"])
        w.writeheader(); w.writerows(rows)
    print("\n=== 汇总 ===")
    print(f"{'区':<11s}{'留下':>6s}{'掉出':>6s}{'新进':>6s}{'借得到':>8s}{'需新下(张)':>12s}")
    for s in summary:
        print(f"{s['region']:<11s}{s['keep']:>6d}{s['drop']:>6d}{s['enter']:>6d}"
              f"{s['borrow_ok']:>8d}{s['need_download']:>12,d}")
    print(f"\n-> {OUT_CSV}（新进种逐条：能借到多少张、还需下多少张）")


if __name__ == "__main__":
    main()

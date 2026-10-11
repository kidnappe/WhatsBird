# -*- coding: utf-8 -*-
"""扩池评估（IOC 统一口径版）：用一本名录重算"扩池到底换掉几个名额、换对了吗"。

**为什么要 IOC 版**

早先的 `pool_expand_diff.py` 直接拿 1,445 个名字参赛，结果混进了两套分类学：
华北旧名单有 `Saxicola stejnegeri`（东亚石䳭），扩池后 `maurus`（黑喉石䳭）和
`stejnegeri` 各自参赛、各自的名次都掉到 150 名之外 → **丢了一只鸟**。

IOC 统一之后：同一个种只有一个名字参赛，无谓竞争消失。
本脚本就是量化"统一之后"的净效果。

输出：
  tools/dataset_traits/ioc_expand_diff.csv    逐区：留下/掉出/新进
  tools/dataset_traits/species_<region>_ioc150.csv   IOC 口径的新名单
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)

GEOMODEL_DIR = r"E:\code\code tools\geomodel"
DATA_ROOT = r"E:\code\code tools\data"
IOC_XLSX = os.path.join(TOOLS, "ioc", "Multiling_IOC_15.2.xlsx")
NAME_MAP = os.path.join(HERE, "species_name_map.csv")
OUT_CSV = os.path.join(HERE, "ioc_expand_diff.csv")

import pick_region_species as prs


def load_ioc():
    """→ (sci2ioc, zh2ioc, ioc2zh)。IOC_15.2 是有效名，Chinese 是中文名。"""
    import openpyxl
    wb = openpyxl.load_workbook(IOC_XLSX, read_only=True, data_only=True)
    it = wb["List"].iter_rows(values_only=True)
    hdr = [str(c or "").strip() for c in next(it)]
    i_sci, i_zh = hdr.index("IOC_15.2"), hdr.index("Chinese")
    sci2ioc, zh2ioc, ioc2zh = {}, {}, {}
    for r in it:
        if not r or not r[i_sci]:
            continue
        s = str(r[i_sci]).strip()
        z = str(r[i_zh] or "").strip()
        sci2ioc[s.lower()] = s
        ioc2zh[s] = z
        if z and z not in zh2ioc:
            zh2ioc[z] = s
    wb.close()
    return sci2ioc, zh2ioc, ioc2zh


def load_geomodel_index():
    p = os.path.join(GEOMODEL_DIR, "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt")
    idx = {}
    for i, l in enumerate(open(p, encoding="utf-8", errors="replace").read().splitlines()):
        parts = l.split("\t")
        if len(parts) >= 2:
            idx.setdefault(parts[1].strip().lower(), i)
    return idx


def load_ours():
    """我们的 1,494 个名字 + 中文 + 已有的 geomodel 索引。"""
    out = []
    with open(NAME_MAP, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append({"q": r["query_name"].strip(), "zh": (r["zh"] or "").strip(),
                        "gi_old": (r["geomodel_index"] or "").strip()})
    return out


def old_list(region):
    for p in (os.path.join(DATA_ROOT, region, f"species_{region}_150.csv"),
              os.path.join(DATA_ROOT, region, "gbif_availability.csv"),
              os.path.join(os.path.dirname(TOOLS), "dist", f"observebird_{region}_pack",
                           "tools", "species", f"species_{region}_150.csv")):
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig", newline="") as f:
                return [x["scientific_name"].strip() for x in csv.DictReader(f)
                        if x.get("scientific_name")]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()

    sci2ioc, zh2ioc, ioc2zh = load_ioc()
    gm = load_geomodel_index()
    ours = load_ours()
    print(f"IOC 有效名 {len(sci2ioc):,}   中文名 {len(zh2ioc):,}   geomodel 标签 {len(gm):,}")

    # ---- 把我们的名字统一到 IOC，并按 IOC 名去重 ----
    cand = {}          # ioc_name -> {zh, via, gi}
    stats = collections.Counter()
    for r in ours:
        q, zh = r["q"], r["zh"]
        ioc = sci2ioc.get(q.lower())
        how = "exact"
        if not ioc and zh:
            ioc = zh2ioc.get(zh)
            how = "via-zh" if ioc else "none"
        if not ioc:
            stats["未映射"] += 1
            continue
        gi = gm.get(ioc.lower())
        if gi is None and r["gi_old"]:
            try:
                gi = int(r["gi_old"])
            except ValueError:
                gi = None
        stats[how] += 1
        prev = cand.get(ioc)
        # 同一个 IOC 名有多条来源时，优先保留有 geomodel 索引的、中文名非空的
        if prev is None or (prev["gi"] is None and gi is not None) or \
                (not prev["zh"] and zh):
            cand[ioc] = {"zh": zh or ioc2zh.get(ioc, ""), "via": how,
                         "gi": gi, "src": q}
    scorable = {k: v for k, v in cand.items() if v["gi"] is not None}
    print(f"统一到 IOC 后：{len(cand)} 个种（去重后），其中能打分的 {len(scorable)}")
    print(f"  匹配方式: exact={stats['exact']}  via-zh={stats['via-zh']}  未映射={stats['未映射']}")
    print(f"  （对比：IOC 版之前能打分的只有 1,343）")

    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(
        os.path.join(GEOMODEL_DIR, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx"),
        so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    rows, summary = [], []
    for region in [r for r in prs.REGIONS if r != "national"]:
        probes = prs.REGIONS[region]
        acc = np.zeros(14082)
        for lat, lon in probes:
            for week in range(1, 49):
                acc += sess.run(None, {iname: np.array([[lat, lon, week]],
                                                       dtype=np.float32)})[0][0]
        acc /= (len(probes) * 48)
        scored = sorted(((float(acc[v["gi"]]), k, v) for k, v in scorable.items()),
                        reverse=True, key=lambda x: x[0])
        new = [s[1] for s in scored[: a.n]]
        # 旧名单 → IOC 名
        old_ioc, unmapped_old = [], 0
        for s in old_list(region):
            i = sci2ioc.get(s.lower())
            if not i:
                z = ""
                for r in ours:
                    if r["q"] == s:
                        z = r["zh"]; break
                i = zh2ioc.get(z) if z else None
            if i:
                old_ioc.append(i)
            else:
                unmapped_old += 1
        ns, os_ = set(new), set(old_ioc)
        keep = [s for s in new if s in os_]
        enter = [s for s in new if s not in os_]
        drop = [s for s in old_ioc if s not in ns]
        print(f"\n=== {region}  探针 {len(probes)} ===")
        print(f"  旧名单(→IOC) {len(old_ioc)} 种（{unmapped_old} 个映射不上）")
        print(f"  新名单 {len(new)} 种：留下 {len(keep)}  掉出 {len(drop)}  新进 {len(enter)}")
        if drop:
            print(f"  掉出: {drop}")
        if enter:
            print(f"  新进: {[(s, cand[s]['zh']) for s in enter]}")
        summary.append({"region": region, "keep": len(keep), "enter": len(enter),
                        "drop": len(drop), "unmapped_old": unmapped_old})
        for s in enter:
            rows.append({"region": region, "species": s, "zh": cand[s]["zh"],
                         "via": cand[s]["via"]})
        # 导出 IOC 口径名单
        with open(os.path.join(HERE, f"species_{region}_ioc{a.n}.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["rank", "ioc_name", "zh", "geomodel_mean", "via", "our_source"])
            for i, (sc, k, v) in enumerate(scored[: a.n], 1):
                w.writerow([i, k, v["zh"], round(sc, 6), v["via"], v["src"]])

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["region", "species", "zh", "via"])
        w.writeheader(); w.writerows(rows)
    print("\n=== 汇总（IOC 统一口径）===")
    print(f"{'区':<11s}{'留下':>6s}{'掉出':>6s}{'新进':>6s}{'旧名单映射不上':>14s}")
    for s in summary:
        print(f"{s['region']:<11s}{s['keep']:>6d}{s['drop']:>6d}{s['enter']:>6d}"
              f"{s['unmapped_old']:>14d}")
    tot = sum(s["enter"] for s in summary)
    print(f"\n五区合计换 {tot} 个名额 / 750 = {tot/750*100:.1f}%")
    print(f"-> {OUT_CSV}")


if __name__ == "__main__":
    main()

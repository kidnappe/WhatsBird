# -*- coding: utf-8 -*-
"""把 geomodel 蒸馏成分区×月份先验查表（端侧重排用）——docs/handoff/root/RERANK-DEVICE.md §3 方案 A。

**为什么**：geomodel 是 (lat,lon,week)→14,082 种的全局模型，端上区域固定，
只需要 150 类 × 12 月。端上查表微秒级、零依赖、离线可 100% 验证；
完整模型每次会话推理是备胎（方案 B，INT8 TFLite 4.0MB 现成）。

**类表来源**：优先 checkpoint best.pt 的 species 列（与模型输出索引严格同序），
没有 checkpoint 时回退 dataset.csv 的 sorted(scientific_name)（train.py 同口径）——
**表与类表必须同源同序**，eval_report 加载时逐项断言。

**采样点**（--grid）：1 = bbox 中心；9 = bbox 3×3 分位点（保留区内空间差异，
对抗"区中心抹平 +1.2 收益"的风险）。月份刻度与 eval_report 同口径：week=(m-1)*4+3。

用法:
  python tools/distill_prior_table.py --regions huabei huazhong huanan_sw dongbei --grid 1
  python tools/distill_prior_table.py --regions huabei --grid 9
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
GEOMODEL = r"E:\code\code tools\geomodel"
ONNX = os.path.join(GEOMODEL, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx")
LABELS = os.path.join(GEOMODEL, "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt")
CODE_TOOLS = r"E:\code\code tools"


def log(m):
    print(f"{time.strftime('%H:%M:%S')} [distill] {m}", flush=True)


def species_of(region, ckpt_path=None):
    """类表 = dataset.csv 的 sorted(scientific_name)（train.py 同口径，数据真值）。

    教训（2026-10-06 实测）：按"最新 checkpoint 找类表"会在 huazhong 抓到
    huazhong_mnv4_300（9.2k 行早期实验，145 类）——名字带 300 的旧实验盖住了
    真值 150 类。数据才是类表的唯一真值；checkpoint 只做一致性核对。
    """
    ds = os.path.join(CODE_TOOLS, "data", region, "dataset.csv")
    with open(ds, encoding="utf-8-sig", newline="") as f:
        sp = sorted({r["scientific_name"].strip() for r in csv.DictReader(f)
                     if r.get("scientific_name")})
    if ckpt_path and os.path.exists(ckpt_path):
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        msp = list(ck.get("species") or [])
        if msp and msp != sp:
            log(f"⚠ {region}: checkpoint 类表 {len(msp)} 类与数据类表 {len(sp)} 类"
                f"不一致（以数据为准，差异 {len(set(sp) - set(msp))} 种）——"
                f"该 checkpoint 是旧数据训的，A/B 评它时注意口径")
    return sp, ds


def sample_points(bbox, grid):
    la0, la1, lo0, lo1 = bbox
    if grid == 1:
        return [[round((la0 + la1) / 2, 3), round((lo0 + lo1) / 2, 3)]]
    if grid == 9:
        qs = [0.25, 0.5, 0.75]
    elif grid == 25:            # 5x5
        qs = [0.1, 0.3, 0.5, 0.7, 0.9]
    else:                       # 49 = 7x7
        qs = [i / 8 for i in range(1, 8)]
    return [[round(la0 + (la1 - la0) * qa, 3), round(lo0 + (lo1 - lo0) * qo, 3)]
            for qa in qs for qo in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--grid", type=int, choices=[1, 9, 25, 49], default=1)
    ap.add_argument("--out", default=os.path.join(TOOLS, "..", "assets", "prior"))
    ap.add_argument("--models-root", default=os.path.join(CODE_TOOLS, "models"))
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    gname = sess.get_inputs()[0].name
    labels = [l.strip() for l in open(LABELS, encoding="utf-8", errors="replace")]
    gidx = {}
    for i, line in enumerate(labels):
        p = line.split("\t")
        key = (p[1] if len(p) >= 2 else line).strip().lower()
        gidx.setdefault(key, i)

    regions = json.load(open(os.path.join(TOOLS, "regions.json"), encoding="utf-8"))
    weeks = [(m - 1) * 4 + 3 for m in range(1, 13)]     # 与 eval_report 同口径
    os.makedirs(a.out, exist_ok=True)

    for region in a.regions:
        bbox = regions[region]["bbox"]
        pts = sample_points(bbox, a.grid)
        ckpt = None
        for cand in (f"{region}_mnv4_rec", f"{region}_mnv4_300", f"{region}_mnv4"):
            p = os.path.join(a.models_root, cand, "best.pt")
            if os.path.exists(p):
                ckpt = p
                break
        species, src = species_of(region, ckpt)
        # 物种 → geomodel 输出索引（找不到就报错退出：没问到 ≠ 没有）
        col, missing = {}, []
        for s in species:
            j = gidx.get(s.lower())
            if j is None:
                missing.append(s)
            else:
                col[s] = j
        if missing:
            sys.exit(f"{region}: {len(missing)} 种没有 geomodel 索引，拒绝静默降级: "
                     f"{missing[:6]}")

        # 一次 batch 跑全部 (点 × 12 月)；prior[month][point][class]
        inp = np.array([[p[0], p[1], w] for p in pts for w in weeks],
                       dtype=np.float32)
        out = sess.run(None, {gname: inp})[0]          # [npts*12, 14082]
        prior = {str(m): [[round(float(out[p * 12 + mi][col[s]]), 6)
                           for s in species] for p in range(len(pts))]
                 for mi, m in enumerate(range(1, 13))}

        doc = {"region": region, "source": "geomodel", "grid": a.grid,
               "points": pts, "months": list(range(1, 13)),
               "species": species,
               "week_convention": "BirdNET 1..48, month m -> (m-1)*4+3",
               "class_source": src, "prior": prior}
        out_path = os.path.join(a.out, f"{region}_geomodel_g{a.grid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        log(f"{region}: {len(species)} 类 × {len(pts)} 点 × 12 月 -> "
            f"{out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
        if "Passer montanus" in species:               # 抽查：常见留鸟 7 月先验
            ci = species.index("Passer montanus")
            log(f"  抽查 Passer montanus 7月@中心 = {prior['7'][len(pts)//2][ci]:.4f}")


if __name__ == "__main__":
    main()

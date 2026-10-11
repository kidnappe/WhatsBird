"""Pick the species list for a regional model using the geomodel.

The geomodel answers "how likely is species X at (lat, lon) in week W". We
average that over a year for a set of probe locations, intersect with the
author's 373-species China list, and keep the top N. That gives a defensible
"common birds of this region" list without hand-picking.

Output: CSV with scientific_name, common_name, score, plus the 373-list index
so labels stay compatible with the released BirdNet head.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_HERE = os.path.dirname(os.path.abspath(__file__))
# 主开发机路径优先；移动包里 geomodel 在 tools/geomodel、373 基准名单在
# tools/species/species_list_373.csv，两者都有回退，包内再生成清单才可用
GEOMODEL = r"E:\code\code tools\geomodel"
BASE_LIST = os.path.join(
    r"E:\code\code tools\bird-cn-373\code\Bird-Classification-main",
    "species_list.csv")
if not os.path.isfile(BASE_LIST):
    BASE_LIST = os.path.join(_HERE, "species", "species_list_373.csv")
if not os.path.isdir(GEOMODEL):
    GEOMODEL = os.path.join(_HERE, "geomodel")

# 区域名与 bbox 以 tools/regions.json v2 (2026-10-04) 为准。探针点决定物种清单，
# 下载与 App 路由用的是 bbox —— 探针必须落在对应 bbox 内，否则清单和下载数据错位。
# 注意 bbox 是矩形近似：华中框(31-37N)实际=豫+鄂北（湘/鄂中南落 huanan_sw 框），
# 华东框(27-35N)不含鲁北/闽中南（鲁北在华北框、闽中南不在任何框内）。
REGIONS = {
    # name: list of (lat, lon) probe points
    "huabei": [(39.90, 116.41), (39.13, 117.20), (38.04, 114.51),
               (37.87, 112.55), (41.80, 123.43), (36.07, 120.38)],
    "dongbei": [(45.80, 126.53), (41.80, 123.43), (43.82, 125.32)],
    "huazhong": [(36.10, 114.35),  # 安阳
                 (34.75, 113.63),  # 郑州
                 (32.99, 112.53),  # 南阳
                 (32.63, 110.80),  # 十堰
                 (32.13, 114.07),  # 信阳
                 (32.04, 112.14)],  # 襄阳
    "huadong": [(34.26, 117.19),  # 徐州（鲁南/苏北）
                (32.06, 118.80),  # 南京
                (31.82, 117.23),  # 合肥
                (31.23, 121.47),  # 上海
                (30.27, 120.15),  # 杭州
                (28.45, 117.94),  # 上饶（赣东北）
                (28.00, 120.70)],  # 温州（浙南）
    "huanan_sw": [(30.57, 104.07),  # 成都
                  (29.56, 106.55),  # 重庆
                  (28.23, 112.94),  # 长沙
                  (26.65, 106.63),  # 贵阳
                  (25.04, 102.72),  # 昆明
                  (23.13, 113.26),  # 广州
                  (22.82, 108.32),  # 南宁
                  (20.04, 110.20)],  # 海口
    "national": [(39.90, 116.41), (23.13, 113.26), (30.57, 104.07),
                 (45.80, 126.53), (22.54, 114.06), (29.56, 106.55)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="huabei", choices=list(REGIONS))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--geomodel-dir", default=None,
                    help="覆盖 geomodel 目录（pipeline_region 传包内 tools/geomodel）")
    a = ap.parse_args()

    cands = [p for p in (a.geomodel_dir, GEOMODEL, os.path.join(_HERE, "geomodel")) if p]
    geomodel_dir = next(
        (p for p in cands if os.path.isfile(os.path.join(
            p, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx"))), None)
    if not geomodel_dir:
        sys.exit("找不到 geomodel（BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx），"
                 "检查 --geomodel-dir 或 tools/geomodel/")

    import onnxruntime as ort
    from PIL import Image  # noqa: F401  (ensures deps present early)

    with open(BASE_LIST, encoding="utf-8") as f:
        china = list(csv.DictReader(f))
    print(f"基准名单(作者 373 种): {len(china)} 种\n")

    # geomodel label index
    lab = os.path.join(geomodel_dir,
                       "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt")
    with open(lab, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    idx = {}
    for i, line in enumerate(lines):
        p = line.split("\t")
        if len(p) >= 2:
            idx.setdefault(p[1].strip().lower(), i)
    print(f"geomodel 标签: {len(lines)} 类\n")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        os.path.join(geomodel_dir,
                     "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx"),
        so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    probes = REGIONS[a.region]
    print(f"区域 {a.region} 探针 {len(probes)} 个 × 48 周 = {len(probes)*48} 次推理")
    acc = np.zeros(len(lines))
    for lat, lon in probes:
        for week in range(1, 49):
            acc += sess.run(None, {name: np.array([[lat, lon, week]],
                                                 dtype=np.float32)})[0][0]
    acc /= (len(probes) * 48)
    print("geomodel 推理完成\n")

    scored = []
    for i, r in enumerate(china):
        j = idx.get(r["scientific_name"].lower())
        if j is None:
            continue
        scored.append((float(acc[j]), i, r))
    scored.sort(reverse=True)

    print(f"=== {a.region} 排名 (前 40) ===")
    print(f"  {'#':>3} {'scientific_name':<26}{'中文':<12}{'geomodel 均值':>12}")
    for rank, (s, i, r) in enumerate(scored[:40], 1):
        print(f"  {rank:>3} {r['scientific_name']:<26}{r['common_name']:<12}{s:>12.4f}")

    sel = scored[: a.n]
    print(f"\n选中 {len(sel)} 种，分数区间 "
          f"{sel[-1][0]:.4f} ~ {sel[0][0]:.4f}")

    out = a.out or os.path.join(_HERE, f"species_{a.region}_{a.n}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["birdnet_index", "scientific_name", "common_name",
                    "geomodel_mean", "count_china"])
        for rank, (s, i, r) in enumerate(sel, 1):
            w.writerow([i, r["scientific_name"], r["common_name"],
                        round(s, 6), r.get("count_china", "")])
    print(f"\n-> {out}")

    meta = {
        "region": a.region, "n": len(sel), "probes": probes,
        "geomodel_version": "v3.0.4", "base_list": "Firefly-ZJ species_list.csv (373)",
        "selected": [{"rank": r, "sci": x[2]["scientific_name"],
                      "zh": x[2]["common_name"], "score": round(x[0], 6),
                      "birdnet_index": x[1]} for r, x in enumerate(sel, 1)],
    }
    mp = os.path.splitext(out)[0] + ".json"
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"-> {mp}")


if __name__ == "__main__":
    main()
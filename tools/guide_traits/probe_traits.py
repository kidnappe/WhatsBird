# -*- coding: utf-8 -*-
"""性状接入探针（一次性诊断）：全国 v2 labels(384) 到性状 CSV(1445) 的命中率 + 各列取值形态。

只读，不改任何数据。正式生成器见 build_traits_asset.py。
"""
import csv
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # tools/guide_traits/ -> 仓库根

TRAITS = ROOT / "tools/dataset_traits/china_bird_traits.csv"
NAMEMAP = ROOT / "tools/dataset_traits/species_name_map.csv"
OVERRIDES = ROOT / "tools/dataset_traits/name_overrides.csv"
LABELS = ROOT / "android/app/src/main/assets/models/labels_national_v2.txt"

# china_bird_traits.csv 是 30 列、6 组"雄性/雌性"重名列，必须按位置取
COL = {
    "order": 0, "family": 1, "zh": 2, "en": 3, "sci": 4, "endemic": 5,
    "mass_m": 6, "mass_f": 7,
    "len_m": 8, "len_f": 9,
    "bill_m": 10, "bill_f": 11,
    "wing_m": 12, "wing_f": 13,
    "tail_m": 14, "tail_f": 15,
    "tarsus_m": 16, "tarsus_f": 17,
    "diet": 18, "clutch": 19, "egg_size": 20, "egg_vol": 21,
    "nest_type": 22, "nest_site": 23, "flock": 24, "migrant": 25,
    "dist_range": 26, "provinces": 27,
}


def read_traits():
    rows = {}
    dup = 0
    with TRAITS.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)  # 表头
        for row in r:
            if len(row) <= COL["sci"]:
                continue
            sci = row[COL["sci"]].strip()
            if not sci:
                continue
            if sci in rows:
                dup += 1
            rows[sci] = row
    return rows, dup


def read_name_map():
    """species_name_map.csv: query_name(性状表名) -> gbif_name。返回反向表 gbif -> query。"""
    gbif2q = {}
    srcs = collections.Counter()
    with NAMEMAP.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            srcs[row.get("src", "")] += 1
            q = (row.get("query_name") or "").strip()
            g = (row.get("gbif_name") or "").strip()
            if q and g:
                gbif2q.setdefault(g, q)
    # 手工覆盖（name_overrides.csv 的 map 行）
    n_ov = 0
    with OVERRIDES.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("action") or "").strip() != "map":
                continue
            q = (row.get("query_name") or "").strip()
            g = (row.get("gbif_name") or "").strip()
            if q and g:
                gbif2q[g] = q
                n_ov += 1
    return gbif2q, srcs, n_ov


def main():
    traits, dup = read_traits()
    gbif2q, srcs, n_ov = read_name_map()
    labels = []
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split("\t")
            labels.append((p[0].strip(), p[1].strip() if len(p) > 1 else ""))

    print("=== 规模 ===")
    print(f"性状表物种: {len(traits)}（重复拉丁名 {dup}）")
    print(f"名字映射行: {sum(srcs.values())}  src 分布: {dict(srcs)}  手工 map 覆盖: {n_ov}")
    print(f"全国 v2 labels: {len(labels)}")

    direct, mapped, miss = [], [], []
    for sci, zh in labels:
        if sci in traits:
            direct.append(sci)
        elif sci in gbif2q and gbif2q[sci] in traits:
            mapped.append((sci, gbif2q[sci]))
        else:
            miss.append((sci, zh))

    print("\n=== 命中 ===")
    print(f"直接命中: {len(direct)}")
    print(f"经异名映射命中: {len(mapped)}")
    print(f"未命中: {len(miss)}  -> 覆盖率 {len(direct)+len(mapped)}/{len(labels)} "
          f"= {(len(direct)+len(mapped))/len(labels)*100:.1f}%")
    if mapped:
        print("\n-- 映射命中的样例(前 15) --")
        for a, b in mapped[:15]:
            print(f"   {a}  <-  {b}")
    if miss:
        print(f"\n-- 未命中清单({len(miss)}) --")
        for sci, zh in miss:
            print(f"   {zh}\t{sci}")

    print("\n=== 各列取值形态（全表抽样） ===")
    for key in ["endemic", "diet", "clutch", "egg_size", "nest_type", "nest_site",
                "flock", "migrant", "provinces", "mass_m", "len_m", "bill_m",
                "wing_m", "tail_m", "tarsus_m", "egg_vol", "dist_range"]:
        i = COL[key]
        vals = collections.Counter()
        na = 0
        for row in traits.values():
            v = row[i].strip() if i < len(row) else ""
            if v in ("", "NA", "na", "—", "-"):
                na += 1
            else:
                vals[v] += 1
        top = ", ".join(f"{k}×{v}" for k, v in vals.most_common(8))
        print(f"[{key:10}] 缺失 {na:4d} / 唯一值 {len(vals):4d} | {top}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""先验表 <-> 区域模型类表 对齐核对（2026-10-10 新增）。

背景：App 端 PriorReranker.kt 的 H1 修复把建表键与查表键统一成 lowercase()。
但 PC 端 eval_report.py 是**按位置**匹配，并断言先验表 species 与 checkpoint 类表
**逐位完全相同**（eval_report.py:277 `if list(tj["species"]) != list(species)`）。
也就是说：Python 侧靠"同源同序"保证对齐，Kotlin 侧靠**学名小写**对齐——
两套实现的对齐前提不同，必须实测 Kotlin 的小写匹配能不能真的命中。

本脚本只读 assets，不改任何东西。退出码 0 = 五区全部对齐。
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(REPO, "android", "app", "src", "main", "assets")
PRIOR_DIR = os.path.join(ASSETS, "prior")
MODEL_DIR = os.path.join(ASSETS, "models")

# 路由 key -> (先验表 key, 类表文件)；华南/西南共用合并模型（与 PriorReranker.tableKey 同口径）
REGIONS = [
    ("huabei", "huabei", "labels_huabei.txt"),
    ("huazhong", "huazhong", "labels_huazhong.txt"),
    ("huanan", "huanan_sw", "labels_huanan_sw.txt"),
    ("xinan", "huanan_sw", "labels_huanan_sw.txt"),
    ("huadong", "huadong", "labels_huadong.txt"),
    ("dongbei", "dongbei", "labels_dongbei.txt"),
]


def load_prior_species(prior_key):
    path = os.path.join(PRIOR_DIR, "%s_geomodel_g49.json" % prior_key)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["species"]


def load_label_sci(labels_file):
    """类表是 'sci\\tzh'；只取学名列。"""
    sci = []
    with open(os.path.join(MODEL_DIR, labels_file), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sci.append(line.split("\t")[0].strip())
    return sci


def check(region, prior_key, labels_file):
    prior = load_prior_species(prior_key)
    labels = load_label_sci(labels_file)

    out = {"region": region, "prior_key": prior_key, "labels_file": labels_file}

    # 1) 逐位完全相同？（Python 的前提；不同就说明两端根本不是同一份类表）
    out["exact_same_order"] = (prior == labels)
    out["exact_same_set"] = (set(prior) == set(labels))

    # 2) Kotlin 修复后：小写集合是否一致
    prior_lc = [s.lower() for s in prior]
    labels_lc = [s.lower() for s in labels]
    out["lc_same_set"] = (set(prior_lc) == set(labels_lc))

    # 3) 查表命中率（小写建表 -> 小写查表），这是 H1 修复后实际能命中的比例
    table = set(prior_lc)
    hit = sum(1 for s in labels_lc if s in table)
    out["n_prior"] = len(prior)
    out["n_labels"] = len(labels)
    out["hit"] = hit
    out["miss"] = labels_lc and [s for s in labels_lc if s not in table]

    # 4) 大小写不一致的条目数（>0 说明 PC 端"逐位相同"的前提在大小写上就不成立）
    out["case_mismatch"] = sum(1 for a, b in zip(prior, labels) if a != b) if len(prior) == len(labels) else -1

    return out


def main():
    rows = []
    for region, prior_key, labels_file in REGIONS:
        try:
            rows.append(check(region, prior_key, labels_file))
        except Exception as e:  # noqa: BLE001
            rows.append({"region": region, "error": repr(e)})

    print("%-9s %-10s %6s %6s %6s %6s %5s  %s" %
          ("region", "prior", "prior", "labels", "hit", "hit%", "case", "verdict"))
    print("-" * 78)

    ok = True
    seen_prior = {}
    for r in rows:
        if "error" in r:
            print("%-9s ERROR %s" % (r["region"], r["error"]))
            ok = False
            continue

        # huanan / xinan 共用同一张表，只判一次
        if r["prior_key"] in seen_prior:
            continue
        seen_prior[r["prior_key"]] = True

        pct = 100.0 * r["hit"] / max(1, r["n_labels"])
        verdict = []
        if not r["lc_same_set"]:
            verdict.append("LOWERCASE-SET-DIFF")
            ok = False
        if pct < 99.0:
            verdict.append("COVERAGE<99%")
            ok = False
        if not r["exact_same_order"]:
            # 不致命（小写匹配已覆盖），但要提醒：PC 端按位置匹配，靠的是这个前提
            verdict.append("PC-POSITION-PREMISE-UNVERIFIED")

        print("%-9s %-10s %6d %6d %6d %5.1f%%  %5d  %s" % (
            r["region"], r["prior_key"], r["n_prior"], r["n_labels"],
            r["hit"], pct, r["case_mismatch"], ",".join(verdict) or "OK"))

        miss = r["miss"]
        if isinstance(miss, list) and miss:
            print("           未命中前 8 个: %s" % ", ".join(miss[:8]))

    print("-" * 78)
    print("RESULT: %s" % ("PASS - 五区小写对齐，H1 修复后查表可命中"
                          if ok else "FAIL - 见上面标红项，Kotlin 小写匹配会漏种"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
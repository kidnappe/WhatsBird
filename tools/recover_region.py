# -*- coding: utf-8 -*-
"""区域数据恢复：补回被误删的行 + 用名字对照表修好取不到图的种。

**为什么需要它**

2026-10-05 发现华南缺 12 个种，其中 6 个本来有大量可用数据（白颊噪鹛 1,239 /
黑水鸡 962 / 黑领椋鸟 807 / 纯色山鹪莺 550 / 绿头鸭 519 / 红头潜鸭 61 条）：

    audit_taxon_keys.py 抓 GBIF 时遇到网络故障（SSL UNEXPECTED_EOF）
      → 写进审计表 rank="NONE"（和"这个分类单元真没 rank"分不清）
      → clean_dataset.py 当成"属级 key"
      → 逐行查 occurrence 的真实种，**查询也失败**
      → 全部判为"不属于名单"→ 删行
      → 结果：这 6 个种 0 张图，而 1,561 张图片文件还好端端留在 images/

另 5 个种（牛背鹭 / 金翅雀 / 金眶鸻 / 凤头鹰 / 黄苇鳽）是**名字对不上 GBIF**，
当时解析出 tk=None，现在 `tools/dataset_traits/species_name_map.json` 里都有 key 了。

**它做什么**（三步，顺序不能换）

  1. 跑 build_dataset_gbif.py
     · 已存在的图片文件会走 "skip" → **补登记一行**（不重新下载，file 名由 occurrence
       key 决定，不会重复）
     · 名字对不上的种，现在会先查对照表拿到 key → 正常下载
  2. 重跑 audit_taxon_keys.py（已改成失败写 FETCH_FAILED）
  3. 重跑 clean_dataset.py（已改成 FETCH_FAILED / 查询失败一律保留，不删）

**安全前提**：同一时间只能有一个下载任务（GBIF 限流 + 内存）。脚本会先检查，
有别的 build_dataset_gbif / train.py 在跑就直接退出，不动数据。

用法:
  python tools/recover_region.py --region huanan_sw
  python tools/recover_region.py --region huanan_sw --dry-run   # 只看会做什么
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
CODE_TOOLS = r"E:\code\code tools"
PY = sys.executable


def sh(cmd, env=None, dry=False):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, env=env).returncode


def busy_processes():
    """有没有别的下载/训练在跑（避免并行）。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=60).stdout or ""
    except Exception:
        return []
    hits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        for pat in ("build_dataset_gbif", "train.py", "auto_region_pipeline",
                    "pipeline_region", "fetch_inat_region"):
            if pat in line:
                hits.append((pat, line.strip()[:110]))
                break
    return hits


def snapshot(data_dir, names):
    """当前状态：有图种数 / 缺哪些种 / 总行数。"""
    have = collections.Counter()
    ds = os.path.join(data_dir, "dataset.csv")
    if os.path.exists(ds):
        with open(ds, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                s = (r.get("scientific_name") or "").strip()
                if s:
                    have[s] += 1
    miss = [n for n in names if have.get(n, 0) == 0]
    return {"rows": sum(have.values()), "species": len(have), "missing": miss}


def load_names(data_dir, region):
    for p in (os.path.join(data_dir, f"species_{region}_150.csv"),
              os.path.join(data_dir, "gbif_availability.csv"),
              os.path.join(REPO, "dist", f"observebird_{region}_pack", "tools",
                           "species", f"species_{region}_150.csv")):
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig", newline="") as f:
                return [r["scientific_name"].strip() for r in csv.DictReader(f)
                        if r.get("scientific_name")], p
    return [], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--per-species", type=int, default=300)
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    regions = json.load(open(os.path.join(TOOLS, "regions.json"), encoding="utf-8"))
    if a.region not in regions:
        sys.exit(f"regions.json 里没有 {a.region}")
    bbox = regions[a.region]["bbox"]
    data_dir = os.path.join(CODE_TOOLS, "data", a.region)

    names, npath = load_names(data_dir, a.region)
    if not names:
        sys.exit(f"找不到 {a.region} 的物种清单")
    print(f"区域 {a.region}  bbox={bbox}  数据目录 {data_dir}")
    print(f"清单 {len(names)} 种 <- {npath}")

    before = snapshot(data_dir, names)
    print(f"现在: {before['rows']:,} 行 / {before['species']} 种 / 缺 {len(before['missing'])} 种")
    if before["missing"]:
        print(f"  缺的种: {before['missing']}")

    busy = busy_processes()
    busy = [b for b in busy if a.region not in b[1]]
    if busy:
        print("\n⚠️ 有别的任务在跑，拒绝并行（GBIF 限流 + 内存约束）：")
        for pat, line in busy:
            print(f"     [{pat}] {line}")
        print("   等它跑完再来。")
        sys.exit(2)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "HF_ENDPOINT": "https://hf-mirror.com"}
    ds = os.path.join(data_dir, "dataset.csv")
    audit = os.path.join(data_dir, "gbif_taxon_audit.csv")

    # ---- 1) 下载（补登记 + 用对照表修名字）----
    print("\n=== 1/3 下载（已存在的图片走 skip 补登记；名字对不上的走对照表）===")
    rc = sh([PY, "-u", os.path.join(TOOLS, "build_dataset_gbif.py"),
             "--species", os.path.join(data_dir, f"species_{a.region}_150.csv")
             if os.path.exists(os.path.join(data_dir, f"species_{a.region}_150.csv"))
             else npath,
             "--out", data_dir,
             "--per-species", str(a.per_species), "--max-pages", str(a.max_pages),
             "--workers", str(a.workers), "--api-timeout", "60",
             "--bbox", *[str(x) for x in bbox], "--nice"], env, a.dry_run)
    if rc != 0:
        sys.exit(f"下载失败 exit={rc}")

    # ---- 2) 重审计（失败会写 FETCH_FAILED，不再写 NONE）----
    print("\n=== 2/3 重审计 taxon key ===")
    rc = sh([PY, os.path.join(TOOLS, "audit_taxon_keys.py"),
             "--cache", os.path.join(data_dir, "gbif_prefetch_cache.json"),
             "--out", audit], env, a.dry_run)
    if rc != 0:
        sys.exit(f"审计失败 exit={rc}")

    # ---- 3) 重清洗（FETCH_FAILED / 查询失败一律保留）----
    print("\n=== 3/3 重清洗（fail-safe）===")
    rc = sh([PY, os.path.join(TOOLS, "clean_dataset.py"),
             "--dataset", ds, "--audit", audit, "--species-list", npath],
            env, a.dry_run)
    if rc != 0:
        sys.exit(f"清洗失败 exit={rc}")

    after = snapshot(data_dir, names)
    print(f"\n=== 结果 ===")
    print(f"  {before['rows']:,} 行 / {before['species']} 种  ->  "
          f"{after['rows']:,} 行 / {after['species']} 种")
    print(f"  恢复 {after['rows'] - before['rows']:,} 行、{after['species'] - before['species']} 个种")
    still = after["missing"]
    if still:
        print(f"  仍缺 {len(still)} 种: {still}")
        print("  （GBIF 框内确实没数据的 → 只能走 iNat 补量：fetch_inat_region.py）")
    print("\n⚠️ 类别数变了 → labels / 图鉴要重生成，然后重训该区。")
    print(f"   清单: {os.path.join(data_dir, f'species_{a.region}_150.csv')}")


if __name__ == "__main__":
    main()

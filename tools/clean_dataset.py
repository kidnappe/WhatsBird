"""Fix the rows whose GBIF taxon key was not species-level.

build_dataset_gbif.py labels every image with the *requested* name, so a species
that GBIF only answers with a genus key (rank=GENUS) gets a class full of other
species. audit_taxon_keys.py finds those classes; this script re-reads each of
their occurrences from GBIF and uses the determination GBIF actually recorded:

  * species in our regional list  -> relabel the row to that species
  * species missing / outside list -> drop the row (genus-level ID is not a label)

Then it drops any occurrence that still ends up under two different species,
because grouped_split partitions per species and such a row leaks one individual
into both train and val.

Run it after the download finishes; dataset.csv is rewritten in place and the
previous version is kept as dataset.csv.<timestamp>.bak.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import json
import os
import shutil
import time
import urllib.request

API = "https://api.gbif.org/v1/occurrence/{}"


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def get_occ(key, tries=3):
    for a in range(tries):
        try:
            with urllib.request.urlopen(API.format(key), timeout=45) as r:
                return json.load(r)
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(2 * (a + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--audit", required=True, help="gbif_taxon_audit.csv")
    ap.add_argument("--species-list", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    meta = {}
    with open(a.species_list, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            meta[r["scientific_name"]] = r

    genus_species = set()
    fetch_failed = []
    with open(a.audit, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # ⚠️ FETCH_FAILED = 审计时**没问到** GBIF（网络故障），不是"这个分类单元无效"。
            # 这种行一律不动 —— 宁可不处理，也绝不因为一次网络抖动删数据。
            # 实测：华南 6 个种（共 4,138 条可用记录）就是这样被删光的，
            # 而 1,561 张图还留在 images/ 里。要处理请先重跑 audit_taxon_keys.py。
            if r.get("rank") == "FETCH_FAILED":
                fetch_failed.append(r["scientific_name"])
                continue
            if r["rank"] != "SPECIES":
                genus_species.add(r["scientific_name"])
    if fetch_failed:
        log(f"⚠️ {len(fetch_failed)} 个种审计抓取失败（rank=FETCH_FAILED）→ "
            f"它们的行一律保留、不参与改标/删除: {fetch_failed[:10]}")
    log(f"属级 key 的种: {sorted(genus_species)}")

    rows = list(csv.DictReader(open(a.dataset, encoding="utf-8")))
    log(f"读入 {len(rows)} 行 / {len({r['scientific_name'] for r in rows})} 种")

    suspect = [r for r in rows if r["scientific_name"] in genus_species]
    occs = sorted({r["occurrence_key"] for r in suspect})
    log(f"需复核 {len(suspect)} 行 / {len(occs)} 个 observation")

    truth = {}
    occ_failed = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (k, d) in enumerate(zip(
                occs, ex.map(get_occ, occs)), 1):
            if d is None:
                # None = 没查到（网络失败）。必须和 ""（查到了但没有 species 字段）
                # 区分开：原来两者都变成 ""，于是"查询失败"被当成"不属于名单里的种"
                # → 行被删。这是同一次事故的第二半。
                occ_failed += 1
                truth[k] = None
            else:
                truth[k] = d.get("species") or ""
            if i % 50 == 0:
                log(f"  {i}/{len(occs)}")
    if occ_failed:
        log(f"⚠️ {occ_failed}/{len(occs)} 个 observation 查询失败 → 对应行一律保留")

    relabel, drop, kept_unresolved = 0, 0, 0
    out = []
    for r in rows:
        if r["scientific_name"] in genus_species:
            new = truth.get(r["occurrence_key"])
            if new is None:
                # 查询失败 → 保留原行（宁可留一个可能属级的标签，也不删数据）
                kept_unresolved += 1
                out.append(r)
                continue
            if new in meta and new != r["scientific_name"]:
                r["scientific_name"] = new
                r["common_name"] = meta[new]["common_name"]
                r["birdnet_index"] = meta[new]["birdnet_index"]
                relabel += 1
                out.append(r)
                continue
            drop += 1
            continue
        out.append(r)
    log(f"改标 {relabel} 行, 删除 {drop} 行, 因查询失败保留 {kept_unresolved} 行")

    byocc = collections.defaultdict(set)
    for r in out:
        byocc[r["occurrence_key"]].add(r["scientific_name"])
    amb = {k for k, v in byocc.items() if len(v) > 1}
    if amb:
        before = len(out)
        out = [r for r in out if r["occurrence_key"] not in amb]
        log(f"跨种共享 observation {len(amb)} 个 -> 删除 {before - len(out)} 行")

    kept = collections.Counter(r["scientific_name"] for r in out)
    log(f"结果: {len(out)} 行 / {len(kept)} 种 "
        f"(中位 {sorted(kept.values())[len(kept)//2]} 张/种, "
        f"最少 {min(kept.values())})")
    thin = [s for s, n in kept.items() if n < 40]
    if thin:
        log(f"  <40 张的种 {len(thin)}: {thin[:10]}")

    bak = a.dataset + "." + time.strftime("%Y%m%d-%H%M%S") + ".bak"
    shutil.copy2(a.dataset, bak)
    tmp = a.dataset + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, a.dataset)
    log(f"-> {a.dataset}  (备份 {bak})")


if __name__ == "__main__":
    main()

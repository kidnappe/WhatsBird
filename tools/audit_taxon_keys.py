"""Audit the GBIF taxon keys used by build_dataset_gbif.py.

species/match with strict=false silently falls back to a higher rank, so a name
GBIF does not carry at species level (e.g. "Ardea coromanda") resolves to the
GENUS key, and every occurrence of every species in that genus gets labelled
with the one wrong name. A genus-level class is poison: it splits one real
species across two classes and inflates the class count.

Reads the prefetch cache written during the download and reports, per species,
the rank GBIF actually answered with and the name that key points back to.

    python tools/audit_taxon_keys.py --cache "E:\\code\\code tools\\data\\huabei\\gbif_prefetch_cache.json"
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import time
import urllib.request

API = "https://api.gbif.org/v1/species/{}"


def get(url, tries=3):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            if a == tries - 1:
                print(f"    give up {type(e).__name__} {e}", flush=True)
                return None
            time.sleep(2 ** a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cache = json.load(open(a.cache, encoding="utf-8"))
    items = [(sci, v.get("tk")) for sci, v in sorted(cache.items())]
    print(f"{len(items)} species in cache", flush=True)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(get, API.format(tk)): (sci, tk)
                for sci, tk in items if tk}
        for i, f in enumerate(cf.as_completed(futs), 1):
            sci, tk = futs[f]
            d = f.result()          # None = 抓取失败（重试完也没成功），**不是**"没有 rank"
            rows.append({
                "scientific_name": sci,
                "taxon_key": tk,
                "gbif_scientific_name": (d or {}).get("scientificName", ""),
                # ⚠️ 抓取失败必须和"真的没有 rank"区分开。
                # 原来写成 `d.get("rank", "NONE")`，于是一次网络抖动被记成 rank=NONE，
                # 下游 clean_dataset.py 把它当成"属级 key"→ 把整种的行删光。
                # 实测华南 6 个种（白颊噪鹛 1239 / 黑水鸡 962 / 黑领椋鸟 807 /
                # 纯色山鹪莺 550 / 绿头鸭 519 / 红头潜鸭 61 条可用）被这样删掉，
                # 而 1,561 张图还好端端留在 images/ 里。
                "rank": ((d or {}).get("rank") or "NONE") if d is not None
                        else "FETCH_FAILED",
                "status": (d or {}).get("status", ""),
                "n_records": cache[sci].get("n", 0),
            })
            if i % 30 == 0:
                print(f"  {i}/{len(futs)}", flush=True)

    def binom(s):
        return s.split(" (")[0].strip().lower()

    rows.sort(key=lambda r: (r["rank"] != "SPECIES", r["scientific_name"]))
    # FETCH_FAILED 单列：它不是"这个分类单元有问题"，是"我们没问到"。
    # 下游 clean_dataset.py 必须跳过它、宁可不动也不能删数据。
    failed = [r for r in rows if r["rank"] == "FETCH_FAILED"]
    bad = [r for r in rows if r["rank"] not in ("SPECIES", "FETCH_FAILED")]
    if failed:
        print(f"\n⚠️ 抓取失败 {len(failed)} 个种（rank=FETCH_FAILED）—— 网络问题，"
              f"**不要**据此删数据；重跑本脚本即可：")
        for r in failed[:20]:
            print(f"  {r['scientific_name']:<28} key={r['taxon_key']:<10} "
                  f"n={r['n_records']}")
        if len(failed) > 20:
            print(f"  ...（共 {len(failed)} 个）")
    print(f"\nrank != SPECIES: {len(bad)}")
    for r in bad:
        print(f"  {r['scientific_name']:<28} key={r['taxon_key']:<10} "
              f"rank={r['rank']:<8} gbif={r['gbif_scientific_name']!r} "
              f"n={r['n_records']}")
    rows.sort(key=lambda r: (r["rank"] != "SPECIES", r["scientific_name"]))
    mismatch = [r for r in rows if r["rank"] == "SPECIES"
                and binom(r["gbif_scientific_name"]) != binom(r["scientific_name"])]
    print(f"genus-mismatch at SPECIES rank: {len(mismatch)}")
    for r in mismatch[:20]:
        print(f"  {r['scientific_name']:<28} -> {r['gbif_scientific_name']}")

    out = a.out or os.path.join(os.path.dirname(a.cache), "gbif_taxon_audit.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

"""Probe iNat for openly-licensed observations in a region + species list.

Answers: how many CC0/CC-BY licensed observations could iNat legally add
per species, on top of what GBIF already gave us?

Usage:
  python tools/probe_inat.py --species "E:/code/code tools/data/huazhong/species_huazhong_150.csv" \
      --bbox 31 37 105 117 --out "E:/code/code tools/data/huazhong/inat_probe.csv"

iNat API: GET https://api.inaturalist.org/v1/observations
  taxon_name=  swlat/swlng/nelat/nelng=  license=cc_by,cc0  per_page=0 (count only)
The quality_grade=research filter is used: research-grade are the ones with
community-confirmed IDs and usable photos.
"""
import argparse
import csv
import os
import time
import urllib.parse
import urllib.request

API = "https://api.inaturalist.org/v1/observations"
OPEN_LICENSES = "cc0,cc_by"  # 合规子集：NC/all-rights-reserved 不碰


def fetch_count(params: dict, retries: int = 4) -> int | None:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "observebird-probe/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json_load(r.read())
            return int(data.get("total_results", 0))
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def json_load(b: bytes):
    import json
    return json.loads(b.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, help="物种清单 csv（scientific_name 列）")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.species, encoding="utf-8-sig") as f:
        species = [r["scientific_name"] for r in csv.DictReader(f)]
    swlat, nelat, swlng, nelng = a.bbox[0], a.bbox[1], a.bbox[2], a.bbox[3]

    base = {"swlat": swlat, "nelat": nelat, "swlng": swlng, "nelng": nelng,
            "quality_grade": "research", "per_page": 0}
    # 地区大盘：开放许可 vs 全部（研究级）
    all_lic = fetch_count({**base, "iconic_taxa": "Aves"})
    open_lic = fetch_count({**base, "iconic_taxa": "Aves", "license": OPEN_LICENSES})
    print(f"地区鸟类大盘: 研究级 {all_lic} | 开放许可(CCO/CC-BY) {open_lic} "
          f"({(100*open_lic/all_lic if all_lic else 0):.0f}%)")

    rows, total_open, hit = [], 0, 0
    for i, sci in enumerate(species, 1):
        n_open = fetch_count({**base, "taxon_name": sci, "license": OPEN_LICENSES})
        n_any = fetch_count({**base, "taxon_name": sci})
        rows.append({"scientific_name": sci, "inat_open": n_open,
                     "inat_any": n_any})
        if n_open:
            total_open += n_open
            hit += 1
        print(f"{i:>3}/{len(species)}  {sci:<32} 开放 {n_open if n_open is not None else '?':>4}  "
              f"全部 {n_any if n_any is not None else '?':>4}", flush=True)
        time.sleep(1.0)  # iNat API 礼貌节流

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scientific_name", "inat_open", "inat_any"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n汇总: 有开放许可记录的种 {hit}/{len(species)} | 开放记录总量 {total_open}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

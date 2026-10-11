"""Fetch a small labelled bird image set from the iNaturalist API.

Serves two purposes at once:
  * INT8 quantisation calibration set (needs images, not labels)
  * accuracy evaluation of the image model with/without geomodel reranking
    (needs correct species labels)

Images are saved short-side 256 JPEG so the model preprocessing
(Resize 256 -> CenterCrop 224 -> ImageNet normalise) can be applied on load.

Usage:
  python fetch_inat.py --species 60 --per-species 10 --out data/inat_sample
  python fetch_inat.py --species 60 --per-species 10 --out data/x --token XXX
"""
import argparse
import csv
import io
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

API = "https://api.inaturalist.org/v1/observations"
UA = "Mozilla/5.0 (compatible; ObserveBird-PoC/0.1)"
COMMERCIAL_OK = {"cc0", "cc-by"}


def http_json(url, token=None, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Token {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url, path, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return data


def save_short_side_256(raw, path):
    """Decode, EXIF-orient, resize short side to 256, save as JPEG q90."""
    img = Image.open(io.BytesIO(raw))
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) > 256:
        s = 256 / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    img.save(path, "JPEG", quality=90)
    return img.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", type=int, default=60)
    ap.add_argument("--per-species", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--species-csv", default=None)
    ap.add_argument("--token", default=os.environ.get("INAT_TOKEN"))
    ap.add_argument("--workers", type=int, default=10, help="parallel image downloads")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--restart", action="store_true", help="ignore existing labels.csv")
    ap.add_argument("--strict-license", action="store_true",
                    help="skip any photo whose license is not cc0/cc-by")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = a.species_csv or os.path.join(
        r"E:\code\code tools\bird-cn-373\code\Bird-Classification-main",
        "species_list.csv")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"species source: {len(rows)} 种 ({csv_path})")

    # Spread the sample across the taxonomy instead of taking the first N.
    step = max(1, len(rows) // a.species)
    picked = rows[::step][: a.species]
    print(f"本次取样 {len(picked)} 种，每种最多 {a.per_species} 张\n")

    os.makedirs(a.out, exist_ok=True)
    labels_path = os.path.join(a.out, "labels.csv")

    # Resume: species already recorded are skipped so a run can be topped up.
    done_sci = set()
    if os.path.exists(labels_path) and not a.restart:
        with open(labels_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_sci.add(row["scientific_name"])
        print(f"断点续传: 已完成 {len(done_sci)} 种，跳过\n")

    fh = open(labels_path, "a" if (done_sci and not a.restart) else "w",
              newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not (done_sci and not a.restart):
        w.writerow(["file", "scientific_name", "common_name", "license_code", "attribution"])

    total = kept = skipped_lic = failed = 0
    lic_seen = {}

    def fetch_one(job):
        """Download one observation photo."""
        idx, sci, zh, ph, strict = job
        lic = (ph.get("license_code") or "").lower()
        if strict and lic not in COMMERCIAL_OK:
            return ("skip_lic", lic)
        url = (ph.get("url") or "").replace("/square.jpg", "/medium.jpg")
        if not url:
            return ("fail", None)
        safe = f"{idx:03d}_{sci.replace(' ', '_')}_{ph.get('id')}"
        dst = os.path.join(a.out, safe + ".jpg")
        try:
            raw = download(url, dst)
            save_short_side_256(raw, dst)
            return ("ok", (os.path.basename(dst), sci, zh, lic,
                           (ph.get("attribution") or "").replace("\n", " ")[:200]))
        except Exception:
            if os.path.exists(dst):
                os.remove(dst)
            return ("fail", None)

    pool = ThreadPoolExecutor(max_workers=a.workers)
    for i, row in enumerate(picked, 1):
        sci, zh = row["scientific_name"], row["common_name"]
        if sci in done_sci:
            print(f"  [{i:>3}/{len(picked)}] {sci:<26} {zh:<10} 已完成,跳过", flush=True)
            continue
        qs = urllib.parse.urlencode({
            "taxon_name": sci, "photos": "true", "per_page": 30,
            "quality_grade": "research", "order_by": "votes",
        })
        try:
            data = http_json(f"{API}?{qs}", a.token)
        except Exception as e:
            print(f"  [{i:>3}/{len(picked)}] {sci:<26} API 失败: {e}", flush=True)
            time.sleep(a.sleep)
            continue

        jobs = []
        for obs in data.get("results", []):
            for ph in obs.get("photos") or []:
                if len(jobs) >= a.per_species:
                    break
                jobs.append((i, sci, zh, ph, a.strict_license))
            if len(jobs) >= a.per_species:
                break
        for ph in (p for _, _, _, p, _ in jobs):
            lic = (ph.get("license_code") or "").lower()
            lic_seen[lic] = lic_seen.get(lic, 0) + 1

        got = 0
        for status, payload in pool.map(fetch_one, jobs):
            total += 1
            if status == "ok":
                w.writerow(payload)
                kept += 1
                got += 1
            elif status == "skip_lic":
                skipped_lic += 1
            else:
                failed += 1

        print(f"  [{i:>3}/{len(picked)}] {sci:<26} {zh:<10} +{got:>2}  (累计 {kept})", flush=True)
        fh.flush()
        time.sleep(a.sleep)
    pool.shutdown()

    fh.close()
    total_files = len([f for f in os.listdir(a.out) if f.endswith(".jpg")])
    print(f"\n本次新增 {kept} 张 / 尝试 {total} / 失败 {failed} / 因license跳过 {skipped_lic}")
    print(f"目录内图片总数: {total_files}  ({[math]::sum(os.path.getsize(os.path.join(a.out,f)) for f in os.listdir(a.out) if f.endswith('.jpg'))/1048576:.1f} MB)")
    print(f"labels -> {labels_path}")
    print("license 分布(本次观察):")
    for k, v in sorted(lic_seen.items(), key=lambda x: -x[1]):
        mark = "OK可商用" if k in COMMERCIAL_OK else "  "
        print(f"  {mark} {k or '(空)':<12} {v}")


if __name__ == "__main__":
    main()

"""Build a regional bird dataset from the iNaturalist API.

Downloads `medium` photos for each species in the region list, stores them
short-side 256, and records everything needed later for leakage-free splitting
and geomodel reranking.

Per image we keep: observation id (for grouping), lat/lon + observed date (for
the geomodel), and license (recorded, not enforced).

Rate limiting follows iNat's published guidance: ~1 request/second, and on HTTP
429 back off for a full minute. Runs are resumable.

Usage:
  python build_dataset.py --species species_huabei_150.csv --out <dir> --per-species 300
"""
import argparse
import csv
import io
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from PIL import Image, ImageOps

API = "https://api.inaturalist.org/v1/observations"
UA = "ObserveBird-PoC/0.1 (dataset build; contact: local)"
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def api_get(url, tries=4):
    """One API call with iNat-friendly pacing and 429 backoff."""
    delay = 1.1
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 65
                log(f"    429 限流 -> 退避 {wait}s (第 {attempt+1} 次)")
                time.sleep(wait)
                delay = 1.5
                continue
            log(f"    HTTP {e.code}")
            time.sleep(5)
        except Exception as e:
            log(f"    失败: {type(e).__name__}")
            time.sleep(5)
    return None


def fetch_species_page(sci, page, lat0=None, lon0=None):
    """One page of observations. per_page=30 measured as the sweet spot:
    per_page=200 returns ~13 MB and takes ~250 s on this link."""
    qs = {"taxon_name": sci, "photos": "true", "per_page": 30,
          "quality_grade": "research", "order_by": "votes", "page": page + 1}
    if lat0 is not None:
        qs.update(geopoint=lat0, lon0=lon0, radius=150)
    data = api_get(f"{API}?{urllib.parse.urlencode(qs)}")
    if not data:
        return []
    out = []
    for obs in data.get("results", []):
        geo = obs.get("geojson") or {}
        coords = geo.get("coordinates") or [None, None]
        on = obs.get("observed_on") or ""
        for ph in obs.get("photos") or []:
            url = (ph.get("url") or "").replace("/square.jpg", "/medium.jpg")
            if not url:
                continue
            out.append({"obs_id": obs.get("id"), "photo_id": ph.get("id"),
                        "url": url, "lat": coords[0], "lon": coords[1],
                        "date": on,
                        "license": (ph.get("license_code") or "").lower(),
                        "attr": (ph.get("attribution") or "").replace("\n", " ")[:180]})
    return out


def save_image(raw, path):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) > 256:
        s = 256 / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                         Image.BILINEAR)
    img.save(path, "JPEG", quality=90)


def disk_free_gb(path):
    try:
        return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize / 1073741824 \
            if os.name != "nt" else \
            (os.statvfs(path).f_bavail * os.statvfs(path).f_frsize / 1073741824)
    except Exception:
        return 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-species", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pages-per-species", type=int, default=8)
    ap.add_argument("--api-sleep", type=float, default=1.1)
    ap.add_argument("--max-obs-per-species", type=int, default=400,
                    help="cap distinct observations per species (leakage control)")
    ap.add_argument("--near", type=float, nargs=2, default=None,
                    metavar=("LAT", "LON"), help="restrict to a radius (km)")
    a = ap.parse_args()

    with open(a.species, encoding="utf-8-sig") as f:
        species = list(csv.DictReader(f))
    log(f"物种清单 {len(species)} 种 -> {a.species}")
    lat0, lon0 = (a.near if a.near else (None, None))

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(a.out, "dataset.csv")

    done_photos = set()
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_photos.add(r["photo_id"])
        log(f"断点续传: 已有 {len(done_photos)} 张")

    fh = open(meta_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not done_photos:
        w.writerow(["file", "scientific_name", "common_name", "birdnet_index",
                    "obs_id", "photo_id", "lat", "lon", "date", "license",
                    "attribution"])
        fh.flush()

    pool = ThreadPoolExecutor(max_workers=a.workers)
    t_start = time.time()
    total_ok = total_fail = 0
    lock = threading.Lock()

    def dl(job):
        sci, meta = job
        safe = f"{sci.replace(' ', '_')}_{meta['photo_id']}"
        path = os.path.join(img_dir, safe + ".jpg")
        if os.path.exists(path):
            return ("skip", safe, meta)
        for _ in range(3):
            try:
                req = urllib.request.Request(meta["url"],
                                             headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=90) as r:
                    raw = r.read()
                save_image(raw, path)
                return ("ok", safe, meta)
            except Exception:
                time.sleep(2)
        return ("fail", safe, meta)

    def record(status, safe, meta, sp, zh):
        nonlocal total_ok, total_fail
        with lock:
            if status == "fail":
                total_fail += 1
                return
            if status == "ok":
                total_ok += 1
            w.writerow([f"{safe}.jpg", meta["sci"], zh, sp.get("birdnet_index", ""),
                        meta["obs_id"], meta["photo_id"], meta["lat"],
                        meta["lon"], meta["date"], meta["license"], meta["attr"]])
            fh.flush()

    def consume(sp, photos, zh, i):
        """Download this species' photos through the pool and write rows."""
        obs_ids, extra = [], []
        for p in photos:
            if p["photo_id"] in done_photos:
                continue
            if p["obs_id"] in seen_obs:
                extra.append(p)
            else:
                seen_obs.add(p["obs_id"])
                obs_ids.append(p["obs_id"])
        random.Random(i).shuffle(obs_ids)
        obs_ids = obs_ids[: a.max_obs_per_species]
        keep = set(obs_ids)
        chosen = [p for p in photos if p["obs_id"] in keep and p["photo_id"] not in done_photos]
        # at most 2 photos per observation so no single bird fills a species
        per = {}
        picked = []
        for p in chosen:
            o = p["obs_id"]
            if per.get(o, 0) < 2:
                picked.append(p)
                per[o] = per.get(o, 0) + 1
            if len(picked) >= a.per_species:
                break

        jobs = []
        for p in picked:
            p["sci"] = sp["scientific_name"]
            jobs.append((sp["scientific_name"], p))
        for status, safe, meta in pool.map(dl, jobs):
            record(status, safe, meta, sp, zh)
        return len(obs_ids), len(picked)

    seen_obs = set()
    # Producer: API calls are the slow part, run them in a worker so image
    # downloads overlap with the next species' fetches.
    def produce():
        for i, sp in enumerate(species, 1):
            sci, zh = sp["scientific_name"], sp["common_name"]
            photos = []
            for page in range(a.pages_per_species):
                got = fetch_species_page(sci, page, lat0, lon0)
                if not got:
                    break
                photos += got
                time.sleep(a.api_sleep)
                if len(photos) >= a.per_species * 1.6:
                    break
            queue.put((i, sp, zh, photos))
            log(f"  api  [{i:>3}/{len(species)}] {sci:<24} 取到 {len(photos)} 张候选")
        queue.put(None)

    queue = Queue(maxsize=4)
    prod = threading.Thread(target=produce, daemon=True)
    prod.start()

    done_species = 0
    while True:
        item = queue.get()
        if item is None:
            break
        i, sp, zh, photos = item
        n_obs, n_pick = consume(sp, photos, zh, i)
        done_species += 1
        el = time.time() - t_start
        rate = total_ok / max(1e-9, el / 60)
        eta = (len(species) - done_species) * (el / max(1, done_species)) / 60
        log(f"[{i:>3}/{len(species)}] {sci if False else sp['scientific_name']:<24}"
            f"{zh:<9} obs={n_obs:>3} 用{n_pick:<4} 累计{total_ok:<6} "
            f"{rate:.0f}张/分 ETA {eta:.0f}分")

    prod.join()
    pool.shutdown()
    fh.close()
    log(f"完成: 新增 {total_ok} 张, 失败 {total_fail} 张 -> {meta_path}")
    n = sum(1 for _ in open(meta_path, encoding="utf-8")) - 1
    log(f"dataset.csv 共 {n} 行")


if __name__ == "__main__":
    main()
"""Build a regional bird dataset from GBIF (the route iNaturalist itself points to).

Why GBIF instead of the iNat API:
  * ~1.3 s for 200 records vs iNat's 3.6 s for 30 (and iNat throttles hard)
  * real geographic bounding box -> "birds actually recorded in North China"
  * license filter -> only CC0 / CC BY / CC BY-SA (commercially usable)
  * every record carries lat / lon / year / month / occurrence key, which is what
    we need for leakage-free splitting and for geomodel reranking

GBIF guidance is ~1 request/second; a process-wide pacer enforces that.

Crash safety (an unattended run must never lose work):
  * every finished species is cached to <out>/gbif_prefetch_cache.json (atomic
    replace, per-region) and <out>/gbif_availability.csv is rewritten, so a
    rerun resumes the probe pass instead of re-querying GBIF for hours
  * dataset.csv is flushed per row -> rerun skips already-downloaded keys
  * progress.json is a machine-readable heartbeat for monitoring

Usage:
  python build_dataset_gbif.py --species species_huabei_150.csv \
      --out <dir> --per-species 300 --bbox 32,42,110,120
"""
import argparse
import csv
import io
import json
import os
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from PIL import Image, ImageOps

# stdout 管道默认 GBK，种中文名里的「䴓」等扩展区字符会 UnicodeEncodeError 炸掉无人值守链
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SEARCH = "https://api.gbif.org/v1/occurrence/search"
MATCH = "https://api.gbif.org/v1/species/match"
SPECIES = "https://api.gbif.org/v1/species"
UA = "ObserveBird-PoC/0.1 (regional bird dataset; non-commercial research)"
_lock = threading.Lock()

# GBIF license values are enum *names*, not URLs or short codes. Membership is
# tested by exact match: "CC_BY_NC_4_0".startswith("CC_BY_") is True, so any
# startswith-based test silently counts NC/ND data as commercially usable.
COMMERCIAL_LICENSES = {
    "CC0_1_0",
    "CC_BY_4_0",
}  # 2026-10-07 实测：GBIF occurrence API 的 license 枚举只剩 5 个
# (CC0_1_0/CC_BY_4_0/CC_BY_NC_4_0/UNSPECIFIED/UNSUPPORTED)，历史版本枚举(如 CC_BY_SA_2_0)
# 全部 400 —— 原清单按旧文档写，--commercial-only 必炸
NONCOMMERCIAL_MARKS = ("-NC", "-ND")


def log(m):
    with _lock:
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class Pacer:
    """Process-wide minimum interval between outbound API calls.

    Sleeping *after* the response (the old behaviour) serialises latency behind
    the cooldown; this hands out timestamps up front so requests pipeline while
    still respecting the interval.
    """

    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_at)
            self.next_at = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


_PACER = Pacer(1.05)
# A GBIF occurrence page with limit=200 carries media metadata for every record
# and regularly takes 60-120 s from a CN client. A 45 s budget burns three
# retries on requests that were going to succeed, which is what made the probe
# pass crawl. Media downloads keep the short budget; API calls get their own.
API_TIMEOUT = 120
MEDIA_TIMEOUT = 45


def backoff(attempt):
    """1.5, 3, 6, 12, ... capped at 60 s, with jitter to de-sync workers."""
    return min(60.0, 1.5 * (2 ** attempt)) * (0.75 + 0.5 * random.random())


def get_json(url, tries=5, timeout=None):
    timeout = timeout or API_TIMEOUT
    for attempt in range(tries):
        try:
            _PACER.wait()
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 403):
                wait = 30 if e.code == 429 else 120
                log(f"    HTTP {e.code} -> 退避 {wait}s")
                time.sleep(wait)
                continue
            log(f"    HTTP {e.code}")
            time.sleep(backoff(attempt))
        except Exception as e:
            log(f"    失败 {type(e).__name__} (第{attempt + 1}次)")
            time.sleep(backoff(attempt))
    return None


_taxon_cache = {}


def taxon_key(sci):
    if sci in _taxon_cache:
        return _taxon_cache[sci]
    d = get_json(f"{MATCH}?name={urllib.parse.quote(sci)}&strict=false")
    k = d.get("usageKey") if d else None
    if k is None:
        d2 = get_json(f"{SEARCH}?scientificName={urllib.parse.quote(sci)}&limit=1")
        rows = (d2 or {}).get("results") or []
        k = rows[0].get("taxonKey") if rows else None
    if k is not None:
        # strict=false degrades to higher ranks: "Ardea coromanda" resolves to the
        # GENUS key, which would label every Ardea in the region with one wrong name.
        s = get_json(f"{SPECIES}/{k}") or {}
        if s.get("rank") != "SPECIES":
            log(f"    {sci}: GBIF 只给到 rank={s.get('rank')} "
                f"({s.get('scientificName')}), 判定为不可用")
            k = None
    _taxon_cache[sci] = k
    return k


def _region_qs(tkey, bbox, limit, offset=0):
    """Query string shared by search and facet calls so counts are comparable."""
    lat, lon = bbox
    qs = {"taxonKey": tkey, "country": "CN", "mediaType": "StillImage",
          "decimalLatitude": f"{lat[0]},{lat[1]}",
          "decimalLongitude": f"{lon[0]},{lon[1]}",
          "hasCoordinate": "true", "limit": limit, "offset": offset}
    return urllib.parse.urlencode(qs)


def search_page(tkey, offset, limit, bbox, license_filter=None):
    q = _region_qs(tkey, bbox, limit, offset)
    for lic in (license_filter or []):
        q += f"&license={lic}"
    d = get_json(f"{SEARCH}?{q}")
    return d or {"results": [], "count": 0}


def license_breakdown(tkey, bbox):
    """Records per license enum, over the *same* region filter as search_page.

    The old version omitted bbox/country/mediaType, so it reported global counts
    that could not be compared against the regional availability numbers.
    """
    q = _region_qs(tkey, bbox, 0) + "&facet=license&facetLimit=100"
    d = get_json(f"{SEARCH}?{q}")
    return {c["name"]: c["count"] for c in
            (((d or {}).get("facets") or [{}])[0]).get("counts") or []}


def license_stats(bk):
    """(total, commercially usable, NC-or-ND) for an exact-enum breakdown."""
    tot = sum(bk.values())
    ok = sum(v for k, v in bk.items() if k in COMMERCIAL_LICENSES)
    nc = sum(v for k, v in bk.items() if any(m in k for m in NONCOMMERCIAL_MARKS))
    return tot, ok, nc


def save_image(raw, path):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) > 256:
        s = 256 / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                         Image.BILINEAR)
    img.save(path, "JPEG", quality=90)


def lower_priority():
    """Yield CPU and IO to whatever else the machine is doing.

    PROCESS_MODE_BACKGROUND_BEGIN | BELOW_NORMAL_PRIORITY_CLASS
    """
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00010000 | 0x00004000)
        log("优先级 -> BelowNormal + 后台IO（让位给前台任务）")
    except Exception as e:
        log(f"优先级设置失败，忽略: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-species", type=int, default=300)
    ap.add_argument("--bbox", type=float, nargs=4,
                    default=[32.0, 42.0, 110.0, 120.0],
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--page", type=int, default=200, help="records per request")
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--api-interval", type=float, default=1.05,
                    help="process-wide minimum seconds between GBIF API calls")
    ap.add_argument("--api-timeout", type=int, default=120,
                    help="per-request timeout for API calls; limit=200 pages are slow")
    ap.add_argument("--media-timeout", type=int, default=45,
                    help="per-request timeout for image downloads")
    ap.add_argument("--probe-only", action="store_true",
                    help="only report how many usable records exist per species")
    ap.add_argument("--refresh-prefetch", action="store_true",
                    help="ignore the cached probe pass and re-query GBIF")
    ap.add_argument("--nice", action="store_true",
                    help="lower CPU/IO priority so foreground work stays fast")
    ap.add_argument("--commercial-only", action="store_true",
                    help="server-side filter to CC0/CC_BY/CC_BY_SA only. Off by "
                         "default: only ~4%% of iNat-sourced GBIF records are "
                         "commercially licensed, so this shrinks the pool by ~20x. "
                         "Every row records its license either way.")
    a = ap.parse_args()

    global _PACER
    _PACER = Pacer(a.api_interval)
    global API_TIMEOUT, MEDIA_TIMEOUT
    API_TIMEOUT = a.api_timeout
    MEDIA_TIMEOUT = a.media_timeout
    socket.setdefaulttimeout(max(API_TIMEOUT, MEDIA_TIMEOUT))

    lic_filter = sorted(COMMERCIAL_LICENSES) if a.commercial_only else None
    bbox = ((a.bbox[0], a.bbox[1]), (a.bbox[2], a.bbox[3]))
    with open(a.species, encoding="utf-8-sig") as f:
        species = list(csv.DictReader(f))

    os.makedirs(a.out, exist_ok=True)
    # Prefetch cache / availability live *inside* the region's out dir: the
    # availability counts are per-bbox facts, and two regions sharing a parent
    # dir (data/huabei, data/dongbei, ...) must not share one cache file.
    cache_path = os.path.join(a.out, "gbif_prefetch_cache.json")
    avail_path = os.path.join(a.out, "gbif_availability.csv")
    progress_path = os.path.join(a.out, "progress.json")

    log(f"物种 {len(species)} 种 | 地理框 lat{a.bbox[0]}~{a.bbox[1]} "
        f"lon{a.bbox[2]}~{a.bbox[3]} | license过滤={lic_filter or '关闭(记录但不过滤)'}")
    if a.nice:
        lower_priority()

    common = {x["scientific_name"]: x.get("common_name", "") for x in species}

    def flush_availability(availability, lic_stat):
        order = sorted(availability, key=lambda s: -availability[s])
        tmp = avail_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["scientific_name", "common_name", "gbif_records",
                        "license_breakdown"])
            for s in order:
                w.writerow([s, common.get(s, ""), availability[s],
                            json.dumps(lic_stat.get(s, {}))])
        os.replace(tmp, avail_path)

    def flush_progress(**kw):
        kw.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            write_json_atomic(progress_path, kw)
        except OSError:
            pass

    # ---- probe pass: availability + license breakdown per species ----
    cache = {}
    if os.path.exists(cache_path) and not a.refresh_prefetch:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            log(f"预取缓存命中 {len(cache)} 种 <- {cache_path}")
        except (OSError, ValueError) as e:
            log(f"预取缓存损坏，忽略: {e}")
            cache = {}
    # seed the taxon cache so produce() never re-queries species/match
    _taxon_cache.update({k: v["tk"] for k, v in cache.items() if v.get("tk")})

    availability, lic_stat = {}, {}
    fresh = 0
    for sp in species:
        sci = sp["scientific_name"]
        hit = cache.get(sci)
        if hit is not None and not a.refresh_prefetch:
            availability[sci] = int(hit.get("n", 0))
            if hit.get("bk"):
                lic_stat[sci] = hit["bk"]
            log(f"  [预取] {sci:<26} 缓存 taxonKey={hit.get('tk')} 可用 {availability[sci]}")
            continue
        tk = taxon_key(sci)
        if not tk:
            availability[sci] = 0
            cache[sci] = {"tk": None, "n": 0}
            log(f"  [预取] {sci:<26} taxonKey 未找到")
        else:
            d = search_page(tk, 0, 1, bbox, lic_filter)
            availability[sci] = int(d.get("count", 0))
            bk = license_breakdown(tk, bbox) if a.probe_only else {}
            if bk:
                lic_stat[sci] = bk
            cache[sci] = {"tk": tk, "n": availability[sci], "bk": bk}
            log(f"  [预取] {sci:<26} taxonKey={tk:<9} 可用 {availability[sci]}")
        fresh += 1
        write_json_atomic(cache_path, cache)      # atomic: crash-safe
        flush_availability(availability, lic_stat)

    flush_availability(availability, lic_stat)
    log(f"-> {avail_path}  (本次新查 {fresh} 种)")

    log(f"\n合计可用记录 {sum(availability.values())} | "
        f"有数据的种 {sum(1 for v in availability.values() if v)}/{len(species)}")
    for thr in (300, 150, 60, 20):
        log(f"  >= {thr:>3} 条的种: {sum(1 for v in availability.values() if v >= thr)}")
    if lic_stat:
        tot = ok = nc = 0
        for bk in lic_stat.values():
            t, o, n = license_stats(bk)
            tot += t
            ok += o
            nc += n
        log(f"  授权(区域口径): 可商用 {ok}/{tot} ({ok / tot * 100:.1f}%)"
            f" | NC/ND {nc} ({nc / tot * 100:.1f}%)")
    if a.probe_only:
        return

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    meta_path = os.path.join(a.out, "dataset.csv")

    done = set()
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r.get("gbif_key", ""))
        log(f"断点续传: 已有 {len(done)} 条")

    fh = open(meta_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not done:
        w.writerow(["file", "scientific_name", "common_name", "birdnet_index",
                    "gbif_key", "occurrence_key", "lat", "lon", "year", "month",
                    "license", "dataset_key", "attribution"])
        fh.flush()

    pool = ThreadPoolExecutor(max_workers=a.workers)
    t0 = time.time()
    stats = {"ok": 0, "fail": 0}

    def dl(job):
        sci, sp, rec, url = job
        safe = f"{sci.replace(' ', '_')}_{rec.get('key')}"
        path = os.path.join(img_dir, safe + ".jpg")
        if os.path.exists(path):
            return ("skip", safe, rec, sp, url)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=MEDIA_TIMEOUT) as r:
                    save_image(r.read(), path)
                return ("ok", safe, rec, sp, url)
            except Exception:
                time.sleep(backoff(attempt))
        return ("fail", safe, rec, sp, url)

    def record(status, safe, rec, sp, url):
        with _lock:
            if status == "fail":
                stats["fail"] += 1
                return
            if status == "ok":
                stats["ok"] += 1
            w.writerow([f"{safe}.jpg", sp["scientific_name"], sp["common_name"],
                        sp.get("birdnet_index", ""), safe.split("_")[-1],
                        rec.get("key"), rec.get("decimalLatitude"),
                        rec.get("decimalLongitude"), rec.get("year"),
                        rec.get("month"), rec.get("license", ""),
                        rec.get("datasetKey", ""),
                        (rec.get("occurrenceID") or "")[:150]])
            fh.flush()

    q = Queue(maxsize=3)

    def produce():
        # Pages of one species are independent. The pacer still spaces the
        # requests 1.05 s apart, so this overlaps GBIF's slow response time
        # without increasing the request rate.
        page_pool = ThreadPoolExecutor(max_workers=a.max_pages)
        for sp in species:
            sci = sp["scientific_name"]
            tk = _taxon_cache.get(sci) or taxon_key(sci)
            if not tk:
                continue
            tp = time.time()
            pages = list(page_pool.map(
                lambda p: search_page(tk, p * a.page, a.page, bbox, lic_filter),
                range(a.max_pages)))
            recs = []
            seen = set()
            for d in pages:
                for rec in d.get("results", []):
                    if not rec.get("media"):
                        continue
                    m = rec["media"][0]
                    url = (m.get("identifier") or "").replace(
                        "/original.jpg", "/medium.jpg")
                    if not url:
                        continue
                    key = rec.get("key")
                    if key in seen or str(key) in done:
                        continue
                    seen.add(key)
                    recs.append((rec, url))
                    if len(recs) >= a.per_species * 1.5:
                        break
                if len(recs) >= a.per_species * 1.5:
                    break
            q.put((sp, recs))
            log(f"  api  {sci:<26} 候选 {len(recs):<4} "
                f"(GBIF 共 {availability.get(sci, '?')}, {time.time() - tp:.0f}s)")
        page_pool.shutdown()
        q.put(None)

    threading.Thread(target=produce, daemon=True).start()

    done_sp = 0
    while True:
        item = q.get()
        if item is None:
            break
        sp, recs = item
        # max 2 photos per occurrence keeps any single bird from dominating and
        # makes an obs-grouped split meaningful
        per, picked = {}, []
        for rec, url in recs:
            k = rec.get("key")
            if per.get(k, 0) < 2:
                picked.append((rec, url))
                per[k] = per.get(k, 0) + 1
            if len(picked) >= a.per_species:
                break
        random.Random(len(recs)).shuffle(picked)
        picked = picked[: a.per_species]
        jobs = [(sp["scientific_name"], sp, r, u) for r, u in picked]
        for status, safe, rec, s, u in pool.map(dl, jobs):
            record(status, safe, rec, s, u)
        done_sp += 1
        el = time.time() - t0
        eta = (len(species) - done_sp) * el / max(1, done_sp) / 60
        log(f"[{done_sp:>3}/{len(species)}] {sp['scientific_name']:<26}"
            f"{sp['common_name']:<9} 用{len(jobs):<4} 累计{stats['ok']:<6} "
            f"ETA {eta:.0f}分")
        flush_progress(stage="download", species_done=done_sp,
                       species_total=len(species), ok=stats["ok"],
                       fail=stats["fail"], eta_min=round(eta, 1),
                       elapsed_min=round(el / 60, 1))

    pool.shutdown()
    fh.close()
    log(f"完成 新增 {stats['ok']} 张 失败 {stats['fail']} -> {meta_path}")
    flush_progress(stage="done", species_done=done_sp, species_total=len(species),
                   ok=stats["ok"], fail=stats["fail"])


def install_signal_handlers():
    import signal

    def handler(signum, _frame):
        log(f"收到信号 {signum}，退出（已下载数据保留，重跑即续传）")
        raise SystemExit(0)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        s = getattr(signal, name, None)
        if s is None:
            continue
        try:
            signal.signal(s, handler)
        except (OSError, ValueError):
            pass


if __name__ == "__main__":
    import traceback

    install_signal_handlers()

    # Unattended runs must never die silently: catch everything, log it, and
    # leave the partial dataset.csv in place so a rerun resumes.
    try:
        main()
    except KeyboardInterrupt:
        log("收到中断，保留已下载数据以便续传")
    except BaseException:
        log("!! 未捕获异常，退出（dataset.csv 已保存，可续传）")
        traceback.print_exc()
        raise
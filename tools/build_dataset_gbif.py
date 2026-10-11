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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from PIL import Image, ImageOps

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
    "CC_BY_1_0", "CC_BY_2_0", "CC_BY_2_5", "CC_BY_3_0", "CC_BY_4_0",
    "CC_BY_SA_1_0", "CC_BY_SA_2_0", "CC_BY_SA_2_5", "CC_BY_SA_3_0",
    "CC_BY_SA_4_0",
}
NONCOMMERCIAL_MARKS = ("-NC", "-ND")


def log(m):
    with _lock:
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---- progress.json = 给监控台的机器可读状态（也是唯一的完成判据）----
# 历史上它只在成功路径写 stage="done"，异常/中断什么都不写，于是外层只能靠
# `xxx.exit` 标记去猜子进程有没有结束 —— 而那个标记由三个地方写、内容还不可信
# （空文件 / "ECHO is off." / 15 / -1）。现在把它做成正式终态文件：
# 中断和失败路径同样落盘，并带 finished_at / pid / exit_code。
_PROGRESS = {}
_PROGRESS_PATH = None


def flush_progress(**kw):
    """把状态并入 progress.json（原子替换）。下载中每个物种调一次，容错优先。"""
    if not _PROGRESS_PATH:
        return
    _PROGRESS.update(kw)
    _PROGRESS["pid"] = os.getpid()
    _PROGRESS["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        write_json_atomic(_PROGRESS_PATH, _PROGRESS)
    except OSError:
        pass


def finalize_progress(stage: str, exit_code: int, **kw):
    """写终态：done / interrupted / failed。下游只认 stage 判断"结束没结束"。"""
    if not _PROGRESS_PATH:
        return
    _PROGRESS.update(kw)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    _PROGRESS.update({"stage": stage, "exit_code": exit_code, "pid": os.getpid(),
                      "finished_at": now, "ts": now})
    try:
        write_json_atomic(_PROGRESS_PATH, _PROGRESS)
    except OSError:
        pass


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
_NAME_MAP = None


def name_map():
    """名字对照表（tools/dataset_traits/species_name_map.json）。

    为什么要它：三套名录的名字口径互不相同 —— geomodel 和我们自己的名单用**新名**
    （Thinornis dubius），而 GBIF 用**旧名**（Charadrius dubius）。这张表已把
    1494 个名字对好了 GBIF key，先用它可避开两条老路的坑：
      * `species/match?strict=false` 静默降级到属 → 该种被弃用 → 0 张图
      * 退路 `species/search?scientificName=` 查不到（金翅雀就是这么丢了
        key=6101015 上的 35,263 条中国记录）
    """
    global _NAME_MAP
    if _NAME_MAP is None:
        _NAME_MAP = {}
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "dataset_traits", "species_name_map.json")
        if os.path.exists(p):
            try:
                _NAME_MAP = json.load(open(p, encoding="utf-8"))
                log(f"名字对照表: 载入 {len(_NAME_MAP)} 条")
            except Exception as e:
                log(f"名字对照表载入失败({type(e).__name__})，回退到按名匹配")
                _NAME_MAP = {}
    return _NAME_MAP


def taxon_key(sci):
    if sci in _taxon_cache:
        return _taxon_cache[sci]
    # ---- ① 名字对照表优先（已经对好的，直接给 key）----
    ent = name_map().get(sci)
    if ent:
        if ent.get("excluded"):
            log(f"    {sci}: 名字对照表标记为「排除」—— {(ent.get('why') or '')[:90]}")
            _taxon_cache[sci] = None
            return None
        k = ent.get("gbif_key")
        if k:
            _taxon_cache[sci] = int(k)
            return int(k)
    # ---- ② 回退：按名匹配 ----
    d = get_json(f"{MATCH}?name={urllib.parse.quote(sci)}&strict=false") or {}
    mt, k = d.get("matchType"), d.get("usageKey")
    if mt == "FUZZY":
        # FUZZY 可能指向一个**名字不同**的分类单元（实测 Ardenna pacificus →
        # 一个名字不同的 key）。直接采用等于悄悄取错数据，宁可报出来。
        canon = (d.get("canonicalName") or "").strip().lower()
        if canon != sci.strip().lower():
            log(f"    {sci}: GBIF 只给到 FUZZY「{d.get('canonicalName')}」，"
                f"与查询名不同，判定为不可用")
            k = None
    if k is None and mt not in ("EXACT", "FUZZY"):
        # 退路：用 q= 全文搜 + **精确同名比对**。
        #（原来用 scientificName= 且取第一条，既查不到也不验证）
        d2 = get_json(f"{SEARCH}?q={urllib.parse.quote(sci)}"
                      f"&rank=SPECIES&highertaxonKey=212&limit=5") or {}
        for x in d2.get("results") or []:
            if (x.get("canonicalName") or "").strip().lower() == sci.strip().lower():
                k = x.get("key") or x.get("nubKey")
                break
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

    # 模块级 flush_progress/finalize_progress 需要知道往哪写（异常路径在 main 之外，
    # 拿不到这个局部变量）。调用点继续写 flush_progress(...)，即用模块级实现。
    global _PROGRESS_PATH
    _PROGRESS_PATH = progress_path

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
    # ---- 缓存 × 名字对照表 一致性守卫（2026-10-06）----
    # 缓存里的 tk 是当年按名单名直接问 GBIF 得到的；对照表（人工审核）是后来才建的。
    # GBIF 对不认识的新名会静默降到属（"Ardea coromanda"→Ardea 属 2480922），这种
    # 属级 tk 留在缓存里，命中分支会拿它直接再下载 → 重审计读到属级 → 重清洗再把行
    # 删掉 —— 实测华北 3 个种就是这样，恢复会白跑一场。两者冲突时以对照表为准。
    nm = name_map()
    stale = 0
    for sci in list(cache):
        tk = cache[sci].get("tk")
        if not tk:
            continue          # tk=null 走下面的自愈路径（每轮重试）
        ent = nm.get(sci) or {}
        k = ent.get("gbif_key")
        bad, why = False, ""
        if ent.get("excluded"):
            bad, why = True, "对照表标记排除"
        elif k:
            try:
                bad = int(tk) != int(k)
                why = f"缓存 tk={tk} ≠ 对照表 key={k}"
            except (TypeError, ValueError):
                bad, why = True, f"缓存 tk={tk!r} 不是数字"
        if bad:
            del cache[sci]
            stale += 1
            log(f"  [预取] {sci:<26} 弃用旧缓存（{why}）→ 按对照表重查")
    if stale:
        log(f"  [预取] ⚠️ 共弃用 {stale} 条与对照表冲突的缓存（属级 key 时代遗留）")

    # seed the taxon cache so produce() never re-queries species/match
    _taxon_cache.update({k: v["tk"] for k, v in cache.items() if v.get("tk")})

    availability, lic_stat = {}, {}
    fresh = 0
    for sp in species:
        sci = sp["scientific_name"]
        hit = cache.get(sci)
        # 注意 `hit.get("tk")` 这一条：缓存里 tk=null 表示**上次没找到**，
        # 那不是"事实"，是"当年的失败"。若不重试，今天为这 6 个种做的名字映射
        # 永远进不来 —— 实测华东 resume 时它们全是 {"tk": null, "n": 0}，
        # 会被无声跳过。加上这一条后缓存能自愈：null 的种每轮重试一次。
        if hit is not None and hit.get("tk") and not a.refresh_prefetch:
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
    have = {}                      # 每个种在 dataset.csv 里已有多少张（用于"已够就跳过"）
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r.get("gbif_key", ""))
                s = (r.get("scientific_name") or "").strip()
                if s:
                    have[s] = have.get(s, 0) + 1
        log(f"断点续传: 已有 {len(done)} 条，涉及 {len(have)} 种")

    fh = open(meta_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not done:
        w.writerow(["file", "scientific_name", "common_name", "birdnet_index",
                    "gbif_key", "occurrence_key", "lat", "lon", "year", "month",
                    "license", "dataset_key", "attribution"])
        fh.flush()

    pool = ThreadPoolExecutor(max_workers=a.workers)
    t0 = time.time()
    # rows = 本轮**实际写入 dataset.csv 的行数**（含"补登记"，见 record()），
    # 张数口径的 ETA 必须用它，不能用 ok（ok 只数"新下的图"）。
    stats = {"ok": 0, "fail": 0, "skip": 0, "rows": 0}

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
            elif status == "skip":
                # 图片已在盘上、只是 dataset.csv 里还没有这一行 → 补登记。
                # **不是重复行**（file 名由 occurrence key 决定，实测 0 个重复），
                # 常见于"上一轮被杀在写行之前"留下的孤儿文件。
                # 但它照样让 rows 增长，所以必须单独计数：
                # 只认 stats["ok"] 会让"本轮下了多少"显示成 0、张数口径的 ETA 永远算不出来。
                stats["skip"] += 1
            stats["rows"] += 1
            w.writerow([f"{safe}.jpg", sp["scientific_name"], sp["common_name"],
                        sp.get("birdnet_index", ""), safe.split("_")[-1],
                        rec.get("key"), rec.get("decimalLatitude"),
                        rec.get("decimalLongitude"), rec.get("year"),
                        rec.get("month"), rec.get("license", ""),
                        rec.get("datasetKey", ""),
                        (rec.get("occurrenceID") or "")[:150]])
            fh.flush()

    q = Queue(maxsize=3)
    n_skip = 0      # "已够目标张数"被跳过的种数（ETA 回退时要排除它们）
    n_nokey = 0     # 解析不出 taxonKey 的种数（同样不产生图片）

    def produce():
        nonlocal n_skip, n_nokey
        # Pages of one species are independent. The pacer still spaces the
        # requests 1.05 s apart, so this overlaps GBIF's slow response time
        # without increasing the request rate.
        page_pool = ThreadPoolExecutor(max_workers=a.max_pages)
        for sp in species:
            sci = sp["scientific_name"]
            # —— 已够就跳过，且只补差额 ——
            # `--per-species` 的含义是「该种在数据集里的目标张数」，**不是**「本轮再拿多少」。
            # 没有这个检查，resume 会把已满 300 的种再下一遍：实测华东 57 个已完成种里
            # 44 个正好 300 张 → 会被推到 600，白跑 13,200 张的功夫（正好等于全区缺口），
            # 而真正缺的 93 个种排在最后。这个检查也顺带让"同名重复下载"消失。
            #
            # 第二条：**GBIF 在该 bbox 的可用量已被取尽时也跳过**。
            # 判据 `have >= min(per_species, availability)`。达不到这个条件时再查一遍
            # 也只会拿回同一批 occurrence（全被 `done` 过滤掉），纯粹白花几十秒/种。
            # 实测：把主开发机已下的数据装进外包包后，133 个已取尽的种会各白等 ~40 秒
            # ≈ 90 分钟。华东那种"可用 500 但只下了 26"的种**不受影响**（仍会去补）。
            target = a.per_species
            _av = availability.get(sci)
            if isinstance(_av, int):
                target = min(target, _av)
            quota = target - have.get(sci, 0)
            if quota <= 0:
                n_skip += 1
                q.put((sp, None))          # None = 已够/已取尽，跳过
                continue
            tk = _taxon_cache.get(sci) or taxon_key(sci)
            if not tk:
                n_nokey += 1
                q.put((sp, []))            # 仍然占一个计数位，避免 [N/150] 编号漂移
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
                    if len(recs) >= quota * 1.5:
                        break
                if len(recs) >= quota * 1.5:
                    break
            q.put((sp, recs))
            log(f"  api  {sci:<26} 候选 {len(recs):<4} 配额 {quota:<4} "
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
        if recs is None:                       # 已够目标张数 或 GBIF 已取尽（produce 里判定）
            done_sp += 1
            n_have = have.get(sp["scientific_name"], 0)
            _av = availability.get(sp["scientific_name"])
            tgt = min(a.per_species, _av) if isinstance(_av, int) else a.per_species
            why = (f"已达 {a.per_species}/种" if tgt == a.per_species
                   else f"GBIF 已取尽（框内可用 {_av}）")
            log(f"[{done_sp:>3}/{len(species)}] {sp['scientific_name']:<26} "
                f"已有 {n_have} 张 ≥ 目标 {tgt}（{why}），跳过")
            continue
        # 只补差额：picked 上限是"目标张数 - 已有"，不是 per_species
        quota = max(0, a.per_species - have.get(sp["scientific_name"], 0))
        # max 2 photos per occurrence keeps any single bird from dominating and
        # makes an obs-grouped split meaningful
        per, picked = {}, []
        for rec, url in recs:
            k = rec.get("key")
            if per.get(k, 0) < 2:
                picked.append((rec, url))
                per[k] = per.get(k, 0) + 1
            if len(picked) >= quota:
                break
        random.Random(len(recs)).shuffle(picked)
        picked = picked[:quota]
        jobs = [(sp["scientific_name"], sp, r, u) for r, u in picked]
        for status, safe, rec, s, u in pool.map(dl, jobs):
            record(status, safe, rec, s, u)
        done_sp += 1
        el = time.time() - t0
        # ---- ETA 用**张数**口径，不用种数 ----
        # 老公式 `(剩余种数) × 总耗时/已处理种数` 假设"每个种都要花时间下图"。
        # 但两种情况会让它错得离谱：
        #   ① "已够目标张数"的种被瞬间跳过（实测 63 个"已处理"里 45 个是跳过）
        #   ② 可用量低于目标的薄种候选 0，也几乎不耗时
        # 结果：把平均值拉垮 → 算出 25 分钟，而实测约 4 小时（差 10 倍）。
        # 改成 张/分 口径：剩余张数 ÷ 本轮实际速率。
        el_min = el / 60.0
        rows_done = sum(have.values()) + stats["rows"]
        rows_goal = sum(min(a.per_species, availability.get(x["scientific_name"], 0))
                        for x in species)
        rate = stats["rows"] / el_min if el_min > 0.3 else 0.0
        if rate >= 1.0:
            eta = max(0.0, (rows_goal - rows_done) / rate)
            eta_txt, eta_val = f"ETA {eta:.0f}分", round(eta, 1)
        else:
            # 本轮还没下到任何图 → **没有速率可用，就不猜**（留空让监控台用它自己的
            # `(target-rows)/rate_now` 张数口径）。曾经在这里退回"种数口径"，结果
            # 除以的是"处理过但没产出图片"的种（跳过 / 候选 0），又算出 80 分钟这种假数。
            eta_txt, eta_val = "ETA --", None
        log(f"[{done_sp:>3}/{len(species)}] {sp['scientific_name']:<26}"
            f"{sp['common_name']:<9} 用{len(jobs):<4} "
            f"累计{stats['rows']:<6} "
            f"{eta_txt}  补登{stats['skip']:<5} "
            f"({rows_done:,}/{rows_goal:,} 张)")
        flush_progress(stage="download", species_done=done_sp,
                       species_total=len(species), ok=stats["ok"],
                       fail=stats["fail"], eta_min=eta_val,
                       rows=rows_done, rows_goal=rows_goal,
                       elapsed_min=round(el / 60, 1))

    pool.shutdown()
    fh.close()
    log(f"完成 新增 {stats['ok']} 张 失败 {stats['fail']} -> {meta_path}")
    # 终态：只有走到这里才写 done。下游（监控台/队列脚本）只认这个字段，
    # 不再需要 xxx.exit 标记。
    finalize_progress("done", 0, species_done=done_sp, species_total=len(species),
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
    # 每条退出路径都写 progress.json 的终态 —— 这是"结束没结束"的唯一可信来源。
    try:
        main()
    except KeyboardInterrupt:
        log("收到中断，保留已下载数据以便续传")
        finalize_progress("interrupted", 130)
    except SystemExit as e:            # 信号处理器 raise SystemExit(0)；argparse 也走这里
        code = e.code if isinstance(e.code, int) else 0
        log(f"收到信号/主动退出（code={code}），保留已下载数据以便续传")
        finalize_progress("interrupted", code)
        raise
    except BaseException as e:
        log(f"!! 未捕获异常（{type(e).__name__}），退出（dataset.csv 已保存，可续传）")
        finalize_progress("failed", 1, error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    else:
        # main() 正常返回但没走到 finalize（例如提前 return）时的兜底
        if _PROGRESS.get("stage") != "done":
            finalize_progress("done", 0)
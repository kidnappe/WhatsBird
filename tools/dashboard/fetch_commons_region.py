r"""Fetch CC-licensed species photos from Wikimedia Commons.

Commons is unreachable from CN networks directly, so requests go through a
proxy — scoped STRICTLY to this process's own opener instance: no environment
variables, no global install_opener, no system proxy settings touched.

Writes dataset_commons.csv (same schema as build_dataset_gbif.py) into the
region's data dir; images land in the same images/ directory with a Commons_
filename prefix so they never collide. Rows carry occurrence_key =
commons-<pageid> and per-file license from extmetadata. Merge into the main
dataset.csv afterwards with --merge-into (dedupe by occurrence_key).

Usage:
  python tools/fetch_commons_region.py --species species_huazhong_150.csv \
      --out "E:\code\code tools\data\huazhong"
  python tools/fetch_commons_region.py ... --merge-into "...\dataset.csv"
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from build_dataset_gbif import lower_priority, save_image, write_json_atomic

API = "https://commons.wikimedia.org/w/api.php"
UA = "ObserveBird-RegionalFetch/0.1 (regional bird dataset)"
TAG_RE = re.compile(r"<[^>]+>")
ALLOWED = ("CC0", "CC BY", "CC BY-SA")  # LicenseShortName 前缀匹配


def make_opener(proxy):
    """只给本进程 Commons 请求用的 opener；proxy=None 时强制直连（忽略系统代理）。"""
    handlers = []
    handlers.append(urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else {}))
    return urllib.request.build_opener(*handlers)


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--out", required=True, help="区域数据目录")
    ap.add_argument("--proxy", default="http://127.0.0.1:7877",
                    help="仅本进程使用的代理；传 none 直连（需自身可达）")
    ap.add_argument("--per-species", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--nice", action="store_true")
    ap.add_argument("--merge-into", default=None,
                    help="抓完后把 dataset_commons.csv 合并进该 dataset.csv")
    a = ap.parse_args()
    if a.nice:
        lower_priority()
    proxy = None if a.proxy.lower() == "none" else a.proxy
    opener = make_opener(proxy)

    def get_json(url):
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with opener.open(req, timeout=45) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as e:
                log(f"    API 失败 {type(e).__name__}: {e} (第{attempt + 1}次)")
                time.sleep(3 * (attempt + 1))
        return None

    species = list(csv.DictReader(open(a.species, encoding="utf-8-sig")))
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(a.out, "dataset_commons.csv")

    done_pages = set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            done_pages.add(r.get("occurrence_key", ""))
        log(f"续传: dataset_commons.csv 已有 {len(done_pages)} 行")
    write_json_atomic(os.path.join(a.out, "fetch_target_commons.json"),
                      {"total": len(done_pages), "ts": time.strftime("%H:%M:%S")})

    fh = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if fh.tell() == 0:
        w.writerow(["file", "scientific_name", "common_name", "birdnet_index",
                    "gbif_key", "occurrence_key", "lat", "lon", "year", "month",
                    "license", "dataset_key", "attribution"])
        fh.flush()

    stats = {"ok": 0, "fail": 0, "skip_lic": 0, "no_cat": 0}

    def dl(job):
        sci, zh, idx, pageid, title, url, lic, artist = job
        safe = f"Commons_{pageid}.jpg"
        path = os.path.join(img_dir, safe)
        if os.path.exists(path):
            return ("skip", job)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with opener.open(req, timeout=60) as r:
                    save_image(r.read(), path)
                return ("ok", job)
            except Exception:
                time.sleep(2 * (attempt + 1))
        return ("fail", job)

    pool = ThreadPoolExecutor(max_workers=a.workers)
    t0 = time.time()
    done_sp, todo_total = 0, len(species)
    for sp in species:
        sci, zh = sp["scientific_name"], sp.get("common_name", "")
        idx = sp.get("birdnet_index", "")
        cat = f"Category:{sci}"
        got = 0
        cont = None
        while got < a.per_species:
            q = urllib.parse.urlencode({
                "action": "query", "format": "json", "generator": "categorymembers",
                "gcmtitle": cat, "gcmtype": "file", "gcmlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 1024,
                "iiextmetadatafilter": "LicenseShortName|Artist|DateTimeOriginal"})
            if cont:
                q += "&gcmcontinue=" + urllib.parse.quote(cont)
            d = get_json(f"{API}?{q}")
            if d is None:
                break
            cont = (d.get("continue") or {}).get("gcmcontinue")
            pages = list((d.get("query") or {}).get("pages", {}).values())
            if not pages and got == 0 and cont is None:
                stats["no_cat"] += 1
                log(f"  {sci:<26} Commons 无分类，跳过")
                break
            jobs = []
            for p in pages:
                if got >= a.per_species:
                    break
                pageid = p.get("pageid")
                if f"commons-{pageid}" in done_pages:
                    continue
                ii = (p.get("imageinfo") or [{}])[0]
                em = ii.get("extmetadata", {}) or {}
                lic = (em.get("LicenseShortName") or {}).get("value", "")
                if not lic.startswith(ALLOWED):
                    stats["skip_lic"] += 1
                    continue
                url = ii.get("thumburl") or ii.get("url")
                if not url:
                    continue
                artist = TAG_RE.sub("", (em.get("Artist") or {}).get("value", ""))[:150]
                jobs.append((sci, zh, idx, pageid, p.get("title", ""), url,
                             lic, artist))
            for status, job in pool.map(dl, jobs):
                sci_, zh_, idx_, pageid, title, url, lic, artist = job
                if status == "fail":
                    stats["fail"] += 1
                    continue
                if status == "ok":
                    stats["ok"] += 1
                    got += 1
                done_pages.add(f"commons-{pageid}")
                w.writerow([f"Commons_{pageid}.jpg", sci_, zh_, idx_, "",
                            f"commons-{pageid}", "", "", "", "", lic,
                            "wikimedia_commons",
                            f"{title} / {artist}".strip(" /")[:200]])
                fh.flush()
            if not cont:
                break
            time.sleep(a.sleep)
        done_sp += 1
        el = time.time() - t0
        eta = (todo_total - done_sp) * el / max(1, done_sp) / 60
        if done_sp % 10 == 0 or done_sp == todo_total:
            log(f"[{done_sp:>3}/{todo_total}] 累计 {stats['ok']:<5} "
                f"(license跳过 {stats['skip_lic']}, 无分类 {stats['no_cat']}) "
                f"ETA {eta:.0f}分")
            write_json_atomic(os.path.join(a.out, "fetch_target_commons.json"),
                              {"total": len(done_pages), "ts": time.strftime("%H:%M:%S")})
    pool.shutdown()
    fh.close()
    write_json_atomic(os.path.join(a.out, "fetch_target_commons.json"),
                      {"total": len(done_pages), "ts": time.strftime("%H:%M:%S")})
    log(f"完成 新增 {stats['ok']} 失败 {stats['fail']} -> {csv_path}")

    if a.merge_into:
        merged = 0
        have = set()
        if os.path.exists(a.merge_into):
            have = {r.get("occurrence_key", "")
                    for r in csv.DictReader(open(a.merge_into, encoding="utf-8"))}
        dst = open(a.merge_into, "a", newline="", encoding="utf-8")
        dw = csv.writer(dst)
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            if r["occurrence_key"] in have:
                continue
            dw.writerow([r[c] for c in ["file", "scientific_name", "common_name",
                                        "birdnet_index", "gbif_key", "occurrence_key",
                                        "lat", "lon", "year", "month", "license",
                                        "dataset_key", "attribution"]])
            merged += 1
        dst.flush()
        dst.close()
        log(f"合并 {merged} 行 -> {a.merge_into}")


if __name__ == "__main__":
    main()

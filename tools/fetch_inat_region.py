r"""Fetch research-grade iNat photos for a regional species list, directly from
the iNat API.

Why direct instead of the GBIF mirror: GBIF only carries open-licensed iNat
photos and lags behind, which costs ~50% of the available pool (huazhong bbox
measured 9,251 via GBIF vs 17,897 via iNat for the same 145 species). Rows are
appended to the region's dataset.csv with the SAME schema as
build_dataset_gbif.py, so train.py / clean_dataset.py / the dashboard consume
them unchanged. Dedupe uses the iNat observation id: existing GBIF rows embed
it in their attribution URL.

Dashboard integration: progress.json + gbif_build.log.*-style lines are written
into the data dir (the download tab parses those), and fetch_target_inat.json
declares the expected row increment for the target counter.

Usage:
  python tools/fetch_inat_region.py --species species_huazhong_150.csv \
      --out "E:\code\code tools\data\huazhong" --bbox 31 37 105 117 --nice
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

API = "https://api.inaturalist.org/v1/observations"
UA = "ObserveBird-RegionalFetch/0.1 (regional bird dataset)"
OBS_ID_RE = re.compile(r"inaturalist\.org/observations/(\d+)")


def api_json(url, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            log(f"    API 失败 {type(e).__name__} (第{attempt + 1}次)")
            time.sleep(2 * (attempt + 1))
    return None


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
    ap.add_argument("--out", required=True, help="区域数据目录（含 dataset.csv / images）")
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("LATMIN", "LATMAX", "LONMIN", "LONMAX"))
    ap.add_argument("--per-species", type=int, default=300)
    ap.add_argument("--max-pages", type=int, default=4, help="每物种最多翻多少页（200 条/页）")
    ap.add_argument("--photo-per-obs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--api-interval", type=float, default=0.4)
    ap.add_argument("--nice", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    a = ap.parse_args()
    if a.nice:
        lower_priority()

    last_call = [0.0]

    def paced():
        wait = a.api_interval - (time.time() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.time()

    species = list(csv.DictReader(open(a.species, encoding="utf-8-sig")))
    ds_path = os.path.join(a.out, "dataset.csv")
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 现状：每物种已有行数 + 已抓过的 iNat observation id（GBIF 行写在 attribution 里）
    have, done_obs, done_by_species = {}, set(), {}
    if os.path.exists(ds_path):
        for r in csv.DictReader(open(ds_path, encoding="utf-8")):
            sci = r["scientific_name"]
            have[sci] = have.get(sci, 0) + 1
            m = OBS_ID_RE.search(r.get("attribution") or "")
            if m:
                done_obs.add(m.group(1))
                done_by_species[sci] = done_by_species.get(sci, 0) + 1
        log(f"现有 dataset.csv {sum(have.values())} 行 / {len(have)} 种；"
            f"已含 iNat 观测 {len(done_obs)} 条")

    bbox_q = (f"&swlat={a.bbox[0]}&swlng={a.bbox[2]}&nelat={a.bbox[1]}&nelng={a.bbox[3]}")

    # ---- 探测 pass：每物种框内研究级总量 → 增量目标 ----
    targets, total_new = {}, 0
    for i, sp in enumerate(species, 1):
        sci = sp["scientific_name"]
        q = urllib.parse.urlencode({"taxon_name": sci, "photos": "true", "per_page": 0,
                                    "quality_grade": "research"})
        paced()
        d = api_json(f"{API}?{q}{bbox_q}")
        total = int((d or {}).get("total_results", 0))
        room = max(0, a.per_species - have.get(sci, 0))
        # 扣除已被此前各轮用掉的观测：榨干的种目标归零，直接跳过不白翻页
        unused = max(0, total - done_by_species.get(sci, 0))
        targets[sci] = max(0, min(room, unused))
        total_new += targets[sci]
        log(f"  [预取] {sci:<26} iNat 框内 {total:<5} 已有 {have.get(sci, 0):<4}"
            f" 已用观测 {done_by_species.get(sci, 0):<4} 目标 +{targets[sci]}")
        if i % 25 == 0:
            write_json_atomic(os.path.join(a.out, "fetch_target_inat.json"),
                              {"total": total_new, "baseline_rows": sum(have.values()),
                               "ts": time.strftime("%H:%M:%S")})
    write_json_atomic(os.path.join(a.out, "fetch_target_inat.json"),
                      {"total": total_new, "baseline_rows": sum(have.values()),
                       "ts": time.strftime("%H:%M:%S")})
    log(f"增量目标合计 +{total_new} 张（轮起点 {sum(have.values())} 行）"
        f"-> fetch_target_inat.json")
    if a.probe_only:
        return

    fh = open(ds_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    stats = {"ok": 0, "fail": 0}

    def dl(job):
        sci, zh, oid, ph, lat, lon, yy, mm = job
        url = (ph.get("url") or "").replace("/square.", "/medium.")
        pid = ph.get("id")
        safe = f"{sci.replace(' ', '_')}_{oid}_{pid}.jpg"
        path = os.path.join(img_dir, safe)
        if os.path.exists(path):
            return ("skip", job)
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=20) as r:
                    save_image(r.read(), path)
                return ("ok", job)
            except Exception:
                time.sleep(2 * (attempt + 1))
        return ("fail", job)

    pool = ThreadPoolExecutor(max_workers=a.workers)
    t0 = time.time()
    done_sp = 0
    todo = [sp for sp in species if targets.get(sp["scientific_name"], 0) > 0]
    for sp in todo:
        sci, zh = sp["scientific_name"], sp.get("common_name", "")
        got, page, obs_seen = 0, 1, set()
        while (got < targets[sci] and page <= a.max_pages):
            q = urllib.parse.urlencode({"taxon_name": sci, "photos": "true",
                                        "per_page": 200, "page": page,
                                        "quality_grade": "research", "order_by": "votes"})
            paced()
            d = api_json(f"{API}?{q}{bbox_q}")
            results = (d or {}).get("results", [])
            if not results:
                break
            jobs = []
            for obs in results:
                oid = str(obs.get("id"))
                if oid in done_obs or oid in obs_seen:
                    continue
                obs_seen.add(oid)
                coords = (obs.get("geojson") or {}).get("coordinates") or [None, None]
                date = (obs.get("observed_on_details") or {}).get("date") or ""
                photos = (obs.get("photos") or [])[: a.photo_per_obs]
                for ph in photos:
                    if len(jobs) >= targets[sci] - got:
                        break
                    jobs.append((sci, zh, oid, ph, coords[1], coords[0],
                                 date[:4], date[5:7]))
                if len(jobs) >= targets[sci] - got:
                    break
            for status, job in pool.map(dl, jobs):
                sci_, zh_, oid, ph, lat, lon, yy, mm = job
                if status == "fail":
                    stats["fail"] += 1
                    continue
                # ok = 新下载；skip = 文件已在盘上但行未写入（上次中断的批次）
                stats["ok"] += 1
                got += 1
                done_obs.add(oid)
                w.writerow([f"{sci_.replace(' ', '_')}_{oid}_{ph.get('id')}.jpg",
                            sci_, zh_, sp.get("birdnet_index", ""),
                            ph.get("id"), oid, lat, lon, yy, mm,
                            (ph.get("license_code") or "").upper(),
                            "inat_api",
                            f"https://www.inaturalist.org/observations/{oid}"])
                fh.flush()
            page += 1
        done_sp += 1
        el = time.time() - t0
        eta = (len(todo) - done_sp) * el / max(1, done_sp) / 60
        # 与 build_dataset_gbif 的日志行同构，监控台下载页可直接解析
        log(f"[{done_sp:>3}/{len(todo)}] {sci:<26}{zh:<9}用{got:<4} "
            f"累计{stats['ok']:<6} ETA {eta:.0f}分")
        write_json_atomic(os.path.join(a.out, "progress.json"),
                          {"stage": "download", "species_done": done_sp,
                           "species_total": len(todo), "ok": stats["ok"],
                           "fail": stats["fail"], "eta_min": round(eta, 1),
                           "elapsed_min": round(el / 60, 1),
                           "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    pool.shutdown()
    fh.close()
    log(f"完成 新增 {stats['ok']} 张 失败 {stats['fail']} -> {ds_path}")
    write_json_atomic(os.path.join(a.out, "progress.json"),
                      {"stage": "done", "species_done": done_sp,
                       "species_total": len(todo), "ok": stats["ok"],
                       "fail": stats["fail"],
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    # 目标对齐到实抓值：探测期目标含"已被 GBIF 抓过的观测"（去重损耗），
    # 不对齐的话进度条永远停在 70~90%
    write_json_atomic(os.path.join(a.out, "fetch_target_inat.json"),
                      {"total": stats["ok"], "ts": time.strftime("%H:%M:%S")})


if __name__ == "__main__":
    main()

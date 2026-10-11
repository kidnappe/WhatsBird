# -*- coding: utf-8 -*-
r"""GBIF 抓一批「没训过的华北图」当独立评测集（无偏抽样，刻意不用旧模型筛）。

为什么要新抓而不是用现有图：
  * 现有 huabei/images 33,637 张里 78% 已进 det_train（bench 华北口径实测），
    "华北检出率 96.5%" 是泄漏数字（模型见过答案）；
  * 剩下 5,151 张未进训练的图，90% 是「现役旧模型全帧+中心扫描都没框到鸟」
    被 make_train_json 剔掉的 —— 那是旧模型的失败案例集，拿它测会严重低估。
  两条路都有偏，所以只能新抓一批「不经过任何模型筛选」的随机样本。

与 build_dataset_gbif.py 的关系：那支是「建训练集」（按物种凑够 N 张、做
   许可过滤、写 dataset.csv）；这支只做「评测集」，不改 dataset.csv、不碰训练数据。
   复用的是同一套 GBIF 查询口径（同 bbox / mediaType / hasCoordinate）。

无偏要点（防重蹈覆辙）：
  1. 每个物种只取「随机 offset」的一页，不按物种补齐到固定张数；
  2. 不做「有没有鸟」预筛（那是旧模型干的活，正是偏差来源）；
  3. 与现有 dataset.csv 按 gbifKey 去重，保证「没训过」。

产物：<out>/images/*.jpg + <out>/manifest.csv（file,sci,gbif_key,license,lat,lon,url）
    + <out>/fetch_progress.json（心跳）

用法（先小样验证）：
  python fetch_gbif_eval.py --out "E:\code\code tools\data\huabei_eval" \
      --species tools\dataset_traits\species_huabei_ioc150.csv --per-species 4 --limit 20
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 训练期间别把 stdout 编码搞炸（GBK 控制台编不出某些字符就整脚本崩）
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

SEARCH = "https://api.gbif.org/v1/occurrence/search"
UA = "ObserveBird-eval-set/0.1 (independent held-out benchmark; research)"
COMMERCIAL = {
    "CC0_1_0", "CC_BY_1_0", "CC_BY_2_0", "CC_BY_2_5", "CC_BY_3_0", "CC_BY_4_0",
    "CC_BY_SA_1_0", "CC_BY_SA_2_0", "CC_BY_SA_2_5", "CC_BY_SA_3_0", "CC_BY_SA_4_0",
}
# 华北 bbox（与 build_dataset_gbif 的 huabei 口径一致：lat,lon）
BBOX = ((32.0, 42.0), (110.0, 120.0))
PACE = 0.34          # GBIF 建议 ~1 req/s；每个请求后歇一下


def log(m):
    print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def get_json(url, tries=5, timeout=45):
    """带指数退避的重试。GBIF 偶发 5xx/超时，不该让整轮抓取崩掉。"""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(1.5 ** i)
                continue
            log("  HTTP %s: %s" % (e.code, url[:120]))
            return None
        except Exception as e:
            if i == tries - 1:
                log("  失败(%s): %s" % (type(e).__name__, url[:120]))
                return None
            time.sleep(1.5 ** i)
    return None


def lic_enum(lic):
    """把 license URL 归一成 GBIF 的枚举名（如 CC_BY_4_0 / CC0_1_0）。

    ⚠️ 踩过的坑：原来用 `lic.rsplit("/",1)[-1]` 取最后一段，而真实 URL 是
      http://creativecommons.org/licenses/by/4.0/legalcode
    最后一段恒为 "legalcode" -> 所有记录都被判成"许可不符"（实测 1000/1000 被拒、
    一张没下下来）。要先剥掉末尾的 legalcode，取真正的版本段。

      .../licenses/by/4.0/legalcode        -> CC_BY_4_0
      .../licenses/by-nc/4.0/legalcode     -> CC_BY_NC_4_0
      .../publicdomain/zero/1.0/           -> CC0_1_0
    """
    if not lic:
        return ""
    s = lic.rstrip("/")
    if s.endswith("/legalcode"):
        s = s[:-len("/legalcode")]
    parts = [p for p in s.split("/") if p]
    if not parts:
        return ""
    if "publicdomain" in parts:
        return "CC0_1_0"
    # 形如 .../licenses/<kind>/<ver>  -> CC_<KIND>_<VER>
    # （GBIF 的枚举名带 CC_ 前缀，如 CC_BY_4_0；漏掉前缀会一条都匹配不上）
    for i, p in enumerate(parts):
        if p == "licenses" and i + 2 < len(parts):
            kind, ver = parts[i + 1], parts[i + 2]
            return ("CC_%s_%s" % (kind, ver)).replace("-", "_").replace(".", "_").upper()
    return ""


_IMG_VARIANT = re.compile(
    r"^(https?://[^/]+/photos/\d+)/[^/]+\.(?:jpe?g|png)$", re.I)


def best_image_url(u):
    """把 iNat 的 original.jpg 换成 medium.jpg。

    ⚠️ 实测（2026-10-06，本机跨境）：同张照片 original 39~41 秒 / 900~1170KB，
    medium 只要 1.6 秒 / 89~120KB —— 快 25 倍，且短边约 500px，远超训练图
    （短边 256）所需。原实现直接用 GBIF 给的 identifier（指向 original），
    导致每张图 40 秒，300 张要跑 9 小时（实测 109 秒/张）。
    项目里 fetch_inat_region.py 早就做了 square->medium 的替换，这里同款处理。
    """
    m = _IMG_VARIANT.match(u or "")
    if m:
        return m.group(1) + "/medium.jpg"
    return u


def load_species(path):
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("our_source") or row.get("ioc_name") or "").strip()
            if name:
                out.append(name)
    return out


def existing_keys(csv_paths):
    """已训过的 gbifKey 集合（保证新图没进过训练）。"""
    keys = set()
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    k = (row.get("gbif_key") or "").strip()
                    if k:
                        keys.add(k)
                    occ = (row.get("occurrence_key") or "").strip()
                    if occ:
                        keys.add(occ)
        except Exception:
            pass
    return keys


def search(tkey, offset, limit):
    lat, lon = BBOX
    qs = {"taxonKey": tkey, "country": "CN", "mediaType": "StillImage",
          "hasCoordinate": "true", "hasGeospatialIssue": "false",
          "decimalLatitude": "%s,%s" % lat, "decimalLongitude": "%s,%s" % lon,
          "limit": limit, "offset": offset}
    url = SEARCH + "?" + urllib.parse.urlencode(qs)
    d = get_json(url)
    time.sleep(PACE)
    return d or {"results": [], "count": 0}


def taxon_key(sci):
    url = ("https://api.gbif.org/v1/species/match?"
           + urllib.parse.urlencode({"name": sci, "kingdom": "Animalia",
                                     "class": "Aves"}))
    d = get_json(url, tries=3)
    time.sleep(PACE)
    if not d or not d.get("usageKey"):
        return None, None
    return d["usageKey"], d.get("canonicalName") or sci


def fetch_image(url, tries=2):
    """下载单张图（并行调用）。失败返回 None，由调用方计 fail 并释放去重占位。"""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(0.8)
    return None


def save_image(raw, path):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    w, h = img.size
    if min(w, h) > 256:                      # 与训练图同规格（短边 256）
        s = 256.0 / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                         Image.BILINEAR)
    img.save(path, "JPEG", quality=90)
    return img.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, help="物种名单 csv（含 ioc_name 或 our_source）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-species", type=int, default=4,
                    help="每个物种随机取几张（默认 4，走随机 offset）")
    ap.add_argument("--limit", type=int, default=0, help="总量上限（0=不限，用于小样验证）")
    ap.add_argument("--exclude-csv", nargs="*", default=[],
                    help="已有的 dataset.csv，按 gbifKey 去重")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--commercial-only", action="store_true",
                    help="只收 CC0/CC-BY/CC-BY-SA。默认**不收**：训练集自身 97.4%% 是 "
                         "CC-BY-NC（当年为凑量放宽），评测集若只留可商用许可就与训练分布"
                         "不一致，测出的分数不代表模型真实表现。评测图不外发，只记录 license。")
    a = ap.parse_args()

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.csv")
    prog_path = os.path.join(a.out, "fetch_progress.json")

    species = load_species(a.species)
    log("物种名单 %d 个：%s" % (len(species), a.species))
    used = existing_keys(a.exclude_csv)
    log("已训 gbifKey 去重集：%d 个" % len(used))

    rng = random.Random(a.seed)
    rows, n_img, n_skip_dup, n_skip_lic, n_fail = [], 0, 0, 0, 0
    t0 = time.time()

    for si, sci in enumerate(species):
        if a.limit and n_img >= a.limit:
            break
        tkey, canon = taxon_key(sci)
        if not tkey:
            log("[%d/%d] %s -> 无 taxonKey，跳过" % (si + 1, len(species), sci))
            continue
        # ★ 无偏核心：先问总数，再随机挑一页（不是从 offset=0 顺着取）
        head = search(tkey, 0, 1)
        total = int(head.get("count") or 0)
        if total == 0:
            continue
        got = 0
        tries = 0
        # ★ 无偏核心：在**全量**范围内随机取页，而不是只在前几百条里转。
        # 踩过的坑：训练集当年就是沿 GBIF 默认顺序（gbifID 降序 = 最新在前）抓的，
        # 所以把随机范围限制在 offset<900 会反复落回训练集已覆盖的区间——实测
        # 11 个物种跑 251 秒、下载 0 张（去重跳过 977）。
        # GBIF 深翻页硬上限 100,000，所以把随机上界开到 min(total, 99000)。
        #
        # 效率：GBIF 搜索单次 5~25 秒（实测跨境），所以**一页就取满**（limit=300
        # 是 GBIF 上限），从这一页里挑够 per-species 张；不要"每页只要 2 张、
        # 失败了再翻 8 页"——那会把十几秒的请求浪费掉（实测 56 秒/张）。
        while got < a.per_species and tries < 4:
            tries += 1
            hi = max(1, min(total, 99000) - 1)
            off = rng.randrange(0, hi)
            page = search(tkey, off, 300)
            res = page.get("results") or []
            if not res:
                continue
            rng.shuffle(res)
            batch = []
            for r in res:
                if got + len(batch) >= a.per_species:
                    break
                if a.limit and n_img + len(batch) >= a.limit:
                    break
                gk = str(r.get("key") or "")
                if not gk or gk in used:
                    n_skip_dup += 1
                    continue
                lic = lic_enum(r.get("license"))
                # 默认不卡许可（见 --commercial-only 的说明）：训练集 97.4% 是 NC，
                # 评测集卡许可会既筛掉 97% 的样本、又让分布与训练不一致。
                if a.commercial_only and lic not in COMMERCIAL:
                    n_skip_lic += 1
                    continue
                media = [m for m in (r.get("media") or [])
                         if m.get("type") == "StillImage" and m.get("identifier")]
                if not media:
                    continue
                batch.append((r, gk, lic, best_image_url(media[0]["identifier"])))
                used.add(gk)
            # 图片并行下载：iNat CDN 单张 1~4 秒，串行会在这一步上积压
            if batch:
                with ThreadPoolExecutor(max_workers=6) as ex:
                    futs = {ex.submit(fetch_image, u): (r, gk, lic, u)
                            for r, gk, lic, u in batch}
                    for fu in futs:
                        r, gk, lic, u = futs[fu]
                        try:
                            raw = fu.result()
                        except Exception:
                            raw = None
                        if not raw:
                            n_fail += 1
                            used.discard(gk)
                            continue
                        fn = "%s_%s.jpg" % (canon.replace(" ", "_"), gk)
                        try:
                            w, h = save_image(raw, os.path.join(img_dir, fn))
                        except Exception:
                            n_fail += 1
                            used.discard(gk)
                            continue
                        rows.append({"file": fn, "scientific_name": canon,
                                     "gbif_key": gk, "occurrence_key": gk,
                                     "license": lic,
                                     "lat": r.get("decimalLatitude"),
                                     "lon": r.get("decimalLongitude"),
                                     "year": r.get("year"), "month": r.get("month"),
                                     "w": w, "h": h, "url": u,
                                     "attribution": r.get("references") or ""})
                        n_img += 1
                        got += 1
        log("[%d/%d] %-28s 取 %d 张（库内 %d）累计 %d"
            % (si + 1, len(species), canon, got, total, n_img))
        json.dump({"stage": "fetch", "species_done": si + 1,
                   "species_total": len(species), "images": n_img,
                   "skip_dup": n_skip_dup, "skip_license": n_skip_lic,
                   "fail": n_fail, "elapsed_sec": round(time.time() - t0, 1)},
                  open(prog_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with open(man_path, "w", encoding="utf-8-sig", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log("完成：%d 张 -> %s" % (n_img, img_dir))
    log("去重跳过 %d / 许可不符 %d / 下载失败 %d / 用时 %.0fs"
        % (n_skip_dup, n_skip_lic, n_fail, time.time() - t0))
    log("manifest: %s" % man_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
r"""给下载页取「每个物种用哪张照片」，四条硬标准同时满足才用：

  1. **物种对得上**：iNaturalist 按学名查 taxon，照片就是这个种；
  2. **模型认得出**：用仓库自带的 yolox 640 + 该区域分类器跑一遍，
     top-1 学名必须等于目标学名（认不出来的照片说明内容不可靠，别用）；
  3. **许可干净**：只收 CC0 / CC-BY（可商用），署名一并记进 manifest；
     --allow-nc 才放行 CC-BY-NC（App 目前免费分发，NC 成立，但将来要收费就得换图）；
  4. **分辨率够**：页面最大的图片区有 1070 px 宽。

为什么不用训练集里的图：那边 100% 是 CC-BY-NC（iNaturalist 默认许可），
且存的是 341x256 缩略图——许可和清晰度都不够。详见 docs/legal/LICENSES.md。

Usage: python fetch_photos.py [--allow-nc] [--per-species 8]
"""
import argparse
import json
import os
import sys
import urllib.request

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_identify import classify, detect_best, expand_crop  # noqa: E402

CACHE = r"E:\code\observebird\.shots\inat"
OUT = r"E:\code\observebird\site\shots\manifest.json"
API = "https://api.inaturalist.org/v1"

# 页面要出现的物种：key -> (学名, 中文名, 用哪个区域模型验)
WANT = {
    "hong-er": ("Pycnonotus jocosus", "红耳鹎", "huanan_sw"),
    "hongshui": ("Phoenicurus fuliginosus", "红尾水鸲", "dongbei"),
    "huixique": ("Cyanopica cyanus", "灰喜鹊", "dongbei"),
    "maque": ("Passer montanus", "麻雀", "dongbei"),
    "queshu": ("Copsychus saularis", "鹊鸲", "huadong"),
}

OK_LIC = {"cc0", "cc-by", "public domain"}
OK_LIC_NC = OK_LIC | {"cc-by-nc", "cc-by-nc-sa"}
UA = "WhatsBird-site-builder/1.0 (bird page; contact via GitHub issues)"


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def jget(url):
    return json.loads(get(url).decode("utf-8"))


def taxon_id(sci):
    d = jget("%s/taxa?q=%s&rank=species" % (API, urllib.parse.quote(sci)))
    for t in d.get("results", []):
        if t.get("name", "").lower() == sci.lower():
            return t["id"]
    return d["results"][0]["id"] if d.get("results") else None


def photos_for(sci, allowed, want):
    tid = taxon_id(sci)
    if tid is None:
        return []
    out = []
    for page in (1, 2):
        url = ("%s/observations?taxon_id=%s&photos=true&quality_grade=research"
               "&per_page=50&order_by=votes&page=%d" % (API, tid, page))
        d = jget(url)
        for o in d.get("results", []):
            for p in o.get("photos", []) or []:
                lic = (p.get("license_code") or "").lower()
                if lic not in allowed:
                    continue
                u = p.get("url") or ""
                if not u:
                    continue
                u_large = u.replace("/square.", "/large.").replace("/medium.", "/large.")
                out.append({"url": u_large, "original": u.replace("/square.", "/original.")
                            .replace("/medium.", "/original."),
                            "license": lic,
                            "attribution": (p.get("attribution") or ""),
                            "user": ((p.get("user") or {}).get("login") or ""),
                            "obs": "%s/observations/%s" % (API.split("/v1")[0], o.get("id")),
                            "obs_id": o.get("id")})
        if len(out) >= want * 4:
            break
    # 去重（同一 photo id 会在多条观察里出现）
    seen, uniq = set(), []
    for p in out:
        key = p["url"].split("/photos/")[-1]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def download(p, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    try:
        with urllib.request.urlopen(urllib.request.Request(p["url"], headers={"User-Agent": UA}),
                                    timeout=90) as r:
            data = r.read()
        if len(data) < 20000:
            return None
        with open(path, "wb") as f:
            f.write(data)
        im = Image.open(path)
        im.verify()
        with Image.open(path) as im:
            w, h = im.size
        return (path, w, h)
    except Exception as e:
        print("      下载失败", name, e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-nc", action="store_true", help="放行 CC-BY-NC（默认只收 CC0/CC-BY）")
    ap.add_argument("--per-species", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=0.35)
    a = ap.parse_args()
    allowed = OK_LIC_NC if a.allow_nc else OK_LIC
    os.makedirs(CACHE, exist_ok=True)

    manifest = {}
    if os.path.exists(OUT):
        try:
            manifest = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            manifest = {}

    for key, (sci, zh, region) in WANT.items():
        cands = photos_for(sci, allowed, a.per_species)
        print("%-9s %-22s license candidates %d (allowed %s)" % (key, sci, len(cands), "/".join(sorted(allowed))))
        got = tried = matched = 0
        best = None
        for c in cands:
            if got >= a.per_species:
                break
            tried += 1
            dl = download(c, "%s_%02d.jpg" % (key, tried))
            if not dl:
                continue
            got += 1
            path, w, h = dl
            if w < 700:
                os.remove(path)
                continue
            det, wh = detect_best(path)
            if not det or det["score"] < a.min_score:
                continue
            rect = expand_crop(det["box"], wh)
            try:
                top5 = classify(Image.open(path).convert("RGB").crop(
                    tuple(int(v) for v in rect)), region)
            except Exception as e:
                print("      分类失败", e)
                continue
            if not top5 or top5[0]["sci"].lower() != sci.lower():
                continue
            matched += 1
            cand = {"key": key, "sci": sci, "zh": zh, "region": region,
                    "file": os.path.basename(path), "path": path, "img_wh": [w, h],
                    "det_score": round(det["score"], 4),
                    "top1_p": round(top5[0]["p"], 4),
                    "det_box": [round(v, 1) for v in det["box"]],
                    "top5": [{"sci": t["sci"], "zh": t["zh"], "p": round(t["p"], 4)} for t in top5],
                    "license": c["license"], "attribution": c["attribution"],
                    "user": c["user"], "source": c["obs"], "photo_url": c["original"]}
            # 先要认得准（top1 概率），再要框分高，最后才看分辨率
            rank = (cand["top1_p"], cand["det_score"], w * h)
            best_rank = (best["top1_p"], best["det_score"],
                         best["img_wh"][0] * best["img_wh"][1]) if best else None
            if best is None or rank > best_rank:
                best = cand
            print("      [ok] %-14s %dx%d det=%.2f top1=%s(%.2f) %s" % (
                os.path.basename(path), w, h, cand["det_score"],
                top5[0]["zh"], top5[0]["p"], c["license"]))
        print("   -> downloaded %d / model top1 hit %d / %s" % (
            got, matched, ("picked " + best["file"]) if best else "no usable photo"))
        if best:
            manifest[key] = best

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("写出", OUT, "共", len(manifest), "个物种")


if __name__ == "__main__":
    import urllib.parse  # noqa: F401  （photos_for 里用）
    main()
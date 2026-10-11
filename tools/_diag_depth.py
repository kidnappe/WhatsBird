# -*- coding: utf-8 -*-
"""问清「每个物种还有多少新图可用」——用于决定取样策略。跑完即删。

结论用途：训练集是从每个物种的**最新那批**（gbifID 降序的前若干页）抓的，
所以新图集中在**深 offset**。与其随机乱撞，不如直接瞄准深区。
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, r"E:\code\observebird\tools")
from fetch_gbif_eval import BBOX, SEARCH, UA  # noqa: E402


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            time.sleep(1.2 ** i)
    return None


used = set()
with open(r"E:\code\code tools\data\huabei\dataset.csv",
          encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        for k in ("gbif_key", "occurrence_key"):
            v = (row.get(k) or "").strip()
            if v:
                used.add(v)

sp = []
with open(r"E:\code\observebird\tools\dataset_traits\species_huabei_ioc150.csv",
          encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        n = (row.get("our_source") or row.get("ioc_name") or "").strip()
        if n:
            sp.append(n)

print("抽查 8 个物种：total / 首个全新 offset")
print("  %-26s %7s %10s %9s" % ("物种", "total", "全新起点", "新增量"))
print("  " + "-" * 58)
lat, lon = BBOX
grand_new = 0
for sci in sp[:8]:
    d = get("https://api.gbif.org/v1/species/match?"
            + urllib.parse.urlencode({"name": sci, "kingdom": "Animalia", "class": "Aves"}))
    time.sleep(0.35)
    tk = (d or {}).get("usageKey")
    if not tk:
        continue
    base = {"taxonKey": tk, "country": "CN", "mediaType": "StillImage",
            "hasCoordinate": "true", "decimalLatitude": "%s,%s" % lat,
            "decimalLongitude": "%s,%s" % lon, "limit": 100}
    h = get(SEARCH + "?" + urllib.parse.urlencode(dict(base, offset=0, limit=0)))
    tot = int((h or {}).get("count") or 0)
    time.sleep(0.35)
    # 二分找"第一个全新区间"的起点（粗扫每 300 一条）
    first_new = None
    for off in range(0, min(tot, 3000), 300):
        p = get(SEARCH + "?" + urllib.parse.urlencode(dict(base, offset=off)))
        time.sleep(0.35)
        ks = [str(r.get("key")) for r in ((p or {}).get("results") or [])]
        if ks and all(k not in used for k in ks):
            first_new = off
            break
    new_amt = (tot - first_new) if first_new is not None else 0
    grand_new += new_amt
    print("  %-26s %7d %10s %9d"
          % (sci, tot, (str(first_new) if first_new is not None else "无"), new_amt))

print()
print("8 物种新增合计 %d  ->  推算 150 物种约 %d 张新图可用"
      % (grand_new, grand_new / 8 * 150))

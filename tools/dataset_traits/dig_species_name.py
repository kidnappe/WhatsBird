# -*- coding: utf-8 -*-
"""「挖」= 找出这批记录在 GBIF 里到底挂在哪个学名下面。

背景：名单用了最新学名（如牛背鹭 Ardea coromanda），GBIF 名录里没这个名字，
下载器就取到 0 条。但记录可能就在库里，只是挂在别的名字下（分类学改来改去）。

做法：
  1. 用「属」当钥匙：species/match 拿属 key → /species/{key}/children 列出属下所有种
  2. 按关键词（coromand / dubius / sinensis ...）挑出正主
  3. 对每个候选名字查两个数：目标 bbox 内记录数、全国记录数
  4. 校准：先查该 bbox 内鸟类总记录数，判断「0 条」是这鸟真没有，还是整个区域都稀疏
"""
import json
import time
import urllib.parse
import urllib.request

BB = "decimalLatitude=31,37&decimalLongitude=105,117"   # 华中 bbox
AVES = 212                                              # GBIF 鸟纲 taxonKey
TARGETS = [
    ("牛背鹭",  "Ardea coromanda",      ["Ardea", "Bubulcus"],       ("coromand", "ibis")),
    ("金眶鸻",  "Thinornis dubius",     ["Charadrius", "Thinornis"],  ("dubius",)),
    ("黄苇鳽",  "Botaurus sinensis",    ["Ixobrychus", "Botaurus"],   ("sinensis",)),
    ("黄腹鹨",  "Anthus japonicus",     ["Anthus"],                   ("japonicus", "rubescens")),
    ("凤头鹰",  "Lophospiza trivirgata", ["Accipiter", "Lophospiza"],  ("trivirgat",)),
    ("金翅雀",  "Chloris sinica",       ["Carduelis", "Chloris"],     ("sinica",)),
]


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ObserveBird-dig/1.0"})
            with urllib.request.urlopen(req, timeout=60) as f:
                return json.load(f)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def count(taxon_key, bbox=None, country=None):
    u = (f"https://api.gbif.org/v1/occurrence/search?taxonKey={taxon_key}"
         f"&limit=0&hasCoordinate=true")
    if bbox:
        u += "&" + bbox
    if country:
        u += f"&country={country}"
    return get(u).get("count", 0)


print("=== 校准：华中 bbox 内鸟类（Aves）带坐标记录总数 ===")
tot = count(AVES, BB)
print(f"    {tot} 条" + ("   → 区域本身很稀疏，0 条不能说明问题" if tot < 5000 else "   → 区域数据量正常"))

for cn, listed, genera, kws in TARGETS:
    print(f"\n=== {cn}   （名单里写的：{listed}）===")
    hit = False
    for g in genera:
        m = get("https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(g))
        gk = m.get("usageKey")
        if not gk:
            print(f"  [属 {g}] GBIF 名录里没有这个属")
            continue
        kids = get(f"https://api.gbif.org/v1/species/{gk}/children?limit=300").get("results", [])
        cand = [k for k in kids
                if any(w in (k.get("canonicalName") or "").lower() for w in kws)]
        if not cand:
            print(f"  [属 {g}] 属下 {len(kids)} 个种，没有名字含 {'/'.join(kws)} 的")
            continue
        for k in cand:
            nm = k.get("canonicalName") or k.get("scientificName")
            b = count(k["key"], BB)
            n = count(k["key"], country="CN")
            mark = "  <== 记录在这儿" if n > 0 else ""
            print(f"  [属 {g}] {nm:<34s} 华中bbox={b:<8d} 中国={n}{mark}")
            if b > 0:
                hit = True
    if not hit:
        print("  → 目标区里取不到：换名字也没用，得换数据源（iNat 直拉 / Commons / eBird）")

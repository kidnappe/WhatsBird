# -*- coding: utf-8 -*-
"""建立「三套名录的名字对照表」。

背景：同一个种在三套系统里名字不同，导致三处故障：
  * 下载取不到图     —— 名单用的新名 GBIF 不认（金眶鸻 Thinornis dubius 等 6 种）
  * 选种被静默跳过   —— 生活史数据集的 2021 名对不上 geomodel 的新名（1445 里 158 个）
  * 重排被记 0 分    —— eval_report.py 对未匹配的种「按 0 处理」，钉在地板上

解析阶梯（逐级下降，命中即止；候选一律要求 GBIF backbone 里 **ACCEPTED 的鸟纲**）：
  1) /species/match                    → EXACT + SPECIES
  2) 中文名当桥：查生活史数据集的同名条目 → 拿它的 2021 名再 match
     （中文名比学名稳定，实测能把 金眶鸻→Charadrius dubius、凤头鹰→Accipiter trivirgatus 桥过去）
  3) 中文名包含匹配（处理"黄苇鳽/黄斑苇鳽"这类叫法差异）
  4) 种加词词干匹配（处理性别词尾 coromanda↔coromandus），限制 Aves，按 GBIF 中国记录数排序
  5) 都不行 → 标 unresolved 并列出候选，**绝不猜**

坑（已踩过，写死在这里别再犯）：
  * /species/search 会返回 status=SYNONYM 或 status=None 的非 backbone 节点
    （Ardea coromanda 能搜到一个 key，但中国记录 0）—— 用它等于"悄悄返回一个没用的 key"
  * 不限定鸟纲会命中同名非鸟（Chloris sinica 的种加词搜曾命中 "Sinica sinica"，41 条中国记录）

输出：
  tools/dataset_traits/species_name_map.csv    人看（含依据与候选）
  tools/dataset_traits/species_name_map.json   机器用（名单名 → {gbif_name, gbif_key}）
  tools/dataset_traits/name_map_cache.json     断点续传缓存
"""
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "species_name_map.csv")
OUT_JSON = os.path.join(HERE, "species_name_map.json")
CACHE = os.path.join(HERE, "name_map_cache.json")
GEOMODEL_LABELS = r"E:\code\code tools\geomodel\BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"
TRAITS_XLSX = os.path.join(HERE, "supplFile_art_20220117115032.xlsx")
POOL_CSV = r"E:\code\code tools\bird-cn-373\code\Bird-Classification-main\species_list.csv"
POOL_FALLBACK = os.path.join(os.path.dirname(HERE), "species", "species_list_373.csv")

GBIF = "https://api.gbif.org/v1"
AVES = 212            # GBIF backbone 里鸟纲的 key
UA = {"User-Agent": "ObserveBird-namemap/1.0"}


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as f:
                return json.load(f)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def ep_stem(e: str) -> str:
    """种加词词干：去掉拉丁语性别词尾，让 coromanda/coromandus 能对上。"""
    e = e.lower()
    for suf in ("orum", "arum", "us", "um", "is", "ae", "a", "e", "i"):
        if e.endswith(suf) and len(e) - len(suf) >= 4:
            return e[: -len(suf)]
    return e


def cn_records(key: int) -> int:
    try:
        return get(f"{GBIF}/occurrence/search?taxonKey={key}&country=CN&limit=0").get("count", 0)
    except Exception:
        return -1


def is_aves_species(key: int):
    """返回 (是否鸟纲的种, 详情dict)。

    **不看 status**：GBIF 的 occurrence 数据并不都来自 backbone，很多可用的
    key 的 status 是 None（实测 Chloris sinica key=6101015，status=None，
    却有 34,944 条中国有坐标记录；而 backbone 里的 Carduelis sinica 只有 3 条）。
    原来要求 status==ACCEPTED，等于把这些 key 全扔了 —— 金翅雀就是这么被漏掉的。
    """
    try:
        d = get(f"{GBIF}/species/{key}")
    except Exception:
        return False, {}
    ok = (d.get("class") == "Aves" and d.get("rank") == "SPECIES")
    return ok, d


def match_one(name: str):
    """name → (key, 学名)，只要 GBIF 匹配到 SPECIES 且**规范名就是它**。

    不能只认 EXACT：实测 `Phylloscopus sibilatrix` 返回的是 **FUZZY**，但
    canonicalName 就是它本身、rank=SPECIES、class=Aves、中国 125 条记录 ——
    只认 EXACT 会把它白扔掉（这是我的 bug，不是数据问题）。
    判据：matchType ∈ {EXACT, FUZZY} 且（EXACT 或 规范名 == 查询名）且是鸟纲的种。
    """
    try:
        m = get(f"{GBIF}/species/match?name={urllib.parse.quote(name)}")
    except Exception:
        return None, None, None
    mt, key = m.get("matchType"), m.get("usageKey")
    if mt not in ("EXACT", "FUZZY") or m.get("rank") != "SPECIES" or not key:
        return None, None, m
    canon = (m.get("canonicalName")
             or (m.get("scientificName") or name).split(" (")[0]).strip()
    if mt == "FUZZY":
        # FUZZY 可能是"另一个近似名"，必须规范名与查询名一致、且确认是鸟纲的种
        if canon.lower() != name.strip().lower():
            return None, None, m
        ok, _ = is_aves_species(key)
        if not ok:
            return None, None, m
    return key, canon, m


def search_full_name(name: str, limit: int = 20):
    """按**全名**在 GBIF 里搜，返回 (key, 学名) 或 None。

    **不限定 status**：很多能取到记录的 key 的 status 是 None（非 backbone）。
    实测 Chloris sinica key=6101015（status=None）有 34,944 条中国有坐标记录，
    而 backbone 里的 Carduelis sinica 只有 3 条 —— 原来要求 ACCEPTED 就把
    金翅雀这类整批漏掉了。
    """
    try:
        sr = get(f"{GBIF}/species/search?q={urllib.parse.quote(name)}"
                 f"&rank=SPECIES&highertaxonKey={AVES}&limit={limit}")
    except Exception:
        return None
    low = name.strip().lower()
    for x in sr.get("results", []):
        if (x.get("canonicalName") or "").strip().lower() != low:
            continue
        k = x.get("key") or x.get("nubKey")
        if not k:
            continue
        ok, d = is_aves_species(k)
        if ok:
            return k, (d.get("canonicalName") or name)
    return None


def zh_bridge(zh: str, zh_index: dict, zh_list: list, max_cand: int = 2):
    """中文名 → 拉丁名。先精确查，再按**字符重合度**模糊匹配。

    为什么要模糊：黄苇鳽 在生活史数据集里叫「黄斑苇鳽」，直接子串匹配失败
    （中间多了个"斑"），但字符重合 3/3 → 能对上。
    """
    if not zh:
        return []
    if zh in zh_index:
        return [(zh_index[zh], "via-zh")]
    zs = set(zh)
    scored = []
    for z, lat in zh_list:
        if len(z) < 2:
            continue
        inter = len(zs & set(z))
        ratio = inter / max(1, min(len(zh), len(z)))
        if inter >= 2 and ratio >= 0.6:
            scored.append((ratio, z, lat))
    scored.sort(reverse=True)
    return [(lat, f"via-zh~{z}({sc:.2f})") for sc, z, lat in scored[:max_cand]]


def ep_variants(ep: str):
    """拉丁词尾/拼写的常见变体。

    实测：14 个"完全解析不出"的种里，**9 个只差这个词尾**
      Ardenna pacificus→pacifica、Calidris pygmeus→pygmaea、
      Carpodacus subhimachala→subhimachalus、Chelidorhynx hypoxanthus→hypoxantha、
      Ictinaetus malaiensis→malayensis、Merops leschenaultia→leschenaulti、
      Monticola cinclorhyncha→cinclorhynchus、Myiomela leucurum→leucura、
      Pycnonotus leucogenis→leucogenys
    这是**拼写规则**问题，不需要任何领域知识/AI。
    """
    e = ep.lower()
    out = []
    for a, b in [("us", "a"), ("a", "us"), ("um", "a"), ("is", "ys"), ("ys", "is"),
                 ("ae", "a"), ("i", "ae"), ("us", "um"), ("a", "um"), ("is", "i"),
                 ("ensis", "ense"), ("icus", "ica"), ("ia", "ius"), ("ea", "eus")]:
        if e.endswith(a):
            out.append(e[: -len(a)] + b)
    if "ae" in e:
        out.append(e.replace("ae", "e"))
    if "ai" in e:
        out.append(e.replace("ai", "ay"))
    if len(e) > 4:
        out.append(e[:-1])
    # 双写辅音（实测：Phylloscopus sibilatrix → GBIF 拼作 sibillatrix，双 L）
    for c in "ltnsrpmcdgf":
        if c in e:
            out.append(e.replace(c, c * 2))
    return [v for v in dict.fromkeys(out) if v != e]


def vernacular_zh(key: int) -> set:
    """该 key 的中文俗名集合（GBIF vernacularNames，只取语言以 zh 开头的）。"""
    try:
        vs = get(f"{GBIF}/species/{key}/vernacularNames?limit=200").get("results", [])
    except Exception:
        return set()
    out = set()
    for v in vs:
        if (v.get("language") or "").lower().startswith("zh"):
            nm = (v.get("vernacularName") or "").strip()
            if nm:
                out.add(nm)
    return out


def load_overrides():
    """人工确认的映射（name_overrides.csv）—— 阶梯 0，最高优先级。

    列：action(map|exclude), query_name, gbif_name, evidence
    放这里的只有"算法判不出来、由人看过证据后定的"：
      * map     —— 属被拆/并（Phoenicuropsis→Phoenicurus），或同名候选需按目/科判断
      * exclude —— GBIF 里没有种级身份（只有亚种），且保留会造成标签冲突
    每条都必须写 evidence —— 免得日后没人知道为什么这么定。
    """
    if not os.path.exists(OVERRIDES):
        return {}
    out = {}
    for r in csv.DictReader(open(OVERRIDES, encoding="utf-8-sig", newline="")):
        q = (r.get("query_name") or "").strip()
        if not q:
            continue
        out[q] = {"action": (r.get("action") or "map").strip().lower(),
                  "gbif_name": (r.get("gbif_name") or "").strip(),
                  "evidence": (r.get("evidence") or "").strip()}
    return out


OVERRIDES = os.path.join(HERE, "name_overrides.csv")
_OVERRIDE_CACHE = None


def resolve(name: str, zh: str, zh_index: dict, zh_list: list) -> dict:
    r = {"query": name, "gbif_name": None, "gbif_key": None, "match_type": None,
         "rank": None, "resolved_by": None, "cn_records": None, "candidates": ""}
    cands = []          # (中国记录数, 依据, 学名, key)
    seen = set()

    # ---- 阶梯 0：人工确认的映射（唯一有资格越过自动判定的）----
    global _OVERRIDE_CACHE
    if _OVERRIDE_CACHE is None:
        _OVERRIDE_CACHE = load_overrides()
    if name in _OVERRIDE_CACHE:
        ov = _OVERRIDE_CACHE[name]
        if ov["action"] == "exclude":
            r.update(resolved_by="excluded", candidates=f"人工排除: {ov['evidence']}")
            return r
        k0, s0, _ = match_one(ov["gbif_name"])
        if k0:
            r.update(gbif_name=s0, gbif_key=k0, resolved_by="override",
                     cn_records=cn_records(k0), candidates=f"人工确认: {ov['evidence']}")
            return r
        r["candidates"] += f"[override 指定的 {ov['gbif_name']} 也匹配不上] "

    # ---- 阶梯 1：直接 match ----
    # 命中就**立刻返回**（只花 2 次 API）。原来即使命中也要把所有阶梯跑完
    # （最多 15 次调用），1494 个名字要 3 小时 —— 没必要。
    key, sci, m = match_one(name)
    r["match_type"] = (m or {}).get("matchType")
    r["rank"] = (m or {}).get("rank")
    if key:
        n = cn_records(key)
        r.update(gbif_name=sci, gbif_key=key, resolved_by="match", cn_records=n,
                 candidates=f"{sci}={n}")
        return r

    # ---- 阶梯 2：中文名当桥（对中国的种最可靠 —— 中文名比学名稳定）----
    #  桥过来的名字本身也可能 GBIF 不认（金翅雀两边都叫 Chloris sinica），
    #  所以桥出来的名字同样要"match → 全名搜索"走一遍。
    for bn, how in zh_bridge(zh, zh_index, zh_list):
        if bn in seen:
            continue
        k2, s2, _ = match_one(bn)
        if k2:
            cands.append((cn_records(k2), how, s2, k2))
            seen.add(k2)
            break
        hit = search_full_name(bn)
        if hit:
            cands.append((cn_records(hit[0]), how + "+search", hit[1], hit[0]))
            seen.add(hit[0])
            break

    # ---- 阶梯 3：拿**我们自己的名字**做全名搜索 ----
    #  认非 backbone 的 key：金翅雀的真 key=6101015（status=None）有 34,944 条
    #  中国有坐标记录，而 backbone 里的 Carduelis sinica 只有 3 条。
    if not cands:
        hit = search_full_name(name)
        if hit:
            cands.append((cn_records(hit[0]), "search-fullname", hit[1], hit[0]))
            seen.add(hit[0])

    # ---- 阶梯 4：种加词匹配，**只在同属下才认**，否则不猜 ----
    #  为什么加同属约束：种加词会撞车。实测 dubius → Carpodacus dubius(3577)
    #  排在正确的 Charadrius dubius(209) 前面；sinensis → Pycnonotus sinensis
    #  (101089) 排第一。按"中国记录最多"选会选到**别的鸟**。
    parts = name.split()
    if len(parts) >= 2:
        genus, ep = parts[0], parts[-1]
        # ① 先试同属下的**拼写变体**（可靠：走 /species/match，命中即 EXACT）
        for cand_ep in [ep] + ep_variants(ep):
            k, s, _ = match_one(f"{genus} {cand_ep}")
            if k:
                how = "samegenus" if cand_ep == ep else f"spelling({ep}→{cand_ep})"
                cands.append((cn_records(k), how, s, k))
                seen.add(k)
                break
        # ② 拼写变体不够，再走搜索（同属约束，防止种加词撞车选错鸟）
        if not cands:
            try:
                sr = get(f"{GBIF}/species/search?q={urllib.parse.quote(ep)}"
                         f"&rank=SPECIES&highertaxonKey={AVES}&limit=60")
                added = 0
                for x in sr.get("results", []):
                    sp = (x.get("canonicalName") or "").split()
                    if len(sp) < 2 or ep_stem(sp[-1]) != ep_stem(ep):
                        continue
                    if sp[0].lower() != genus.lower():      # ← 同属才算
                        continue
                    k = x.get("key") or x.get("nubKey")
                    if not k or k in seen:
                        continue
                    ok, d = is_aves_species(k)
                    if not ok:
                        continue
                    seen.add(k)
                    cands.append((cn_records(k), "stem-samegenus",
                                  d.get("canonicalName") or " ".join(sp), k))
                    added += 1
                    if added >= 4:
                        break
            except Exception as e:
                r["candidates"] += f"[stem搜索异常 {type(e).__name__}] "

    # ---- 阶梯 5：种加词广搜（不限属）→ 用 **GBIF 的中文俗名** 验证 ----
    #  专门对付"属名被拆"的种：Phoenicuropsis frontalis（蓝额红尾鸲）在 GBIF 里
    #  叫 Phoenicurus frontalis。同属约束救不了，但查候选的中文俗名能确认身份。
    #  **必须有中文名对上才认** —— 这是证据，不是猜。
    if not cands and zh:
        ep = parts[-1] if len(parts) >= 2 else ""
        if ep:
            try:
                sr = get(f"{GBIF}/species/search?q={urllib.parse.quote(ep)}"
                         f"&rank=SPECIES&highertaxonKey={AVES}&limit=60")
                tried = 0
                for x in sr.get("results", []):
                    sp = (x.get("canonicalName") or "").split()
                    if len(sp) < 2 or ep_stem(sp[-1]) != ep_stem(ep):
                        continue
                    k = x.get("key") or x.get("nubKey")
                    if not k or k in seen or tried >= 8:
                        continue
                    tried += 1
                    ok, d = is_aves_species(k)
                    if not ok:
                        continue
                    if zh in vernacular_zh(k):      # ← 中文名对上才算
                        seen.add(k)
                        cands.append((cn_records(k), "via-vernacular-zh",
                                      d.get("canonicalName") or " ".join(sp), k))
                        break
            except Exception as e:
                r["candidates"] += f"[俗名验证异常 {type(e).__name__}] "

    if not cands:
        r["candidates"] = (r["candidates"] or "") + "无候选，需人工确认（属名已变，种加词也撞车）"
        return r
    cands.sort(key=lambda c: -c[0])
    r["candidates"] += "; ".join(f"{c[2]}={c[0]}" for c in cands[:5])
    best = next((c for c in cands if c[0] > 0), cands[0])
    r.update(cn_records=best[0], gbif_name=best[2], gbif_key=best[3], resolved_by=best[1])
    return r


def load_inputs():
    """name -> {zh, src}；同时建 中文名 -> 拉丁名 的索引（来自生活史数据集）。"""
    names, zh_index, zh_list = {}, {}, []

    def add(n, z, src):
        n, z = (n or "").strip(), (z or "").strip()
        if not n:
            return
        if n not in names:
            names[n] = {"zh": z, "src": src}
        elif z and not names[n]["zh"]:
            names[n]["zh"] = z

    if os.path.exists(TRAITS_XLSX):
        import openpyxl
        wb = openpyxl.load_workbook(TRAITS_XLSX, read_only=True, data_only=True)
        rows = [r for r in wb["Chinesebirdsdata"].iter_rows(values_only=True)][2:]
        wb.close()
        for r in rows:
            if r[4]:
                add(r[4], r[2], "traits1445")
                z = str(r[2]).strip()
                if z:
                    zh_index.setdefault(z, str(r[4]).strip())
                    zh_list.append((z, str(r[4]).strip()))
    p = POOL_CSV if os.path.exists(POOL_CSV) else POOL_FALLBACK
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            add(r.get("scientific_name"), r.get("common_name"), "pool373")
    return names, zh_index, zh_list


def load_geomodel_index():
    if not os.path.exists(GEOMODEL_LABELS):
        return {}
    idx = {}
    for i, l in enumerate(open(GEOMODEL_LABELS, encoding="utf-8",
                               errors="replace").read().splitlines()):
        p = l.split("\t")
        if len(p) >= 2:
            idx.setdefault(p[1].strip().lower(), i)
    return idx


def main():
    names, zh_index, zh_list = load_inputs()
    gm = load_geomodel_index()
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    if "--retry-failed" in sys.argv:
        bad = [k for k, v in cache.items() if not (v or {}).get("gbif_name")]
        for k in bad:
            cache.pop(k)
        print(f"--retry-failed: 清掉 {len(bad)} 条未解析的，只重查这些", flush=True)
    todo = [n for n in sorted(names) if n not in cache]
    print(f"名字总数 {len(names)}（已缓存 {len(cache)}，本轮需查 {len(todo)}）", flush=True)
    print(f"中文名索引 {len(zh_index)} 条；geomodel 标签 {len(gm)} 个", flush=True)
    t0 = time.time()
    # 并发查（GBIF 吃得下；项目自己的下载器就用 12 个 worker）。
    # 顺序无关（结果按名字入缓存），缓存写入放在主线程定期做。
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(resolve, n, names[n]["zh"], zh_index, zh_list): n
                for n in todo}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                cache[n] = fut.result()
            except Exception as e:
                cache[n] = {"query": n, "gbif_name": None, "gbif_key": None,
                            "resolved_by": None,
                            "candidates": f"异常 {type(e).__name__}: {e}"}
            done += 1
            if done % 50 == 0 or done == len(todo):
                json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
                el = time.time() - t0
                print(f"  ... {done}/{len(todo)}  已用 {el/60:.1f} 分钟  "
                      f"预计还需 {el/done*(len(todo)-done)/60:.1f} 分钟", flush=True)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)

    rows, ok, n_excl, unresolved = [], 0, 0, []
    for n in sorted(names):
        c = cache.get(n) or {}
        zh = names[n]["zh"]
        gname, gkey = c.get("gbif_name"), c.get("gbif_key")
        is_excl = (c.get("resolved_by") == "excluded")
        gi = gm.get(n.lower())
        if gi is None and gname:
            gi = gm.get(gname.lower())
        if is_excl:
            n_excl += 1
        elif gname:
            ok += 1
        else:
            unresolved.append(n)
        rows.append({"query_name": n, "zh": zh, "src": names[n]["src"],
                     "gbif_name": gname or "", "gbif_key": gkey or "",
                     "gbif_cn_records": c.get("cn_records") if c.get("cn_records") is not None else "",
                     "resolved_by": c.get("resolved_by") or "",
                     "match_type": c.get("match_type") or "",
                     "in_geomodel": "Y" if gi is not None else "N",
                     "geomodel_index": gi if gi is not None else "",
                     "candidates": c.get("candidates") or ""})
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    slim = {}
    for r in rows:
        if r["resolved_by"] == "excluded":
            slim[r["query_name"]] = {"excluded": True, "why": r["candidates"]}
        elif r["gbif_name"]:
            slim[r["query_name"]] = {"gbif_name": r["gbif_name"], "gbif_key": r["gbif_key"],
                                     "zh": r["zh"], "geomodel_index": r["geomodel_index"]}
    json.dump(slim, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== 结果 ===", flush=True)
    print(f"  输入名字       : {len(names)}")
    print(f"  解析出 GBIF 名 : {ok}  ({ok/len(names)*100:.1f}%)")
    print(f"  人工排除       : {n_excl}")
    print(f"  未解析         : {len(unresolved)}")
    print(f"  有中国记录     : {sum(1 for r in rows if isinstance(r['gbif_cn_records'], int) and r['gbif_cn_records'] > 0)}")
    print(f"  geomodel 命中  : {sum(1 for r in rows if r['in_geomodel'] == 'Y')}  "
          f"({sum(1 for r in rows if r['in_geomodel'] == 'Y')/len(names)*100:.1f}%)")
    by = {}
    for r in rows:
        k = (r["resolved_by"] or "(无)").split("(")[0]
        by[k] = by.get(k, 0) + 1
    for k, v in sorted(by.items(), key=lambda x: -x[1]):
        print(f"    经 {k}: {v}")
    print(f"\n  解析不出的 {len(unresolved)} 个（前 25，附候选供人工确认）:")
    for n in unresolved[:25]:
        c = cache.get(n) or {}
        print(f"    {n:<32s} {names[n]['zh']:<10s} {c.get('candidates','')[:70]}")
    print(f"\n-> {OUT_CSV}\n-> {OUT_JSON}\n-> {CACHE}")


if __name__ == "__main__":
    main()

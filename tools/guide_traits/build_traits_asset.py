# -*- coding: utf-8 -*-
"""生成「图鉴详情页」用的物种性状资产。

输入
----
tools/dataset_traits/china_bird_traits.csv    性状表正本（1445 种，2021 中国鸟类名录口径）
tools/dataset_traits/species_name_map.csv     既有 GBIF 名字映射（query_name -> gbif_name）
tools/dataset_traits/name_overrides.csv       既有手工覆盖
tools/guide_traits/rename_overrides.csv       本工具维护的异名覆盖（逐条带证据，可复核）
android/app/src/main/assets/models/labels_national_v2.txt   图鉴骨架（384 种，GBIF 口径）

输出
----
android/app/src/main/assets/guide/traits.json   端上资产（按学名索引，已归一化）
tools/guide_traits/match_report.csv             逐种匹配方法与证据（审计用，不打包）

用法
----
python tools/guide_traits/build_traits_asset.py --propose   # 只打印"中文名可补"的候选，供人工复核
python tools/guide_traits/build_traits_asset.py             # 正式生成资产 + 报告
python tools/guide_traits/build_traits_asset.py --check     # 生成后自检（不写文件）
"""
import argparse
import collections
import csv
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAITS_CSV = ROOT / "tools/dataset_traits/china_bird_traits.csv"
NAME_MAP = ROOT / "tools/dataset_traits/species_name_map.csv"
NAME_OVERRIDES = ROOT / "tools/dataset_traits/name_overrides.csv"
RENAME_OVERRIDES = Path(__file__).resolve().parent / "rename_overrides.csv"
LABELS = ROOT / "android/app/src/main/assets/models/labels_national_v2.txt"
OUT_ASSET = ROOT / "android/app/src/main/assets/guide/traits.json"
OUT_REPORT = Path(__file__).resolve().parent / "match_report.csv"

# china_bird_traits.csv 是 30 列、6 组"雄性/雌性"重名列，DictReader 会把重名键吃掉，
# 所以必须按位置取列。
C = {
    "order": 0, "family": 1, "zh": 2, "en": 3, "sci": 4, "endemic": 5,
    "mass_m": 6, "mass_f": 7,
    "length_m": 8, "length_f": 9,
    "bill_m": 10, "bill_f": 11,
    "wing_m": 12, "wing_f": 13,
    "tail_m": 14, "tail_f": 15,
    "tarsus_m": 16, "tarsus_f": 17,
    "diet": 18, "clutch": 19, "egg_size": 20, "egg_vol": 21,
    "nest_type": 22, "nest_site": 23, "flock": 24, "migrant": 25,
    "dist_range": 26, "provinces": 27,
}

MISSING = {"", "NA", "N/A", "NA.", "?", "—", "-", "－", "／", "/"}
ANOMALIES = []  # 数据异常记录（如区间写反），进报告

# 区划代码形如 I / IIA / VIIB / VIID（罗马数字 + 可选大写后缀），
# 括号里是该区的居留型集合 R/S/W/P/V，取值时丢掉。
_DIST_RE = re.compile(r"^\s*([IVX]+[A-Z]?)\s*\(")


def parse_dist_range(raw):
    """"VIIB(R)" / "VA(R),VIIB(R)" -> ["VIIB","VA"]；缺失或无合法代码 -> []。"""
    if _blank(raw):
        return []
    out = []
    for part in str(raw).split(","):
        m = _DIST_RE.match(part.strip())
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


# ---------------------------------------------------------------- 取值解析

def _blank(s):
    return s is None or s.strip() in MISSING or s.strip().upper() in ("NA", "N/A")


def parse_range(raw, where=""):
    """'325~430' -> [325.0, 430.0]；'340' -> [340.0, 340.0]；缺失/非数值 -> None。

    原表分隔符混用 ~ ～ - — –，统一后再拆。区间写反（如 28.06-19.67）按 min/max 归一，
    并记一条异常，供人工回查原表。
    """
    if _blank(raw):
        return None
    s = raw.strip()
    for sep in ("～", "~", "—", "–", "－"):
        s = s.replace(sep, "-")
    parts = [p.strip() for p in s.split("-") if p.strip()]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            return None
    if not nums:
        return None
    lo, hi = nums[0], nums[-1]
    if lo > hi:
        ANOMALIES.append(f"{where}: 区间写反 '{raw}'")
        lo, hi = hi, lo
    return [round(lo, 2), round(hi, 2)]


def parse_egg(raw, where=""):
    """'40.6*31.9' / '28-43*31-34' -> [[Lmin,Lmax],[Bmin,Bmax]]"""
    if _blank(raw):
        return None
    s = raw.strip()
    sep = "*" if "*" in s else ("×" if "×" in s else ("x" if "x" in s else None))
    if sep is None:
        return None
    a, _, b = s.partition(sep)
    L, B = parse_range(a, where + "/卵长"), parse_range(b, where + "/卵宽")
    if L is None or B is None:
        return None
    # ⚠️ 原表有 58 行把卵宽写在前面（如家燕 '13-16*18-20'、白鹭 '34*48'）。
    # 鸟卵恒为长 > 宽的长球形，故按「大的一维是卵长」归一，并记异常供回查原表。
    if L[1] < B[1]:
        ANOMALIES.append(f"{where}: 卵长/卵宽写反 '{raw}'（已按长>宽归一）")
        L, B = B, L
    return [L, B]


def parse_codes(raw):
    """'1,2' / '0,1' -> ['1','2']；'RSWP' 这种字母码另有 parse_letters。"""
    if _blank(raw):
        return None
    parts = [p.strip() for p in re.split(r"[,，、;；/]", raw) if p.strip()]
    return parts or None


def parse_letters(raw):
    """'RSWP' -> ['R','S','W','P']；'R' -> ['R']。非字母内容原样单元素返回。"""
    if _blank(raw):
        return None
    s = raw.strip()
    if re.fullmatch(r"[A-Za-z]+", s):
        return list(s.upper())
    return [s]


def parse_int(raw):
    if _blank(raw):
        return None
    try:
        return int(float(raw.strip()))
    except ValueError:
        return None


def parse_name(raw):
    """'鸡形目 GALLIFORMES' -> ('鸡形目', 'GALLIFORMES')"""
    if _blank(raw):
        return None, None
    s = raw.strip()
    m = re.match(r"^(\S+?)\s+([A-Za-z].*)$", s)
    if m:
        return m.group(1), m.group(2).strip()
    return s, None


# ---------------------------------------------------------------- 数据装载

def load_traits():
    rows = {}
    with TRAITS_CSV.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) <= C["sci"]:
                continue
            sci = row[C["sci"]].strip()
            if sci:
                rows[sci] = row
    return header, rows


def load_name_map():
    """返回 gbif_name -> query_name（仅 traits1445 来源）。"""
    gbif2q = {}
    with NAME_MAP.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("src") or "").strip() != "traits1445":
                continue
            q = (row.get("query_name") or "").strip()
            g = (row.get("gbif_name") or "").strip()
            if q and g:
                gbif2q.setdefault(g, q)
    # 既有手工覆盖（name_overrides.csv 的 map 行）
    with NAME_OVERRIDES.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("action") or "").strip() != "map":
                continue
            q = (row.get("query_name") or "").strip()
            g = (row.get("gbif_name") or "").strip()
            if q and g:
                gbif2q[g] = q
    return gbif2q


def load_rename_overrides():
    """本工具维护的异名覆盖：labels_sci -> csv_sci。"""
    ov = {}
    if not RENAME_OVERRIDES.exists():
        return ov
    with RENAME_OVERRIDES.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            a = (row.get("labels_sci") or "").strip()
            b = (row.get("csv_sci") or "").strip()
            if a and b:
                ov[a] = (b, (row.get("evidence") or "").strip())
    return ov


def load_labels():
    out = []
    with LABELS.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split("\t")
            out.append((p[0].strip(), p[1].strip() if len(p) > 1 else ""))
    return out


# ---------------------------------------------------------------- 匹配

def build_index(traits):
    by_zh = {}
    for sci, row in traits.items():
        by_zh.setdefault(row[C["zh"]].strip(), []).append(sci)
    return by_zh


def match_all(traits, gbif2q, renames):
    by_zh = build_index(traits)
    labels = load_labels()
    matched, misses, proposals = [], [], []
    for sci, zh in labels:
        if sci in traits:
            matched.append((sci, zh, "exact", sci, ""))
            continue
        if sci in gbif2q and gbif2q[sci] in traits:
            matched.append((sci, zh, "namemap", gbif2q[sci], "species_name_map.csv"))
            continue
        if sci in renames:
            tgt, ev = renames[sci]
            if tgt in traits:
                matched.append((sci, zh, "override", tgt, ev))
            else:
                misses.append((sci, zh, "override 指向的 csv_sci 不存在: " + tgt))
            continue
        cand = by_zh.get(zh, [])
        if len(cand) == 1:
            proposals.append((sci, zh, cand[0], traits[cand[0]][C["sci"]].strip()))
            misses.append((sci, zh, "待复核：中文名唯一命中 " + cand[0]))
        elif len(cand) > 1:
            misses.append((sci, zh, "中文名歧义: " + " / ".join(cand)))
        else:
            misses.append((sci, zh, "性状表中无此中文名"))
    return matched, misses, proposals


# ---------------------------------------------------------------- 资产构造

def build_entry(row):
    def pair(m, f):
        a, b = parse_range(row[C[m]], C[m]), parse_range(row[C[f]], C[f])
        if a is None and b is None:
            return None
        return [a, b]

    o_zh, o_lat = parse_name(row[C["order"]])
    f_zh, f_lat = parse_name(row[C["family"]])
    ent = {
        "order": o_zh, "orderLat": o_lat,
        "family": f_zh, "familyLat": f_lat,
        "endemic": row[C["endemic"]].strip() == "是",
        "mass": pair("mass_m", "mass_f"),
        "length": pair("length_m", "length_f"),
        "bill": pair("bill_m", "bill_f"),
        "wing": pair("wing_m", "wing_f"),
        "tail": pair("tail_m", "tail_f"),
        "tarsus": pair("tarsus_m", "tarsus_f"),
        "diet": None if _blank(row[C["diet"]]) else row[C["diet"]].strip(),
        "clutch": parse_range(row[C["clutch"]]),
        "egg": parse_egg(row[C["egg_size"]]),
        "nestType": parse_codes(row[C["nest_type"]]),
        "nestSite": parse_codes(row[C["nest_site"]]),
        "flock": parse_codes(row[C["flock"]]),
        "migrant": parse_letters(row[C["migrant"]]),
        # ⚠️ 2026-10-10 补：原先没落这一列，导致端上 traits.json 里没有分布区代码，
        # 图鉴卡片只能退化成「全国兜底」。分布区及其迁徙状态是数据集完整度 100% 的字段之一
        # （论文摘要：除迁徙状况/特有种/地理分布范围/分布省份外其余均有缺失）。
        # 格式形如 "VIIB(R)" 或 "VA(R),VIIB(R)"，解析成 ["VIIB","VA"]，字母部分（居留型）丢弃。
        "distRange": parse_dist_range(row[C["dist_range"]]),
        "provinces": parse_int(row[C["provinces"]]),
    }
    return {k: v for k, v in ent.items() if v not in (None, [], "")}


def verify_asset():
    """读回已生成的资产，检查不变量。返回进程退出码（0=通过）。"""
    if not OUT_ASSET.exists():
        print(f"资产不存在: {OUT_ASSET}")
        return 1
    data = json.loads(OUT_ASSET.read_text(encoding="utf-8"))
    sp = data.get("species", {})
    bad = []
    present = collections.Counter()
    for sci, e in sp.items():
        for key in ("mass", "length", "bill", "wing", "tail", "tarsus"):
            m = e.get(key)
            if not m:
                continue
            present[key] += 1
            for side in m:
                if side and side[0] > side[1]:
                    bad.append(f"{sci}.{key} 区间反了 {side}")
        if e.get("egg"):
            present["egg"] += 1
            if e["egg"][0][1] < e["egg"][1][1]:
                bad.append(f"{sci}.egg 卵长<卵宽 {e['egg']}")
        for k in ("diet", "clutch", "nestType", "nestSite", "flock", "migrant", "provinces"):
            if e.get(k):
                present[k] += 1
    print(f"资产物种数 {len(sp)}（骨架 384，缺口 {384 - len(sp)}）")
    print("字段覆盖:", dict(present))
    if bad:
        print(f"❌ 不变量失败 {len(bad)} 条")
        for b in bad[:20]:
            print("   " + b)
        return 1
    print("✅ 不变量全部通过（区间有序、卵长 ≥ 卵宽）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true", help="只打印中文名可补的候选")
    ap.add_argument("--check", action="store_true", help="自检匹配，不写文件")
    ap.add_argument("--verify", action="store_true", help="读回已生成资产检查不变量")
    ap.add_argument("--find", metavar="KW", nargs="+",
                    help="在性状表里按关键词（中文名/英文名/拉丁名）找行，用于人工核异名")
    args = ap.parse_args()

    if args.verify:
        return verify_asset()

    header, traits = load_traits()

    if args.find:
        for kw in args.find:
            print(f"== 关键词 {kw}")
            hits = 0
            for sci, row in traits.items():
                hay = f"{row[C['zh']]} | {row[C['en']]} | {sci} | {row[C['family']]}"
                if kw.lower() in hay.lower():
                    print(f"   {row[C['zh']]}\t{sci}\t[{row[C['family']]}]")
                    hits += 1
            if not hits:
                print("   （表里没有）")
        return

    gbif2q = load_name_map()
    renames = load_rename_overrides()
    matched, misses, proposals = match_all(traits, gbif2q, renames)
    total = len(matched) + len(misses)

    if args.propose:
        print(f"# 中文名唯一命中、可作为异名覆盖的候选：{len(proposals)} 条")
        print("# 复核要点：确认是同一物种（分类学改名），不是同名异物")
        print("labels_sci,csv_sci,csv_zh,labels_zh,evidence")
        for sci, zh, csv_sci, csv_zh in proposals:
            print(f"{sci},{csv_sci},{csv_zh},{zh},中文名唯一命中且分类学改属名")
        return

    print(f"图鉴骨架 {total} 种 | 命中 {len(matched)} | 未命中 {len(misses)} "
          f"= {len(matched)/total*100:.1f}%")
    for sci, zh, why in misses:
        print(f"  [缺] {zh}\t{sci}\t{why}")
    if ANOMALIES:
        print(f"原表数据异常 {len(ANOMALIES)} 条（已按 min/max 归一）")
    if args.check:
        print(f"原表数据异常 {len(ANOMALIES)} 条（已归一）")
        for a in ANOMALIES[:20]:
            print("   " + a)
        return

    species = {}
    for sci, zh, tier, csv_sci, ev in matched:
        species[sci] = build_entry(traits[csv_sci])

    asset = {
        "_meta": {
            "source": "中国鸟类的生活史和生态学特征数据集 (Biodiversity Science 2021, 29(9):1149-1153)",
            "doi": "10.17520/biods.2021201",
            "license": "数据论文，引用即可商用（docs/legal/LICENSES.md §D）",
            "built": datetime.date.today().isoformat(),
            "coverage": f"{len(matched)}/{total}",
            "note": "按学名索引；数值为区间 [min,max]，体长/嘴峰/翅长/尾长/跗蹠 mm，体重 g，卵 mm",
        },
        "species": species,
    }
    OUT_ASSET.parent.mkdir(parents=True, exist_ok=True)
    OUT_ASSET.write_text(json.dumps(asset, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")

    with OUT_REPORT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["labels_sci", "labels_zh", "match_tier", "csv_sci", "evidence"])
        for sci, zh, tier, csv_sci, ev in matched:
            w.writerow([sci, zh, tier, csv_sci, ev])
        for sci, zh, why in misses:
            w.writerow([sci, zh, "MISS", "", why])

    size = OUT_ASSET.stat().st_size
    print(f"\n写出 {OUT_ASSET.relative_to(ROOT)}  {size/1024:.1f} KB，{len(species)} 种")
    print(f"写出 {OUT_REPORT.relative_to(ROOT)}")
    tiers = {}
    for _, _, tier, _, _ in matched:
        tiers[tier] = tiers.get(tier, 0) + 1
    print("匹配方式分布:", tiers)


if __name__ == "__main__":
    sys.exit(main())

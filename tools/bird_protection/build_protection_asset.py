# -*- coding: utf-8 -*-
"""生成图鉴保护状态资产 assets/guide/protection.json。

数据源与口径
-----------
① `tools/bird_protection/raw/china_bird_list_v10.csv`
   取自 GitHub `makenv/bird-list-en`（中国观鸟年报《中国鸟类名录 v10.0》整理表）。
   7 列：序号 / 学名 / 中文名 / 英文名 / IUCN等级 / 国家保护等级 / 备注。
   ⚠️ 维基百科（docs/legal/LICENSES.md §D 原计划的来源）在本机网络不可达，
      改用等价整理表。授权状态见 docs/legal/LICENSES.md §D。

② `tools/bird_protection/raw/sanyou_2023_birds.csv`
   国家林业和草原局公告 2023 年第 17 号《有重要生态、科学、社会价值的陆生野生动物名录》
   的**鸟纲**部分（1028 种）。官方附件是 57 页扫描件 PDF（pypdf 抽不出文字），
   改从宁乡市政府转载页挂的 `.doc` 附件抽取，见 extract_sanyou_docx.py / extract_sanyou_birds.py。

字段口径
--------
iucn = IUCN 红色名录等级（CR/EN/VU/NT/LC/DD/NR）。LC/NR/空 = 不单独挂徽章。
cn   = 行政保护级别，取值 `一级` / `二级` / `三有` / 空。
       `一级`/`二级` 来自《国家重点保护野生动物名录》(2021 调整版)；
       `三有` 来自 2023 版三有名录。
       **两份名录互斥**（2023 版三有名录明确排除已列入重点保护名录的种），
       骨架 384 种实测交集为 0 —— 所以一个种最多只有一个行政级别，单字段够用。

异名处理
--------
骨架 labels_national_v2.txt 用较新的属名拆分口径，两份名录都还是旧属名，故两张别名表分开：
  ALIASES        —— 骨架 -> 中国鸟类名录 v10.0（取 IUCN / 一级二级）
  SANYOU_ALIASES —— 骨架 -> 三有名录 2023（取三有）
每条都由 probe_synonym.py / probe_sanyou_match.py 按「中文名」双重命中验证过。
同名不同种的（黄腹鹨：骨架 Anthus japonicus，两份名录里都是 Anthus rubescens）**不映射**。

用法：python tools/bird_protection/build_protection_asset.py
"""
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "tools/bird_protection/raw/china_bird_list_v10.csv"
SANYOU_PATH = ROOT / "tools/bird_protection/raw/sanyou_2023_birds.csv"
LABELS = ROOT / "android/app/src/main/assets/models/labels_national_v2.txt"
GUIDE_DIR = ROOT / "android/app/src/main/assets/guide"
OUT = GUIDE_DIR / "protection.json"

# 属名拆分异名：骨架新名 -> 中国鸟类名录 v10.0 旧名。须经 probe_synonym.py 双重命中验证。
ALIASES = {
    "Anarhynchus alexandrinus":   "Charadrius alexandrinus",     # 环颈鸻
    "Anarhynchus leschenaultii":  "Charadrius leschenaultii",    # 铁嘴沙鸻
    "Ardea coromanda":            "Bubulcus coromandus",         # 牛背鹭
    "Astur gentilis":             "Accipiter gentilis",          # 苍鹰
    "Botaurus sinensis":          "Ixobrychus sinensis",         # 黄苇鳽
    "Lophospiza trivirgata":      "Accipiter trivirgatus",       # 凤头鹰
    "Paradoxornis gularis":       "Psittiparus gularis",         # 灰头鸦雀
    "Paradoxornis heudei":        "Calamornis heudei",           # 震旦鸦雀
    "Periparus venustulus":       "Pardaliparus venustulus",     # 黄腹山雀
    "Saundersilarus saundersi":   "Chroicocephalus saundersi",   # 黑嘴鸥
    "Suthora alphonsiana":        "Sinosuthora alphonsiana",     # 灰喉鸦雀
    "Suthora webbiana":           "Sinosuthora webbiana",        # 棕头鸦雀
    "Thinornis dubius":           "Charadrius dubius",           # 金眶鸻
}

# 三有名录 2023 用的是更老的分类。骨架新名 -> 三有名录学名。
# 前 25 条：属名变更，中文名与三有名录完全一致（probe_sanyou_match.py 逐条验过）。
# 后 5 条：**近年从名录内亲本种拆出来的种**，名录尚未更新；
#          拆分是分类学修订、不是保护地位变化，故按亲本种计入三有。
SANYOU_ALIASES = {
    # --- 属名变更（中文名一致）---
    "Actinodura cyanouroptera":    "Siva cyanouroptera",           # 蓝翅希鹛
    "Alcippe davidi":              "Alcippe morrisonia",           # 灰眶雀鹛
    "Ardea coromanda":             "Bubulcus ibis",                # 牛背鹭
    "Caprimulgus jotaka":          "Caprimulgus indicus",          # 普通夜鹰
    "Coloeus dauuricus":           "Corvus dauuricus",             # 达乌里寒鸦
    "Copsychus malabaricus":       "Kittacincla malabarica",       # 白腰鹊鸲
    "Corvus torquatus":            "Corvus pectoralis",            # 白颈鸦
    "Parayuhina diademata":        "Yuhina diademata",             # 白领凤鹛
    "Pericrocotus speciosus":      "Pericrocotus flammeus",        # 赤红山椒鸟
    "Phoenicurus frontalis":       "Phoenicuropsis frontalis",     # 蓝额红尾鸲
    "Phoenicurus fuliginosus":     "Rhyacornis fuliginosa",        # 红尾水鸲
    "Phoenicurus leucocephalus":   "Chaimarrornis leucocephalus",  # 白顶溪鸲
    "Phoenicurus schisticeps":     "Phoenicuropsis schisticeps",   # 白喉红尾鸲
    "Phylloscopus castaniceps":    "Seicercus castaniceps",        # 栗头鹟莺
    "Pica serica":                 "Pica pica",                    # 喜鹊
    "Pterorhinus davidi":          "Garrulax davidi",              # 山噪鹛
    "Pterorhinus lanceolatus":     "Babax lanceolatus",            # 矛纹草鹛
    "Pterorhinus pectoralis":      "Garrulax pectoralis",          # 黑领噪鹛
    "Pterorhinus perspicillatus":  "Garrulax perspicillatus",      # 黑脸噪鹛
    "Pterorhinus sannio":          "Garrulax sannio",              # 白颊噪鹛
    "Rubigula flaviventris":       "Pycnonotus melanicterus",      # 黑冠黄鹎
    "Spilopelia chinensis":        "Streptopelia chinensis",       # 珠颈斑鸠
    "Yungipicus canicapillus":     "Dendrocopos canicapillus",     # 星头啄木鸟
    "Yungipicus kizuki":           "Dendrocopos kizuki",           # 小星头啄木鸟
    "Zosterops simplex":           "Zosterops japonicus",          # 暗绿绣眼鸟
    # --- 从名录内亲本种拆出的种 ---
    "Alcippe hueti":               "Alcippe morrisonia",           # 淡眉雀鹛 <- 灰眶雀鹛
    "Nucifraga hemispila":         "Nucifraga caryocatactes",      # 南方星鸦 <- 星鸦
    "Pica bottanensis":            "Pica pica",                    # 青藏喜鹊 <- 喜鹊
    "Saxicola stejnegeri":         "Saxicola maurus",              # 东亚石䳭 <- 黑喉石䳭
    "Staphida torqueola":          "Yuhina castaniceps",           # 栗颈凤鹛 <- 栗耳凤鹛
}

# 骨架里**两份名录都查不到**的种，不做任何猜测映射。UI 显示「暂缺」。
KNOWN_ABSENT = {
    "Anthus japonicus":     "黄腹鹨：两份名录里的「黄腹鹨」都是 Anthus rubescens，同名不同种，不映射",
    "Cairina moschata":     "疣鼻栖鸭：外来/家禽，名录未收",
    "Cygnus atratus":       "黑天鹅：澳洲引入种，名录未收",
    "Corvus corone":        "小嘴乌鸦：2023 版三有名录未收（鸦属只收秃鼻乌鸦、渡鸦、达乌里寒鸦、白颈鸦）",
    "Corvus macrorhynchos": "大嘴乌鸦：2023 版三有名录未收（同上）",
}

# IUCN 代码白名单。NR/NE 都要留：NR = 名录里表示「无危/未评估」，
# 早期把它当脏值过滤掉会导致整行被丢弃、明明查得到的种显示「暂缺」。
IUCN_CODES = {"CR", "EN", "VU", "NT", "LC", "DD", "NR", "NE"}

# 骨架自身的学名问题：这里只记录 + 报警，不改骨架。
TAXONOMY_NOTES = {
    "Parus cinereus": ("骨架标成「大山雀」，但名录里 Parus cinereus 是苍背山雀(Cinereous Tit)，"
                       "大山雀应为 Parus minor。两者 iucn/cn 显示结果相同（NR / 未列入），"
                       "故按精确名照收，但骨架学名应另行修正。"),
}


def norm(sci: str) -> str:
    s = re.sub(r"\[[^\]]*\]", "", sci.strip().lower())
    return re.sub(r"\s+", " ", s).strip(" ,;")


def load_source():
    """学名(归一) -> (中文名, iucn, 行政保护级别)。"""
    out = {}
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            sci = row[1].strip()
            if not sci or " " not in sci:      # 只有拉丁双名才是物种行
                continue
            out[norm(sci)] = (row[2].strip(), row[4].strip().upper(), row[5].strip())
    return out


def load_sanyou():
    """三有名录鸟纲：学名(归一) -> 中文名。"""
    out = {}
    with SANYOU_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[norm(row["学名"])] = row["中文名"]
    return out


def load_skeleton():
    """[(学名, 中文名)]，骨架 = 全国 v2 模型 labels（384 行）。"""
    rows = []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        rows.append((p[0].strip(), p[1].strip() if len(p) > 1 else ""))
    return rows


def load_legacy_iucn():
    """现有 guide_*.json 里的 iucn，用来和新数据交叉校验。"""
    legacy = {}
    for f in sorted(GUIDE_DIR.glob("guide_*.json")):
        for obj in json.loads(f.read_text(encoding="utf-8")):
            code = (obj.get("iucn") or "").strip().upper()
            if code:
                legacy.setdefault(obj["sci"], code)
    return legacy


def main():
    src = load_source()
    sanyou = load_sanyou()
    skeleton = load_skeleton()
    legacy = load_legacy_iucn()

    species, missing, conflicts = {}, [], []
    n_sanyou = n_both = 0

    for sci, zh in skeleton:
        # ---- ① IUCN + 一级/二级：中国鸟类名录 v10.0
        row = src.get(norm(sci))
        if row is None and sci in ALIASES:
            row = src.get(norm(ALIASES[sci]))
        iucn, cn = "", ""
        if row is not None:
            _, iucn, cn = row
            # 保护级别只认一级/二级；"-" 与空串等价（未列入重点保护名录）
            cn = cn if cn in ("一级", "二级") else ""
            if iucn not in IUCN_CODES:
                iucn = ""

        # ---- ② 三有：2023 版三有名录鸟纲
        # 先查自己的别名表；再退回 ALIASES——有 8 个种（环颈鸻/金眶鸻/黄苇鳽/灰头鸦雀…）
        # 的「重点保护」旧名恰好就是三有名录里的名字，两边共用同一个旧属名。
        is_sanyou = (
            norm(sci) in sanyou
            or (sci in SANYOU_ALIASES and norm(SANYOU_ALIASES[sci]) in sanyou)
            or (sci in ALIASES and norm(ALIASES[sci]) in sanyou)
        )
        if is_sanyou:
            n_sanyou += 1
            if cn:
                n_both += 1          # 应当恒为 0：两名录互斥
            else:
                cn = "三有"

        if row is None and not is_sanyou:
            missing.append((sci, zh))
            continue

        # 只要任一名录里有这一行就收录：LC + 未列入也是有效信息，
        # 详情页要显示「无危 LC / 未列入名录」，不能因为两格都空就当缺数据。
        species[sci] = {"iucn": iucn, "cn": cn}

        old = legacy.get(sci)
        if old and iucn and old != iucn:
            conflicts.append((sci, zh, old, iucn))

    meta = {
        "source": "中国鸟类名录 v10.0 整理表（GitHub makenv/bird-list-en）+ "
                  "国家林业和草原局公告 2023 年第 17 号（三有名录）鸟纲",
        "taxonomy": "中国鸟类名录 v10.0",
        "iucn": "IUCN 红色名录等级",
        "cn": "行政保护级别：一级/二级 =《国家重点保护野生动物名录》(2021 调整版)；"
              "三有 =《有重要生态、科学、社会价值的陆生野生动物名录》(2023)。两名录互斥。",
        "built": date.today().isoformat(),
        "coverage": f"{len(species)}/{len(skeleton)}",
        "note": ("按学名索引；iucn 为等级代码，cn 为 一级/二级/三有/空。"
                 "两份名录都用旧属名，异名映射见 build_protection_asset.py 的 "
                 "ALIASES / SANYOU_ALIASES。授权见 docs/legal/LICENSES.md §D。"),
        "absent": {k: v for k, v in KNOWN_ABSENT.items() if k in {m[0] for m in missing}},
        "taxonomyNotes": TAXONOMY_NOTES,
    }

    OUT.write_text(
        json.dumps({"_meta": meta, "species": species}, ensure_ascii=False,
                   separators=(",", ":")),
        encoding="utf-8",
    )

    n_iucn = sum(1 for v in species.values() if v["iucn"])
    n_cn = sum(1 for v in species.values() if v["cn"])
    n_lv = sum(1 for v in species.values() if v["cn"] in ("一级", "二级"))
    n_sy = sum(1 for v in species.values() if v["cn"] == "三有")
    print(f"写出 {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"  骨架 {len(skeleton)}  收录 {len(species)}  缺 {len(missing)}")
    print(f"  有 iucn {n_iucn}   有行政级别 {n_cn}（重点保护 {n_lv} + 三有 {n_sy}）")
    if n_both:
        print(f"\n⚠️ 三有 ∩ 重点保护 = {n_both} 条，与「两名录互斥」不符，需复核！")
    else:
        print("\n✅ 三有 ∩ 重点保护 = 0（两名录互斥，单字段设计成立）")
    if missing:
        print("\n缺数据（UI 显示「暂缺」）：")
        for sci, zh in missing:
            why = KNOWN_ABSENT.get(sci, "未匹配")
            print(f"  {sci:<28} {zh:<8} {why}")
    if conflicts:
        print(f"\n⚠️ 与 guide_*.json 旧 iucn 冲突 {len(conflicts)} 条：")
        for sci, zh, old, new in conflicts:
            print(f"  {sci:<28} {zh:<8} 旧={old:<4} 新={new}")
    else:
        print("\n✅ 与 guide_*.json 旧 iucn 无冲突")
    for sci, note in TAXONOMY_NOTES.items():
        if sci in species:
            print(f"\n⚠️ 骨架学名存疑 {sci}\n     {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

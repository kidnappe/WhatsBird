# -*- coding: utf-8 -*-
"""生成"移动包"（外包给其他机器用）。

之前打包是手工做的，没有脚本 —— 本脚本把它固定下来，并且**确保带上当天的修复**。

包内容 = 主仓当前的 tools（含全部修复）+ geomodel + 物种清单 + 名字对照表 + 背景文档。
**不含 data/**（外包机器从零下载）。这与既有的 `observebird_dongbei_pack` 一致。

必须在包里一起走的两个新增依赖（缺了会静默退化）：
  * `tools/dataset_traits/species_name_map.json`
      `build_dataset_gbif.py` 的 `name_map()` 会从 `<tools>/dataset_traits/` 读它。
      没有它，名字对不上的种（金翅雀等）又会取不到图。
  * `tools/dataset_traits/species_name_map.csv`
      给人看的同一张表（排查用，可选但建议带）。

用法:
  python tools/make_pack.py --region dongbei
  python tools/make_pack.py --region dongbei --species-csv tools/dataset_traits/species_dongbei_ioc150.csv
  python tools/make_pack.py --region dongbei --no-zip
"""
import argparse
import collections
import csv
import datetime
import json
import os
import shutil
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
DIST = os.path.join(REPO, "dist")
# geomodel 不在仓库里：主开发机放在 E:\code\code tools\geomodel；包内只需 4 个文件
GEOMODEL_SRC = next((d for d in (os.path.join(TOOLS, "geomodel"),
                                 r"E:\code\code tools\geomodel")
                     if os.path.isdir(d)), None)
GEOMODEL_FILES = ["BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx",
                  "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt",
                  "ACCEPTABLE_USE.md", "LICENSE-MODELS.md"]
GEOMODEL_LABELS = (os.path.join(GEOMODEL_SRC or "", GEOMODEL_FILES[1]))
# 从主仓拷进包里的脚本（照既有包的集合）
PY_FILES = ["audit_taxon_keys.py", "build_dataset_gbif.py", "clean_dataset.py",
            "eval_report.py", "fetch_inat_region.py",
            "dashboard/monitor_train.py", "dashboard/train_dashboard.py",
            "pick_region_species.py", "pipeline_region.py", "train.py"]
ROOT_FILES = ["AGENTS.md", "HANDOFF.md", "LICENSES.md", "PLAN.md",
              "README-新机器.md", "requirements.txt", "start_dashboard.bat"]


def log(m):
    print(m, flush=True)


def add_data_to_pack(pack, region, data_dir, new_list):
    """把主开发机已有的数据**过滤后**装进包（`<pack>/data/<region>/`）。

    两件事都必须做对，否则会毁掉外包机器的跑：

    1. **必须按新名单过滤**
       `dataset.csv` 的 `scientific_name` 决定机器的类别表。不过滤的话，那些掉出
       新名单的种会变成**额外的类** → 模型类别数和清单对不上。

    2. **必须排除状态文件**（尤其是 `progress.json`）
       `pipeline_region.py:164` 判定 `progress.json.stage == "done"` 就**整体跳过下载**。
       带上它 → 新名单里那 17 个没数据的种**永远下不到**。
       同理排除 `gbif_taxon_audit.csv`（旧名单的，带上会跳过 audit）和 `.clean_done`。
    """
    keep = {r["scientific_name"].strip() for r in new_list}
    src_ds = os.path.join(data_dir, "dataset.csv")
    if not os.path.exists(src_ds):
        sys.exit(f"找不到 {src_ds}")
    src_img = os.path.join(data_dir, "images")
    dst = os.path.join(pack, "data", region)
    dst_img = os.path.join(dst, "images")
    os.makedirs(dst_img, exist_ok=True)

    # ---- dataset.csv 过滤 + 拷图 ----
    rows, dropped_sp, missing_img = [], collections.Counter(), []
    with open(src_ds, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        for r in rd:
            s = (r.get("scientific_name") or "").strip()
            if s not in keep:
                dropped_sp[s] += 1
                continue
            rows.append(r)
    for r in rows:
        fn = r.get("file")
        sp = os.path.join(src_img, fn) if fn else None
        if sp and os.path.exists(sp):
            shutil.copy2(sp, os.path.join(dst_img, fn))
        else:
            missing_img.append(fn)
    if missing_img:
        sys.exit(f"⚠️ {len(missing_img)} 行引用的图片在源目录里不存在（数据已不一致），"
                 f"先修数据再打包。例: {missing_img[:5]}")
    with open(os.path.join(dst, "dataset.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # ---- 有用的状态：预取缓存 + 可用量表（都过滤到新名单）----
    cp = os.path.join(data_dir, "gbif_prefetch_cache.json")
    if os.path.exists(cp):
        cache = json.load(open(cp, encoding="utf-8"))
        kept = {k: v for k, v in cache.items() if k in keep}
        json.dump(kept, open(os.path.join(dst, "gbif_prefetch_cache.json"),
                             "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ap_ = os.path.join(data_dir, "gbif_availability.csv")
    if os.path.exists(ap_):
        with open(ap_, encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            af, arows = rd.fieldnames, [r for r in rd
                                        if (r.get("scientific_name") or "").strip() in keep]
        with open(os.path.join(dst, "gbif_availability.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=af)
            w.writeheader(); w.writerows(arows)

    # ---- 断言：绝不能带的状态文件 ----
    for bad in ("progress.json", "gbif_taxon_audit.csv", ".clean_done",
                "fetch_target_inat.json"):
        if os.path.exists(os.path.join(dst, bad)):
            sys.exit(f"❌ 包里不该有 {bad} —— 会让管线跳过对应阶段")

    n_img = len(os.listdir(dst_img))
    if n_img != len(rows):
        sys.exit(f"⚠️ 行数 {len(rows)} ≠ 图片数 {n_img}（数据不一致）")
    size = sum(os.path.getsize(os.path.join(dst_img, f))
               for f in os.listdir(dst_img))
    log(f"已有数据装入包内 data/{region}/：")
    log(f"  dataset.csv {len(rows):,} 行（过滤掉 {sum(dropped_sp.values()):,} 行 / "
        f"{len(dropped_sp)} 个掉出种）")
    log(f"  images/     {n_img:,} 张  {size/1024/1024:.0f} MB")
    log(f"  状态文件    预取缓存 + 可用量表（已过滤）")
    log(f"  已排除      progress.json / audit / .clean_done / fetch_target_inat.json"
        f" / 各类日志 / reserve.csv")
    n_sp = len({r["scientific_name"] for r in rows})
    log(f"  → 外包机器会跳过这 {n_sp} 个已取尽的种，只下新名单里缺的那些")
    return len(rows), n_img


def build_existing_manifest(pack, data_dir, new_list, with_data=False):
    """写「主开发机已有数据清单」到包里。**只出清单，不带图片。**

    为什么要它：外包机器从零跑，但主开发机其实已经下过一批。清单让两边都知道
    ① 已经有什么 ② 这次该重点补什么（名字对不上、之前 0 张的那些种）。
    """
    import collections
    have = collections.Counter()
    ds = os.path.join(data_dir, "dataset.csv")
    if os.path.exists(ds):
        with open(ds, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                s = (r.get("scientific_name") or "").strip()
                if s:
                    have[s] += 1
    new_map = {r["scientific_name"].strip(): r for r in new_list}
    # 之前因为名字对不上而 0 张的种（本次应能取到）—— 从对照表的 resolved_by 认
    namefix = {"Chloris sinica", "Thinornis dubius", "Botaurus sinensis",
               "Ardea coromanda", "Butorides striata", "Anthus japonicus",
               "Lophospiza trivirgata"}
    both = {s: n for s, n in have.items() if s in new_map}
    only_new = [s for s in new_map if have.get(s, 0) == 0]
    only_old = {s: n for s, n in have.items() if s not in new_map}

    # 逐种 CSV
    csv_path = os.path.join(pack, "tools", "species",
                            "existing_data_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scientific_name", "common_name", "in_new_list",
                    "images_on_main_machine", "note"])
        for s in sorted(set(list(new_map) + list(have))):
            zh = (new_map.get(s) or {}).get("common_name", "")
            n = have.get(s, 0)
            note = []
            if s not in new_map:
                note.append("掉出新名单")
            if n == 0 and s in new_map:
                note.append("本次要下的重点")
            if s in namefix:
                note.append("此前因名字对不上取不到图，本版包已修")
            w.writerow([s, zh, "Y" if s in new_map else "N", n, "；".join(note)])

    # 人读的 md
    md = os.path.join(pack, "已有数据清单.md")
    L = []
    L.append("# 东北 · 主开发机已有数据清单\n")
    if with_data:
        L.append("> **这批数据已经装进包里**（`data/dongbei/`），你**不需要**重下它们 ——\n"
                 "> 下载阶段会自动跳过（判据：`已有张数 ≥ min(300, 框内 GBIF 可用量)`）。\n"
                 "> **不要删 `data/dongbei/dataset.csv`，也不要找 `progress.json`**（故意没带，\n"
                 "> 带了会让管线整体跳过下载 → 下面那批还没数据的种就永远下不到）。\n")
    else:
        L.append("> **本清单只是记录，图片没有随包发出。** 外包机器仍需完整跑一遍链条。\n")
    L.append("> 用途：① 知道主开发机已经下过什么 ② 跑完后对照，确认该补的补上了。\n")
    L.append(f"\n生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    L.append("## 主开发机已有\n")
    L.append(f"- 数据：**{sum(have.values()):,} 张 / {len(have)} 个种**"
             f"（GBIF 阶段，2026-10-05 16:56 完成）")
    L.append("- iNat 补量：目标 19,547 张，**未完成**（那轮 17:05 因 URLError 中断）")
    L.append("- 模型：**一次都没训过**（无 `models/` 目录）")
    L.append(f"\n## 与本次新名单（IOC v15.2，{len(new_map)} 种）的关系\n")
    L.append("| | 种数 | 张数 |")
    L.append("|---|---|---|")
    L.append(f"| 新名单里、本机已有数据 | **{len(both)}** | {sum(both.values()):,} |")
    L.append(f"| 新名单里、本机没有数据（**外包机器要下的重点**） | **{len(only_new)}** | 0 |")
    L.append(f"| 本机有、但**掉出**新名单 | {len(only_old)} | {sum(only_old.values())} |")
    L.append("\n### 新名单里本机没有数据的（重点）\n")
    L.append("| 学名 | 中文 | 备注 |")
    L.append("|---|---|---|")
    for s in only_new:
        zh = new_map[s].get("common_name", "")
        note = "此前因名字对不上取不到图，本版包已修" if s in namefix else ""
        L.append(f"| `{s}` | {zh} | {note} |")
    L.append("\n### 掉出新名单的（图留在主开发机，已登记 `reserve.csv`）\n")
    L.append("| 学名 | 张数 |")
    L.append("|---|---|")
    for s, n in sorted(only_old.items(), key=lambda x: -x[1]):
        L.append(f"| `{s}` | {n} |")
    L.append(f"\n逐种明细见 `tools/species/existing_data_manifest.csv`。")
    if with_data:
        L.append(f"\n> **已装进包**：`data/dongbei/`（dataset.csv + images/ + 预取缓存）。"
                 f"共 {len(both)} 个种 / {sum(both.values()):,} 张。\n"
                 f"> 未随包的：`progress.json`（故意排除）/ `gbif_taxon_audit.csv` / `.clean_done`"
                 f" / 各类日志 / 主开发机的 `reserve.csv`。")
    else:
        L.append(f"\n> 主开发机侧同步登记了候补池："
                 f"`<主开发机>/dongbei/data/dongbei/reserve.csv`"
                 f"（{len(only_old)} 种 / {sum(only_old.values())} 张，**图片未删**）。")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return md


def load_geomodel_index():
    idx = {}
    for i, l in enumerate(open(GEOMODEL_LABELS, encoding="utf-8",
                               errors="replace").read().splitlines()):
        p = l.split("\t")
        if len(p) >= 2:
            idx.setdefault(p[1].strip().lower(), i)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--species-csv", default=None,
                    help="源清单。默认用 dist 里现有包的同名清单；"
                         "给 IOC 版则会自动转成下载器要的列格式")
    ap.add_argument("--pack-name", default=None)
    ap.add_argument("--existing-data", default=None,
                    help="主开发机已有的数据目录（含 dataset.csv）。给了就在包里生成"
                         "「已有数据清单」——**只出清单，不带图片**")
    ap.add_argument("--with-data", action="store_true",
                    help="把 --existing-data 的数据**过滤后装进包**（data/<region>/）。"
                         "外包机器会自动跳过已取尽的种，省掉重下的几个小时")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    pack_name = a.pack_name or f"observebird_{a.region}_pack"
    pack = os.path.join(DIST, pack_name)
    src_pack = os.path.join(DIST, f"observebird_{a.region}_pack")

    # ---- 源清单 ----
    src_csv = a.species_csv
    if not src_csv:
        src_csv = os.path.join(src_pack, "tools", "species",
                               f"species_{a.region}_150.csv")
    if not os.path.exists(src_csv):
        sys.exit(f"找不到源清单: {src_csv}")
    log(f"源清单: {src_csv}")

    # ---- 准备目录 ----
    if os.path.exists(pack):
        log(f"清空旧包: {pack}")
        shutil.rmtree(pack)
    for sub in ("tools/geomodel", "tools/species", "tools/dataset_traits"):
        os.makedirs(os.path.join(pack, sub), exist_ok=True)

    # ---- 工具脚本（含当天修复）----
    n = 0
    for f in PY_FILES:
        p = os.path.join(TOOLS, f)
        if not os.path.exists(p):
            sys.exit(f"缺脚本: {p}")
        dst = os.path.join(pack, "tools", f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)  # dashboard/ 子目录
        shutil.copy2(p, dst); n += 1
    for f in ("regions.json", "run_detached.ps1"):
        shutil.copy2(os.path.join(TOOLS, f), os.path.join(pack, "tools", f)); n += 1
    log(f"tools/ 拷入 {n} 个脚本")

    # ---- geomodel + 373 基准名单 ----
    if not GEOMODEL_SRC:
        sys.exit("找不到 geomodel 目录（tools/geomodel 或 E:\\code\\code tools\\geomodel）")
    for f in GEOMODEL_FILES:
        s = os.path.join(GEOMODEL_SRC, f)
        if not os.path.exists(s):
            sys.exit(f"geomodel 缺文件: {s}")
        shutil.copy2(s, os.path.join(pack, "tools", "geomodel", f))
    log(f"geomodel/ 拷入 {len(GEOMODEL_FILES)} 个文件 <- {GEOMODEL_SRC}")
    shutil.copy2(os.path.join(src_pack, "tools", "species", "species_list_373.csv"),
                 os.path.join(pack, "tools", "species", "species_list_373.csv"))
    log("geomodel/ + species_list_373.csv 拷入")

    # ---- 名字对照表（**必须带**，否则下载器静默退化）----
    dt = os.path.join(TOOLS, "dataset_traits")
    for f in ("species_name_map.json", "species_name_map.csv", "name_overrides.csv"):
        p = os.path.join(dt, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(pack, "tools", "dataset_traits", f))
    log(f"dataset_traits/ 名字对照表拷入（build_dataset_gbif 依赖它）")

    # ---- 物种清单：转成下载器要的列 ----
    # ⚠️ `birdnet_index` 的项目口径 = **该种在 species_list_373.csv 里的行号**（作者名单的种号），
    #    不是 geomodel 标签索引。旧包的 271/249 正好是 喜鹊/麻雀 在 373 名单里的行号。
    #    填错不会崩（没有代码拿它当下标），但语义不一致会误导后人，所以按口径来。
    #    `count_china` 也一并从 373 名单取。
    pool_idx, pool_cnt = {}, {}
    pool_csv = os.path.join(src_pack, "tools", "species", "species_list_373.csv")
    if os.path.exists(pool_csv):
        for i, r in enumerate(csv.DictReader(open(pool_csv, encoding="utf-8-sig"))):
            s = (r.get("scientific_name") or "").strip()
            if s:
                pool_idx[s] = i
                pool_cnt[s] = r.get("count_china", "")
    gm = load_geomodel_index()
    # count_china：373 池子里没有的种（= 从 IOC/traits 新进来的），用对照表的
    # `gbif_cn_records` 填上 —— 空列看着像 bug，而且这个数更有用
    cn_from_map = {}
    mpath = os.path.join(dt, "species_name_map.csv")
    if os.path.exists(mpath):
        for r in csv.DictReader(open(mpath, encoding="utf-8-sig")):
            v = (r.get("gbif_cn_records") or "").strip()
            if v:
                cn_from_map[r["query_name"].strip()] = v
    rows = list(csv.DictReader(open(src_csv, encoding="utf-8-sig")))
    cols = set(rows[0].keys())
    out = []
    for r in rows:
        sci = (r.get("ioc_name") or r.get("scientific_name") or "").strip()
        if not sci:
            continue
        out.append({"birdnet_index": pool_idx.get(sci, ""),
                    "scientific_name": sci,
                    "common_name": (r.get("common_name") or r.get("zh") or "").strip(),
                    "geomodel_mean": r.get("geomodel_mean", ""),
                    "count_china": (r.get("count_china") or pool_cnt.get(sci)
                                    or cn_from_map.get(sci, ""))})
    nopool = [r["scientific_name"] for r in out if r["birdnet_index"] == ""]
    nogm = [r["scientific_name"] for r in out if r["scientific_name"].lower() not in gm]
    sp_out = os.path.join(pack, "tools", "species", f"species_{a.region}_150.csv")
    # （清单文件在校验通过后统一写出，见下面 —— 这样能把校验时补入的种也填全）
    if nopool:
        log(f"  （{len(nopool)} 个不在 373 池子里 -> birdnet_index 留空: {nopool[:6]}"
            f"{' …' if len(nopool) > 6 else ''}）")
    if nogm:
        log(f"  ⚠️ {len(nogm)} 个不在 geomodel 标签里（清单里能用，但重排时会被跳过）: {nogm}")

    # ---- 校验：清单里**每个**种都必须能解析出 GBIF taxonKey ----
    # ⚠️ 不做这一步就会静默丢数据：下载器遇到解析不出的种直接 0 张图、只留一行日志。
    #    实测 `Astur gentilis`（苍鹰，IOC 新属名）GBIF 只给到属级 → 会被拒；
    #    而对照表里当时只有 `Accipiter gentilis`（旧属名）。这类缺口必须在发包前挡住。
    sys.path.insert(0, dt)
    import build_name_map as bnm
    nm_path = os.path.join(pack, "tools", "dataset_traits", "species_name_map.json")
    nm = json.load(open(nm_path, encoding="utf-8"))
    _names, zh_index, zh_list = bnm.load_inputs()
    added, failed = [], []
    for r in out:
        s = r["scientific_name"].strip()
        if s in nm:
            continue
        try:
            res = bnm.resolve(s, (r["common_name"] or "").strip(), zh_index, zh_list)
        except Exception as e:
            res = {"gbif_key": None, "note": f"{type(e).__name__}: {e}"}
        if res.get("gbif_key"):
            nm[s] = {"gbif_name": res["gbif_name"], "gbif_key": res["gbif_key"],
                     "zh": (r["common_name"] or "").strip(),
                     "geomodel_index": gm.get(s.lower(), ""),
                     "cn_records": res.get("cn_records")}
            added.append((s, res["gbif_name"], res["gbif_key"], res.get("resolved_by")))
        else:
            failed.append((s, res.get("note") or res.get("candidates") or "无候选"))
    if added:
        json.dump(nm, open(nm_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        log(f"对照表补入 {len(added)} 条（清单里有、原表没有的）:")
        for s, g, k, how in added:
            log(f"    {s} -> {g}  key={k}  ({how})")
    if failed:
        log(f"\n❌ {len(failed)} 个种解析不出 GBIF taxonKey —— **不要发这个包**，"
            f"先补 name_overrides.csv 或换名字：")
        for s, why in failed:
            log(f"    {s}: {why}")
        sys.exit(2)
    # 补 count_china：373 池子和仓库 CSV 都没有的，用刚才解析出的中国记录数
    for r in out:
        if not r["count_china"]:
            e = nm.get(r["scientific_name"].strip()) or {}
            if e.get("cn_records"):
                r["count_china"] = e["cn_records"]
    with open(sp_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["birdnet_index", "scientific_name",
                                          "common_name", "geomodel_mean", "count_china"])
        w.writeheader(); w.writerows(out)
    log(f"物种清单 {len(out)} 行 -> {os.path.relpath(sp_out, REPO)}")
    log(f"解析校验通过：清单 {len(out)} 个种全部有 GBIF key（对照表 {len(nm)} 条）")

    # ---- 已有数据清单（+ 可选：把数据装进包）----
    if a.existing_data:
        if a.with_data:
            add_data_to_pack(pack, a.region, a.existing_data, out)
        else:
            log("（未给 --with-data，图片**没有**装进包）")
        mf = build_existing_manifest(pack, a.existing_data, out, a.with_data)
        log(f"已有数据清单 -> {os.path.relpath(mf, REPO)}")
    else:
        log("（未给 --existing-data，跳过已有数据清单）")

    # ---- 背景文档（沿用既有包的那份，缺则从主仓拷）----
    for f in ROOT_FILES:
        s = os.path.join(src_pack, f)
        if not os.path.exists(s):
            # 2026-10-06 起 bat 等启动器搬进 tools/dashboard/（工具+启动器同仓），
            # 2026-10-07 起主仓文档归档进 docs/（见 docs/README.md）。
            # 旧包根仍第一优先；仓库里按 docs/ 新位置 → tools/ 的顺序兜底
            for base in (REPO,
                         os.path.join(REPO, "docs", "handoff", "root"),
                         os.path.join(REPO, "docs", "legal"),
                         os.path.join(REPO, "docs", "project"),
                         os.path.join(REPO, "tools", "dashboard"),
                         os.path.join(REPO, "tools")):
                s = os.path.join(base, f)
                if os.path.exists(s):
                    break
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(pack, f))
    log(f"背景文档 {len(ROOT_FILES)} 个拷入")

    # ---- 给外包 agent 的补充说明（patch 进 AGENTS.md）----
    ag = os.path.join(pack, "AGENTS.md")
    if os.path.exists(ag):
        txt = open(ag, encoding="utf-8").read()
        s8 = ("记录**主开发机**已经下过的数据（本次是 5,500 张 / 143 种）。\n"
              "   **这批数据的图片已经装进包里了**（见下面第 9 条），你不需要重下。"
              if a.with_data else
              "记录**主开发机**已经下过的数据（本次是 5,500 张 / 143 种）。\n"
              "   **图片没有随包发出**，你仍要完整跑一遍链条。")
        patch = f"""
## 7. 本版包的变更（2026-10-05，**必读**）

本包的脚本已包含当天的一批修复，外包机器上**不要**用旧版本覆盖：

1. **名字对照表随包提供** —— `tools/dataset_traits/species_name_map.json`（1,494 条）。
   `build_dataset_gbif.py` 会优先查它拿 GBIF taxonKey。**不要删**；没有它，名字对不上的种
   （金翅雀 `Chloris sinica`、金眶鸻 `Thinornis dubius`、黄苇鳽 `Botaurus sinensis`、
   牛背鹭 `Ardea coromanda` 等）会再次取不到图。
2. **`--per-species` 语义变了**：现在是「该种在 dataset.csv 里的**目标张数**」，不是
   「本轮再拿多少」。已有 ≥ 目标的种会被**瞬间跳过**。所以重跑很快、也不会重复下已下的图。
3. **审计遇到网络失败写 `rank=FETCH_FAILED`**（不再写 `NONE`），
   而 `clean_dataset.py` 对 `FETCH_FAILED` 的种**一律保留行**、occurrence 查询失败也保留。
   > 背景：2026-10-05 华南因为一次网络抖动被记成 `rank=NONE`，`clean_dataset.py` 据此删掉了
   > 6 个种的行（4,138 条可用记录），而 1,561 张图片还好端端留在 `images/` 里。
   > **网络失败绝不允许变成删数据。**
4. **日志行格式**：`[ 64/150] <学名> <中文> 用N 累计R ETA X分 补登S (rows/goal 张)`。
   `累计` 现在是"本轮实际写入 dataset.csv 的行数"（含补登记）。ETA 是**张数口径**；
   算不出速率时会写 `ETA --`（不猜），此时监控台用它自己的口径兜底。
5. **`run_detached.ps1` 必须是 UTF-8 带 BOM**（否则 PowerShell 按 GBK 读 → 引号错位 →
   整个脚本解析失败）。同时它已给子进程设 `PYTHONIOENCODING=utf-8`（本机 Python 默认 cp936，
   不设的话日志中文是乱码）。
6. **物种清单换成 IOC 口径**（IOC World Bird List v15.2，2026-04-13）。
   与 geomodel 的分类学口径一致；此前 4 个取不到图的种现在能取到了。
   `tools/species/species_dongbei_150.csv` 就是新清单，**沿用**即可、不要重新生成。
7. **依赖**：`requirements.txt` 已含 `onnxruntime`（geomodel 推理用）。
8. **`已有数据清单.md`（只在给了 `--existing-data` 时存在）**：{s8}
   用法：
   - 跑之前看一眼——里面列了「新名单里主开发机没有数据的 17 个种」，那是重点；
     其中 4 个（金翅雀/金眶鸻/黄苇鳽/牛背鹭）此前因名字对不上取不到图，**本版包已修**
   - 跑完之后**对照检查**：这 17 个种在 `dataset.csv` 里是否都有行了。
     尤其那 4 个：如果你看到它们仍然是 0 张，说明名字对照表没生效（见上面第 1 条），**报告**。
   逐种明细在 `tools/species/existing_data_manifest.csv`。
9. **本包可能已自带数据**（`data/dongbei/` 里已有 `dataset.csv` + `images/`）。
   如果存在，说明主开发机已经把一部分种下好了，**这是有意为之**：
   - **不要删 `data/dongbei/dataset.csv`，也不要手工编辑它**
   - **不要找 `progress.json`** —— 它被**故意排除**了。有它的话 `pipeline_region.py`
     会判定"下载已完成"而**整体跳过下载**，那么新名单里那批还没数据的种就永远下不到。
     没有它 → 下载阶段正常跑，但已取尽的种会被**自动跳过**（判据：
     `已有张数 ≥ min(300, 框内 GBIF 可用量)`），所以只会补真正缺的
   - 同理 `gbif_taxon_audit.csv` 和 `.clean_done` 也被排除了 → audit 和清洗**会重跑**，这是对的
   - `gbif_prefetch_cache.json` 和 `gbif_availability.csv` **带着**，能省掉预取探测量时间；
     新名单里没有缓存的种会被现场查询
   - **预期现象**：下载日志开头会有一大批
     `[  N/150] <学名> 已有 M 张 ≥ 目标 X（GBIF 已取尽（框内可用 X）），跳过`。
     这不是出错，是省时间。真正要下的是「已有数据清单.md」里列的那批种。
"""
        if "本版包的变更" not in txt:
            open(ag, "w", encoding="utf-8").write(txt.rstrip() + "\n" + patch)
            log("AGENTS.md 追加了「本版包的变更」小节")

    # ---- 清掉 __pycache__ ----
    for root, dirs, files in os.walk(pack):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d)); dirs.remove(d)

    # ---- 统计 ----
    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(pack) for f in fs)
    log(f"\n包目录: {os.path.relpath(pack, REPO)}   {tot/1024/1024:.1f} MB")

    # ---- 打 zip ----
    if not a.no_zip:
        stamp = datetime.datetime.now().strftime("%Y%m%d")
        zp = os.path.join(DIST, f"{pack_name}_{stamp}.zip")
        if os.path.exists(zp):
            os.remove(zp)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for r, _, fs in os.walk(pack):
                for f in fs:
                    full = os.path.join(r, f)
                    z.write(full, os.path.join(pack_name,
                                               os.path.relpath(full, pack)))
        log(f"zip: {os.path.relpath(zp, REPO)}   {os.path.getsize(zp)/1024/1024:.1f} MB")
    log("\n完成。" + ("含 data/（外包机器会自动跳过已取尽的种）。"
                     if a.with_data else "不含 data/（外包机器从零下载）。"))


if __name__ == "__main__":
    main()

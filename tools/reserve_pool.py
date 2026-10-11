# -*- coding: utf-8 -*-
"""候补池（reserve）：种掉出名单后，图**不删**，只登记进 <data>/reserve/<region>.csv，
供别的区借。图片留在原地 `data/<region>/images/`，登记行本身就是"这张图存在、它属于哪个种"。

为什么要它（HANDOFF-20261005-EXT §1-③）：扩池推演显示每区只会换 1~11 个名额，短期不掉种；
但"重挑名单 → 类别编号整体位移"这条链一旦走，没有候补池就会有人手删图。删掉的图是**不可再生**的
（稀有种的 GBIF 记录本来就没几张）。

用法：
    # 常规巡检：按各区当前名单找出"有图但不在名单"的种（今天实测 0 条）
    python tools/reserve_pool.py register --all

    # 重挑名单之后：拿新名单当准，把掉出的种登记进池
    python tools/reserve_pool.py register --region huabei --against tools/dataset_traits/species_huabei_new150.csv

    python tools/reserve_pool.py list              # 池子现状
    python tools/reserve_pool.py forget --region huabei --species "Ardea coromanda"   # 种回到名单，撤登记（图不动）

消费端：`tools/borrow_cross_region.py` 默认把 <data>/reserve/*.csv 一起当借出池。
"""
import argparse
import csv
import os
import time
from collections import defaultdict

# 名单口径与 tools/dataset_traits/pool_expand_diff.py::old_list 保持一致（那里是三级兜底的
# 原始定义，HANDOFF-20261005-EXT §4.2 记过只试第一级导致"旧名单=0"的坑）。这里内联是为了让
# reserve_pool 不拖进 numpy/pick_region_species 那条重依赖链——改兜底顺序时两处都要动。
DATA_ROOT = r"E:\code\code tools\data"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def old_list(region, data_root=None):
    """该区当前实际在用的名单，依次尝试；都没有返回 None。

    `data_root` 必须跟着 `--data-root` 走 —— 外包区（东北）的数据不在
    `E:\\code\\code tools\\data` 下，只改 dataset.csv 的路径而漏了这里，
    会让不带 `--against` 的调用走空兜底。

    ⚠️ **兜底顺序有讲究**：`dist/observebird_<r>_pack_ioc/`（IOC 新名单）必须排在
    `dist/observebird_<r>_pack/`（旧名单）**前面**。2026-10-06 实测踩过：
    顺序反了 → `register --all` 拿东北**旧包**的名单当准绳 → 那 10 个掉出种
    看起来"还在名单里" → `cmd_register` 把它们**全部撤登记**，129 行池子被清空。
    """
    root = data_root or DATA_ROOT
    for p in (os.path.join(root, region, f"species_{region}_150.csv"),
              os.path.join(root, region, "gbif_availability.csv"),
              os.path.join(REPO, "dist", f"observebird_{region}_pack_ioc",
                           "tools", "species", f"species_{region}_150.csv"),
              os.path.join(REPO, "dist", f"observebird_{region}_pack",
                           "tools", "species", f"species_{region}_150.csv")):
        if not os.path.exists(p):
            continue
        try:
            out = [r["scientific_name"].strip()
                   for r in csv.DictReader(open(p, encoding="utf-8-sig", newline=""))
                   if r.get("scientific_name")]
            if out:
                if os.path.join(REPO, "dist") in p:
                    print(f"  ⚠️ {region}: 在册名单取自包内快照 {os.path.relpath(p, REPO)}"
                          f" —— 若该包已过时，请用 --against 指定权威名单")
                return out
        except Exception:
            continue
    return None


RESERVE_COLS = ["region", "file", "scientific_name", "common_name", "gbif_key",
                "occurrence_key", "lat", "lon", "year", "month", "license",
                "dataset_key", "registered_at", "reason"]
REGIONS = ["huabei", "huadong", "huazhong", "huanan_sw", "dongbei"]
IMG_ROOTS = [DATA_ROOT]  # 候补池只登记不搬图，图片仍按区留在 data/<region>/images/


def reserve_path(region, reserve_dir):
    return os.path.join(reserve_dir, f"{region}.csv")


def resolve_image(row, reserve_dir=None):
    """池行 -> 图片的实际路径（图不搬，所以只能按 region 名回原地找）。

    先试池目录的同级结构（<dir>/<region>/images/<file>，测试/自包含场景），
    再回落到项目的 DATA_ROOT。都找不到返回 None —— 调用方跳过，不报错。
    """
    cands = []
    if reserve_dir:
        cands.append(os.path.join(os.path.dirname(os.path.normpath(reserve_dir)),
                                  row.get("region", ""), "images", row.get("file", "")))
    cands += [os.path.join(root, row.get("region", ""), "images", row.get("file", ""))
              for root in IMG_ROOTS]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def read_reserve(path):
    """返回 {(region, file): row}；文件不存在给空。

    池目录里可能混进别人的 csv（测试名单、临时文件），没有 region 列就不是池文件 —— 跳过并说一声，
    别让巡检在别人的文件上崩掉。
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "region" not in reader.fieldnames:
            print(f"  跳过非池文件（缺 region 列）: {path}")
            return {}
        return {(r["region"], r["file"]): r for r in reader}


def write_reserve(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(RESERVE_COLS)
        for key in sorted(rows):
            r = rows[key]
            w.writerow([r.get(c, "") for c in RESERVE_COLS])
    os.replace(tmp, path)  # 原子替换：池子被写坏比不写更糟


def read_list(path):
    """外部名单 CSV 的学名列。

    兼容两种列名（`species_<r>_150.csv` / `gbif_availability.csv` 用 `scientific_name`；
    `tools/dataset_traits/species_<r>_ioc150.csv` 用 `ioc_name`）。

    ⚠️ 认不出列名时**必须报错**，不能静默返回空集 —— 空集会让"名单里所有种"都判为掉出。
    2026-10-06 实测踩过：`--against` 传了 `ioc_name` 列的 IOC 名单，
    结果东北 **5,500 行 / 143 个种全被登记成"掉出"**，池子瞬间被灌满。
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        key = next((c for c in ("scientific_name", "ioc_name") if c in cols), None)
        if key is None:
            raise SystemExit(
                f"❌ {path} 里找不到学名列（试过 scientific_name / ioc_name），"
                f"实际列 = {cols}。\n"
                f"   改名或用列名正确的名单文件；**不要**让它退化成空集 —— "
                f"那会把全区都登记成掉出。")
        return {r[key].strip() for r in rd if r.get(key)}


def region_in_list(region, against, data_root=None):
    """该区的在册名单：--against 优先，否则用 old_list 的三级兜底。"""
    if against:
        return read_list(against)
    lst = old_list(region, data_root)
    if lst is None:
        raise LookupError(f"{region}: 找不到名单文件（--against 指定，或 data/<r>/species_<r>_150.csv）")
    return set(lst)


def dropped_rows(dataset_csv, in_list):
    """dataset.csv 里"有图但不在名单"的行。"""
    if not os.path.exists(dataset_csv):
        return []
    region = os.path.basename(os.path.dirname(dataset_csv))
    img_dir = os.path.join(os.path.dirname(dataset_csv), "images")
    out = []
    with open(dataset_csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            sci = (r.get("scientific_name") or "").strip()
            if not sci or sci in in_list:
                continue
            if not os.path.exists(os.path.join(img_dir, r.get("file", ""))):
                continue  # 图不在盘上，登记了也没得借
            row = {c: r.get(c, "") for c in RESERVE_COLS}
            row.update(region=region, registered_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       reason="dropped")
            out.append(row)
    return out


def cmd_register(a):
    regions = REGIONS if a.all else [a.region]
    reserve_dir = a.reserve_dir
    os.makedirs(reserve_dir, exist_ok=True)
    for region in regions:
        in_list = region_in_list(region, a.against, a.data_root)
        ds = os.path.join(a.data_root, region, "dataset.csv")
        if not os.path.exists(ds):
            # 不能说"掉出 0 行" —— 那是"没查"，不是"没掉出"。
            # 外包区（东北）数据在仓库外，必须显式 --data-root。
            print(f"{region:<10} ⚠️ 跳过：找不到 {ds}"
                  + ("（外包区数据在仓库外，加 --data-root 指向它）"
                     if region not in ("huabei", "huadong", "huazhong", "huanan_sw")
                     else ""))
            continue
        rows = dropped_rows(ds, in_list)
        path = reserve_path(region, reserve_dir)
        pool = read_reserve(path)
        before = len(pool)
        for r in rows:
            pool[(r["region"], r["file"])] = r
        # 名单里又回来的种，把它的旧登记撤掉（图仍不动）
        for key in [k for k, v in pool.items() if v["scientific_name"] in in_list]:
            del pool[key]
        if pool or before:
            write_reserve(path, pool)  # 从没登记过且现在也没得登记，就别留空文件
        by_sci = defaultdict(int)
        for v in pool.values():
            by_sci[v["scientific_name"]] += 1
        print(f"{region:<10} 掉出 {len(rows)} 行 -> 池内 {len(pool)} 行"
              f"（{len(by_sci)} 种，原 {before}）")
        if by_sci and a.verbose:
            for sci, n in sorted(by_sci.items(), key=lambda x: -x[1]):
                print(f"    {sci:<28} {n}")


def cmd_list(a):
    if not os.path.isdir(a.reserve_dir):
        print(f"池目录还没有：{a.reserve_dir}")
        return
    total = 0
    for name in sorted(os.listdir(a.reserve_dir)):
        if not name.endswith(".csv"):
            continue
        pool = read_reserve(os.path.join(a.reserve_dir, name))
        by_sci = defaultdict(int)
        for v in pool.values():
            by_sci[v["scientific_name"]] += 1
        total += len(pool)
        print(f"{name[:-4]:<12} {len(pool):>5} 行 / {len(by_sci):>2} 种  "
              f"{'、'.join(sorted(by_sci)[:6])}")
    print(f"合计 {total} 行；图片仍在 data/<region>/images/ 原地未动")


def cmd_forget(a):
    path = reserve_path(a.region, a.reserve_dir)
    pool = read_reserve(path)
    if not pool:
        print(f"{a.region}: 池子是空的，没什么可撤")
        return
    keep = {k: v for k, v in pool.items() if v["scientific_name"] not in set(a.species)}
    write_reserve(path, keep)
    print(f"{a.region}: 撤下 {len(pool) - len(keep)} 行登记（图片留在原地），池内剩 {len(keep)} 行")


def main():
    # --reserve-dir 放在子命令上（而不是顶层），`register --reserve-dir X` 才不用记位置
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--reserve-dir", default=os.path.join(DATA_ROOT, "reserve"))
    common.add_argument("--data-root", default=DATA_ROOT,
                        help="各区数据根目录（默认 %(default)s）。外包区不在标准位置时用，"
                             "例如东北：--data-root 'E:\\code\\observebird\\dongbei\\data'")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("register", parents=[common],
                       help="找出掉出的种并登记（幂等，可反复跑）")
    p.add_argument("--region")
    p.add_argument("--all", action="store_true", help="五个区全巡")
    p.add_argument("--against", default=None, help="新名单 CSV；缺省用该区当前在册名单")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("list", parents=[common], help="池子现状")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("forget", parents=[common], help="种回到名单后撤登记（不碰图片）")
    p.add_argument("--region", required=True)
    p.add_argument("--species", nargs="+", required=True)
    p.set_defaults(func=cmd_forget)

    a = ap.parse_args()
    if a.cmd == "register" and not (a.all or a.region):
        ap.error("register 要加 --region 或 --all")
    a.func(a)


if __name__ == "__main__":
    main()

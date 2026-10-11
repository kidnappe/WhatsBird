# -*- coding: utf-8 -*-
"""图鉴底册（assets/guide/guide_<region>.json）从 labels 反向生成，杜绝手工维护导致的编号错位。

背景（2026-10-05 实测）：
- guide_huazhong.json 停留在旧 145 类时代，模型已是 150 类 → 从 idx=18 起 127/145 条编号错位，
  且新进的 5 个种（金翅雀/牛背鹭/黄苇鳽/黄腹鹨/金眶鸻）根本不在底册里，永远点不亮。
- 华南/西南是 138 类合并模型按分布拆成的两册，idx 不连续是设计如此（保留模型原始索引）。

用法：
    python tools/regen_guide.py --check              # 只校验，不写文件（四区全查）
    python tools/regen_guide.py --region huazhong    # 重生成单模型区域
    python tools/regen_guide.py --all                # 四区全重生成（华南/西南转交 split 脚本）

产物结构保持不变：[{"sci", "zh", "idx", "iucn"}, ...]，idx = labels 行号（0-based，= 模型类别索引）。
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "android", "app", "src", "main", "assets")
RARITY = os.path.join(ASSETS, "guide", "rarity.json")
LABELS_373 = os.path.join(ASSETS, "models", "labels_birdnet373.txt")

# 一个模型对应一册的区域：region -> (labels 文件名, 图鉴文件名)
SINGLE = {
    "huabei": ("labels_huabei.txt", "guide_huabei.json"),
    "huazhong": ("labels_huazhong.txt", "guide_huazhong.json"),
    "huadong": ("labels_huadong.txt", "guide_huadong.json"),
    "dongbei": ("labels_dongbei.txt", "guide_dongbei.json"),
}
# 138 类合并模型拆成的两册，由 split_huanan_sw_guide.py 按分布生成
SPLIT = {
    "huanan": "guide_huanan.json",
    "xinan": "guide_xinan.json",
}
SPLIT_LABELS = "labels_huanan_sw.txt"
SPLIT_SCRIPT = os.path.join(ROOT, "tools", "split_huanan_sw_guide.py")


def read_labels(name):
    """labels 文件：每行 '学名\\t中文名'，行号 = 模型类别索引。返回 [(idx, sci, zh)]。"""
    rows = []
    with open(os.path.join(ASSETS, "models", name), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            zh = parts[1].strip() if len(parts) > 1 else ""
            rows.append((len(rows), parts[0].strip(), zh))
    return rows


def zh_fallback():
    """中文名缺失时回落到全国 373 labels。"""
    m = {}
    with open(LABELS_373, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = line.rstrip("\n").split("\t")
                m[p[0].strip()] = p[1].strip() if len(p) > 1 else ""
    return m


def build_book(labels, rarity, zh_map, keep=None):
    """labels -> 图鉴条目列表。keep(sci) 用于分册筛选；idx 恒为模型类别索引。"""
    book, missing_name = [], []
    for idx, sci, zh in labels:
        if keep is not None and not keep(sci):
            continue
        name = zh or zh_map.get(sci, "")
        if not name:
            missing_name.append(sci)
        book.append({"sci": sci, "zh": name, "idx": idx, "iucn": rarity.get(sci) or ""})
    book.sort(key=lambda e: e["sci"])
    return book, missing_name


def write_book(path, book):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False)


def check_region(region, labels_name, guide_name):
    """逐条比对图鉴 idx 与 labels 行号。返回错位条目列表。"""
    labels = read_labels(labels_name)
    sci_at = {i: s for i, s, _ in labels}
    with open(os.path.join(ASSETS, "guide", guide_name), encoding="utf-8") as f:
        book = json.load(f)
    bad = [e for e in book if sci_at.get(e["idx"]) != e["sci"]]
    unknown = [e["sci"] for e in book if e["sci"] not in {s for _, s, _ in labels}]
    absent = [s for _, s, _ in labels if s not in {e["sci"] for e in book}]
    print(f"{region:<9} 图鉴 {len(book):>3} / 类别 {len(labels):>3}  编号错位 {len(bad):>3}  "
          f"不在labels {len(unknown):>2}  类别有而图鉴缺 {len(absent):>2}")
    for e in bad[:3]:
        print(f"    idx={e['idx']} 图鉴={e['sci']} 实际={sci_at.get(e['idx'])}")
    if absent:
        print(f"    缺: {'、'.join(absent)}")
    return bad, unknown, absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=sorted({**SINGLE, **SPLIT}))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true", help="只校验不写盘")
    args = ap.parse_args()

    targets = ([args.region] if args.region
               else sorted({**SINGLE, **SPLIT}) if (args.all or args.check)
               else [])
    if not targets:
        ap.error("需要 --region / --all / --check 之一")

    if args.check:
        rc = 0
        for r in targets:
            name = SINGLE[r][1] if r in SINGLE else SPLIT[r]
            labels = SINGLE[r][0] if r in SINGLE else SPLIT_LABELS
            bad, unknown, absent = check_region(r, labels, name)
            rc = rc or (1 if bad or unknown else 0)
        sys.exit(rc)

    if any(r in SPLIT for r in targets):
        print(f"华南/西南分册依赖分布数据，转交 {os.path.basename(SPLIT_SCRIPT)}")
        subprocess.run([sys.executable, SPLIT_SCRIPT], check=True)

    rarity = json.load(open(RARITY, encoding="utf-8"))
    zh_map = zh_fallback()
    for r in targets:
        if r not in SINGLE:
            continue
        labels_name, guide_name = SINGLE[r]
        labels = read_labels(labels_name)
        book, missing_name = build_book(labels, rarity, zh_map)
        out = os.path.join(ASSETS, "guide", guide_name)
        write_book(out, book)
        print(f"{r}: {len(labels)} 类 -> {guide_name} {len(book)} 条"
              + (f"（缺中文名: {missing_name}）" if missing_name else ""))
        bad, unknown, _ = check_region(r, labels_name, guide_name)
        if bad or unknown:
            sys.exit(f"{r} 重生成后仍有错位/未知种，停下")


if __name__ == "__main__":
    main()

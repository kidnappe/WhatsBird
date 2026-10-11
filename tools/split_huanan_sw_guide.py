# -*- coding: utf-8 -*-
"""华南+西南合并图鉴解耦：按物种分布点拆成 华南(粤桂琼) / 西南(云贵川渝) 两册底册。

背景：
- 外包交付的 huanan_sw 138 类模型 labels 缺中文名（观感差的主因之一），本脚本顺带回填。
- 模型本身仍是 138 类合并模型，不动；解耦只发生在图鉴底册层。

拆分口径（动物地理区代码区分度不够——138 类多为常见广布种，VI/VII 双跨 123/126）：
- 用 data/huanan_sw/dataset.csv 每条 GBIF 记录的经纬度，按最近省界质心归到 7 省之一；
- 华南侧 = 粤桂琼，西南侧 = 云贵川渝；
- 物种进某册的条件：该侧记录数 >= MIN_N 且占比 >= SHARE；
- 两册都满足的双册收录（真实分布如此，华北京津冀/山西交叉同理）；
- 2026-10-05 实测阈值 SHARE=0.15：华南册 127 / 西南册 115，华南独 23、西南独 11，无落单。

产物：
- android/app/src/main/assets/guide/guide_huanan.json / guide_xinan.json（替代 guide_huanan_sw.json）
- android/app/src/main/assets/models/labels_huanan_sw.txt 回填中文名列（顺序=类别索引，只补第二列）

用法：python tools/split_huanan_sw_guide.py
"""
import csv
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "android", "app", "src", "main", "assets")
LABELS = os.path.join(ASSETS, "models", "labels_huanan_sw.txt")
LABELS_373 = os.path.join(ASSETS, "models", "labels_birdnet373.txt")
RARITY = os.path.join(ASSETS, "guide", "rarity.json")
DATASET = r"E:\code\code tools\data\huanan_sw\dataset.csv"
OUT_HUANAN = os.path.join(ASSETS, "guide", "guide_huanan.json")
OUT_XINAN = os.path.join(ASSETS, "guide", "guide_xinan.json")

SHARE = 0.15  # 该侧记录占比阈值
MIN_N = 3     # 该侧最少记录数

# 省界质心（近似，lat, lon）。华南册 = 粤桂琼；西南册 = 云贵川渝。
CENTROIDS = {
    "云南": (25.0, 101.8), "贵州": (26.6, 106.9), "四川": (30.3, 103.0), "重庆": (29.7, 107.0),
    "广西": (23.7, 108.8), "广东": (23.3, 113.8), "海南": (19.0, 109.8),
}
HUANAN_PROV = {"广东", "广西", "海南"}


def load_labels():
    """labels_huanan_sw.txt：每行 '学名\t中文名'，行序=类别索引。返回 [(idx, sci, zh)]。"""
    rows = []
    with open(LABELS, encoding="utf-8") as f:
        for i, line in enumerate(filter(lambda l: l.strip(), f)):
            parts = line.rstrip("\n").split("\t")
            rows.append((len(rows), parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")))
    return rows


def zh_map():
    m = {}
    with open(LABELS_373, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            m[p[0].strip()] = p[1].strip() if len(p) > 1 else ""
    return m


def side_counts():
    """sci -> (总记录, 华南侧, 西南侧)。坐标按最近质心归省。"""
    cnt = defaultdict(lambda: [0, 0, 0])
    with open(DATASET, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                la, lo = float(row["lat"]), float(row["lon"])
            except (ValueError, KeyError):
                continue
            sci = row["scientific_name"].strip()
            prov = min(CENTROIDS, key=lambda k: (CENTROIDS[k][0] - la) ** 2 * 1.2 + (CENTROIDS[k][1] - lo) ** 2)
            c = cnt[sci]
            c[0] += 1
            c[1 if prov in HUANAN_PROV else 2] += 1
    return cnt


def main():
    labels = load_labels()
    zh = zh_map()
    rarity = json.load(open(RARITY, encoding="utf-8"))
    counts = side_counts()

    huanan, xinan, report = [], [], []
    missing_zh = []
    for idx, sci, _ in labels:
        name = zh.get(sci, "")
        if not name:
            missing_zh.append(sci)
        entry = {"sci": sci, "zh": name, "idx": idx, "iucn": rarity.get(sci, "")}
        n, hn, sn = counts.get(sci, [0, 0, 0])
        in_h = n > 0 and hn >= MIN_N and hn / n >= SHARE
        in_s = n > 0 and sn >= MIN_N and sn / n >= SHARE
        if not in_h and not in_s:  # 兜底：记录不足或占比不突出，两册都收，别丢种
            in_h = in_s = True
        if in_h:
            huanan.append(entry)
        if in_s:
            xinan.append(entry)
        report.append((name or sci, hn, sn, n, in_h, in_s))

    for path, book in [(OUT_HUANAN, huanan), (OUT_XINAN, xinan)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(book, f, ensure_ascii=False)

    # 回填合并模型 labels 的中文名列（只动第二列，行序不变）
    lines = []
    for _, sci, old in labels:
        lines.append(f"{sci}\t{zh.get(sci) or old}")
    with open(LABELS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    only_h = [r for r in report if r[4] and not r[5]]
    only_s = [r for r in report if r[5] and not r[4]]
    print(f"底体 138 -> 华南册 {len(huanan)} 种 / 西南册 {len(xinan)} 种；"
          f"华南独 {len(only_h)}，西南独 {len(only_s)}")
    if missing_zh:
        print("缺中文名:", missing_zh)
    print("华南独有:", "、".join(r[0] for r in only_h))
    print("西南独有:", "、".join(r[0] for r in only_s))


if __name__ == "__main__":
    main()

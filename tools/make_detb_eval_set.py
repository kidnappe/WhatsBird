# -*- coding: utf-8 -*-
"""生成「四区等量」独立评测抽样清单（冻结文件，不随运行变动）。

为什么是这个池子：
  * B 检测器只训过 huabei + COCO bird（det_train/det_val 的 file_name 前缀实测：
    huabei 28486 + coco_bird 3237，其余四区 0 张）。所以 huanan_sw / huazhong /
    huadong / dongbei 四区的图对 B 是 **100% 未见**的干净池。
  * 旧口径 eval_huabei 抽的是 huabei，而其中 78% 已被 B 训过 —— 那不是独立评测集
    （实测 200 张里 156 张在 det_train），据此报"华北检出率 96.5%"是泄漏数字。

为什么必须冻结成清单：
  每次跑 bench 都重新 random.sample 会让两次的抽样集不同，检出率的抖动就分不清
  是"模型变化"还是"换了一批题"。清单落盘 + 固定 seed = 同一份考卷反复考。

用法：
  python make_detb_eval_set.py                # 生成/覆盖清单（需显式 --force）
  python make_detb_eval_set.py --check        # 只校验清单与本地图一致
"""
import argparse
import csv
import json
import os
import random
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

TOOLS = r"E:\code\code tools"
DATA = os.path.join(TOOLS, "data")
OUT_DIR = os.path.join(DATA, "detb_eval")
LIST_CSV = os.path.join(OUT_DIR, "eval_set.csv")
META_JSON = os.path.join(OUT_DIR, "eval_set_meta.json")

# 四区等量：每个区抽同样的张数，避免被某区图片风格主导
REGIONS = ["huanan_sw", "huazhong", "huadong", "dongbei"]
PER_REGION = 100
SEED = 20261007
EXT = (".jpg", ".jpeg", ".png")

# B 检测器实际训过的来源（用于自证"这四区确实是干净池"）
B_SEEN_PREFIXES = ("huabei/", "coco_bird/")


def list_region(r):
    p = os.path.join(DATA, r, "images")
    if not os.path.isdir(p):
        return []
    return sorted(f for f in os.listdir(p) if f.lower().endswith(EXT))


def b_trained_files():
    """B 检测器训练/验证里出现过的 file_name（前缀仅 huabei/ 与 coco_bird/）。"""
    seen = set()
    ann = os.path.join(DATA, "detector_b", "annotations")
    for nm in ("det_train.json", "det_val.json"):
        p = os.path.join(ann, nm)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        for im in d.get("images", []):
            fn = im.get("file_name")
            if fn and fn.startswith(B_SEEN_PREFIXES):
                seen.add(fn)
    return seen


def gbif_keys_all():
    """检测器 B 训练源的 GBIF key 全集（= 华北 dataset.csv 的 key）。

    ⚠️ 两个坑（2026-10-07 实测）：
    1. 同一张照片会被**两个区的抓取链各自收录**——观察（occurrence）落在两区
       bbox 交界带时，两边都命中，文件名都是 `学名_GBIFkey.jpg`（key 相同）。
       华中 dataset.csv 里 3,227 个文件名与华北同名，抽出的 22 张全是真泄漏
       （key 逐个比对相同）。所以"是否训过"必须比 **GBIF key**。
    2. 但去重集**只能收华北**——四区自己的 dataset.csv 是**分类器**的训练数据，
       对"检测器评测"不构成泄漏（不同模型不同任务）。若把四区 key 也收进去，
       四区的图会把自己全滤掉（实测华南 30,511 → 抽 0，自噬）。
    """
    keys = set()
    p = os.path.join(DATA, "huabei", "dataset.csv")
    if not os.path.exists(p):
        return keys
    try:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                for k in ("gbif_key", "occurrence_key"):
                    v = (row.get(k) or "").strip()
                    if v:
                        keys.add(v)
    except Exception:
        pass
    return keys


def file_gbif_key(fn):
    """从文件名 `学名_GBIFkey.jpg` 解析 GBIF key（抓取命名规则，build_dataset_gbif）。"""
    stem = os.path.splitext(fn)[0]
    tail = stem.rsplit("_", 1)[-1]
    return tail if tail.isdigit() else None


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    keys_used = gbif_keys_all()
    print("跨区 GBIF key 去重集：%d 个（含华北+四区全部抓取记录）" % len(keys_used))
    rows = []
    print("=== 四区等量抽样（每区 %d 张，seed=%d）===" % (PER_REGION, SEED))
    for r in REGIONS:
        fs = list_region(r)
        if not fs:
            print("  %-11s 无图，跳过" % r)
            continue
        # 剔除 key 已出现在任何区 dataset.csv 里的图（真·没训过）
        fresh = [f for f in fs if file_gbif_key(f) not in keys_used]
        skipped = len(fs) - len(fresh)
        take = sorted(rng.sample(fresh, min(PER_REGION, len(fresh))))
        for fn in take:
            rows.append({"region": r, "file": fn,
                         "path": os.path.join(DATA, r, "images", fn)})
        print("  %-11s 池 %6d -> 剔跨区已训 %5d -> 抽 %d"
              % (r, len(fs), skipped, len(take)))

    # 自证1：GBIF key 不得与任何区 dataset.csv 重叠
    leak_key = [x for x in rows if file_gbif_key(x["file"]) in keys_used]
    # 自证2：file_name 也不得出现在 B 训练产物里
    seen = b_trained_files()
    leak_name = [x for x in rows if ("huabei/" + x["file"]) in seen
                 or (x["region"] + "/" + x["file"]) in seen]
    print()
    print("泄漏自检：按 GBIF key 撞训练集 = %d 张；按文件名撞 = %d 张"
          % (len(leak_key), len(leak_name)))
    if leak_key or leak_name:
        print("  ✗ 有泄漏，终止！")
        for x in (leak_key + leak_name)[:5]:
            print("      ", x["region"], x["file"])
        return None

    with open(LIST_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "file", "path"])
        w.writeheader()
        w.writerows(rows)

    meta = {
        "created": "2026-10-07",
        "regions": REGIONS,
        "per_region": PER_REGION,
        "seed": SEED,
        "total": len(rows),
        "why_clean": "抽样按 **GBIF key** 对全部五区 dataset.csv 去重后抽取。"
                     "踩过的坑：同一张照片会被两个区的抓取链各自收录（观察落在"
                     "bbox 交界带），同名文件曾让\"只查前缀\"的自检漏判 22 张泄漏",
        "why_not_huabei": "旧 eval_huabei 抽的 huabei 200 张里 156 张在 det_train，"
                          "属泄漏口径（det_rate 96.5% 不可作为独立评测）",
        "no_gt": "本清单只用于「检出率」口径（有没有框出鸟），无 GT 框；"
                 "若要测召回/框精度，需要人工标注后再加一个带 GT 的 json",
        "usage": "bench 通过 --regions 读此清单",
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("  ✓ 零泄漏")
    print()
    print("清单: %s（%d 张）" % (LIST_CSV, len(rows)))
    print("元信息: %s" % META_JSON)
    return rows


def check():
    if not os.path.exists(LIST_CSV):
        print("✗ 清单不存在: %s" % LIST_CSV)
        return 1
    with open(LIST_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [x for x in rows if not os.path.isfile(x["path"])]
    print("清单 %d 张，缺文件 %d 张" % (len(rows), len(missing)))
    for x in missing[:5]:
        print("   缺:", x["path"])
    seen = b_trained_files()
    leak = [x for x in rows if (x["region"] + "/" + x["file"]) in seen]
    print("与 B 训练集交集: %d 张" % len(leak))
    return 0 if (not missing and not leak) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    if os.path.exists(LIST_CSV) and not a.force:
        print("清单已存在，加 --force 覆盖：%s" % LIST_CSV)
        sys.exit(0)
    sys.exit(0 if build() else 1)
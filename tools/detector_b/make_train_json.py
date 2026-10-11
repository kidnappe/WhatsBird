# -*- coding: utf-8 -*-
"""伪标签 + COCO bird → YOLOX 训练/验证 COCO json（含冻结划分清单 + 目录 junction）。

划分纪律（同分类器侧 freeze_val）：
- val 10%：huabei 图按 occurrence_key（缺则按 file 名）md5 分桶，稳定可复现；
  同一 observation 的图不会一半进 train 一半进 val（防兄弟观察泄漏）
- COCO bird 图全部进 train（val2017 是 bench 评测集，绝不混入）
- COCO bird tiny/small 桶（GT 面积 <1%）的图额外复制 ×2 份进 train = 对策④ 过采样

标签取舍：
- low_conf 框（分类器 top1<0.3）默认保留；人审 quality/ 样张后可 --drop-low-conf 重出
- 全帧+rescan 均无框的图默认【不进】训练集——华北图基本都含鸟，无框 ≈ 漏检，
  当负样本教"这里没鸟"是自举盲区下的主动伤害（--keep-empty 才收）

yolox 路径语义（CocoDataset 硬编码）：标注在 <data_dir>/annotations/<json>，
图片在 <data_dir>/<name>/<file_name>，train/val 的 name 硬编码 train2017/val2017。
本脚本用 Windows 目录 junction 免拷贝接上 huabei 与 coco_bird。

Usage: python make_train_json.py [--drop-low-conf] [--keep-empty]
产出：detector_b/{annotations/det_train.json, annotations/det_val.json, split_manifest.json}
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

DB = r"E:\code\code tools\data\detector_b"
DATA = r"E:\code\code tools\data\huabei"
COCO_BIRD = os.path.join(DB, "coco_bird")
ANN_DIR = os.path.join(DB, "annotations")
CAT = [{"id": 1, "name": "bird"}]


def junction(link, target):
    os.makedirs(os.path.dirname(link), exist_ok=True)  # mklink 要求父目录存在
    if os.path.isdir(link):
        return
    if os.path.lexists(link):
        os.rmdir(link)
    subprocess.run(["cmd", "/c", "mklink", "/J", link, target], check=True,
                   capture_output=True)
    print(f"junction: {os.path.basename(link)} -> {target}")


def md10(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-low-conf", action="store_true")
    ap.add_argument("--keep-empty", action="store_true")
    args = ap.parse_args()

    jl = os.path.join(DB, "pseudo_label", "pseudo_dets.jsonl")
    if not os.path.exists(jl):
        sys.exit(f"缺 {jl}——先跑 pseudo_label.py")
    occ = {}
    with open(os.path.join(DATA, "dataset.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv_row(f):
            if r.get("file"):
                occ[r["file"]] = r.get("occurrence_key") or r["file"]

    entries = {}
    with open(jl, encoding="utf-8") as f:
        for line in f:  # 重跑追加时按 file 后行覆盖
            j = json.loads(line)
            entries[j["file"]] = j
    # 人审闭环（apply_review.py 产物）：verdict=g 的整图剔除
    excl = set()
    ep = os.path.join(DB, "excluded_files.txt")
    if os.path.exists(ep):
        excl = {l.strip() for l in open(ep, encoding="utf-8") if l.strip()}

    os.makedirs(ANN_DIR, exist_ok=True)
    for nm, tgt in (("train2017/huabei", os.path.join(DATA, "images")),
                    ("train2017/coco_bird", os.path.join(COCO_BIRD, "train2017_bird")),
                    ("val2017/huabei", os.path.join(DATA, "images"))):
        junction(os.path.join(DB, *nm.split("/")), tgt)

    img_id, ann_id = 0, 0
    tr_i, tr_a, va_i, va_a = [], [], [], []
    val_files, n_low_drop, n_empty_skip, n_g_skip = [], 0, 0, 0

    def add(imgs, anns, fname, wh, boxes):
        nonlocal img_id, ann_id
        img_id += 1
        imgs.append({"id": img_id, "file_name": fname, "width": wh[0], "height": wh[1]})
        for x1, y1, x2, y2 in boxes:
            ann_id += 1
            anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})

    for fn, j in entries.items():
        if fn in excl:
            n_g_skip += 1
            continue
        boxes = []
        for d in j["dets"]:
            if args.drop_low_conf and d["low_conf"]:
                n_low_drop += 1
                continue
            boxes.append((d["x1"], d["y1"], d["x2"], d["y2"]))
        if not boxes and not args.keep_empty:
            n_empty_skip += 1
            continue
        wh = [j["w"], j["h"]]
        if md10(occ.get(fn, fn)) % 10 == 0:
            val_files.append(fn)
            add(va_i, va_a, f"huabei/{fn}", wh, boxes)
        else:
            add(tr_i, tr_a, f"huabei/{fn}", wh, boxes)

    # COCO bird：全部进 train，tiny/small（GT 面积<1%）×3 份（原件 + 2 复制）
    man_p = os.path.join(COCO_BIRD, "manifest.json")
    if os.path.exists(man_p):
        man = json.load(open(man_p))
        for fn, wh in man["wh"].items():
            boxes, small = [], False
            lp = os.path.join(COCO_BIRD, "labels", os.path.splitext(fn)[0] + ".txt")
            for line in open(lp):
                _, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
                bw, bh = bw * wh[0], bh * wh[1]
                boxes.append((cx * wh[0] - bw / 2, cy * wh[1] - bh / 2,
                              cx * wh[0] + bw / 2, cy * wh[1] + bh / 2))
                if bw * bh / (wh[0] * wh[1]) < 0.01:
                    small = True
            reps = 3 if small else 1
            for _ in range(reps):
                add(tr_i, tr_a, f"coco_bird/{fn}", wh, boxes)
        print(f"COCO bird 并入：{man['n_images']} 图（tiny+small {man['n_tiny_small']} ×3 份）")
    else:
        print(f"⚠️ 缺 {man_p}——本轮 json 不含 COCO bird（对策④缺失，跑 make_coco_bird_subset.py 后重出）")

    for nm, ii, aa in (("det_train", tr_i, tr_a), ("det_val", va_i, va_a)):
        with open(os.path.join(ANN_DIR, nm + ".json"), "w") as f:
            json.dump({"images": ii, "annotations": aa, "categories": CAT}, f)
    with open(os.path.join(DB, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "val_files": val_files, "drop_low_conf": args.drop_low_conf,
                   "keep_empty": args.keep_empty,
                   "n_train": len(tr_i), "n_val": len(va_i),
                   "n_train_ann": len(tr_a), "n_val_ann": len(va_a),
                   "n_low_drop": n_low_drop, "n_empty_skip": n_empty_skip},
                  f, ensure_ascii=False, indent=2)
    print(f"train {len(tr_i)} 图/{len(tr_a)} 框，val {len(va_i)} 图/{len(va_a)} 框"
          f"（低置信剔除 {n_low_drop}，空图跳过 {n_empty_skip}，坏图剔除 {n_g_skip}）")
    print(f"冻结清单 → {os.path.join(DB, 'split_manifest.json')}（val 文件列表已固化）")


def csv_row(f):
    import csv as _csv
    yield from _csv.DictReader(f)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""组装「鸟/非鸟」拒识二分类数据集（tools/train.py 直喂格式）。

正样本 __bird__：五区 dataset.csv 均衡抽样（全图，与五区分类器训练口径一致）。
负样本 __not_bird__：
  ① negatives_fp/ 的检测器 FP crop（harvest_negatives.py 产物，最对症）
  ② 实测视频随机背景块（避开检出框）
  ③ COCO val2017 非鸟图（排除 bird GT 的 125 张）

Usage: python build_reject_dataset.py --out "E:\code\code tools\data\reject_clf"
产出: <out>/dataset.csv + <out>/images/*.jpg
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys

import cv2

sys.path.insert(0, r"E:\code\observebird\tools")

REGIONS = ["huabei", "huazhong", "huanan_sw", "huadong", "dongbei"]
DATA = r"E:\code\code tools\data"
COCO_VAL = os.path.join(DATA, "coco", "val2017")
VIDEOS = [
    r"E:\Mishare\Screenrecorder-2026-10-07-16-39-50-586.mp4",
    r"E:\Mishare\Screenrecorder-2026-10-07-16-45-07-991.mp4",
]
FP_DIR = os.path.join(DATA, "reject_clf", "negatives_fp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "reject_clf"))
    ap.add_argument("--pos-per-region", type=int, default=1600)
    ap.add_argument("--coco-neg", type=int, default=2000)
    ap.add_argument("--bg-per-video", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pos-crop", type=float, default=0.7,
                    help="正样本中心裁剪比例：GBIF 全图鸟占比小、四周大片亮色天空，"
                         "与部署时『鸟占满框』的 crop 统计不符——v1/v2 的拒识模型学出"
                         "『绿暗物体+亮背景=鸟』的伪相关（绿瓶两代模型都漏）。"
                         "中心裁剪让鸟占满画面，贴近部署分布。")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []

    # ① 正样本：五区均衡抽样
    for region in REGIONS:
        csv_path = os.path.join(DATA, region, "dataset.csv")
        with open(csv_path, encoding="utf-8") as f:
            all_rows = [r for r in csv.DictReader(f)
                        if os.path.exists(os.path.join(DATA, region, "images", r["file"]))]
        rng.shuffle(all_rows)
        import io
        for r in all_rows[: a.pos_per_region]:
            name = f"pos_{region}_{len(rows):06d}.jpg"
            src_path = os.path.join(DATA, region, "images", r["file"])
            if a.pos_crop < 1.0:
                img = cv2.imread(src_path)
                if img is None:
                    continue
                ih, iw = img.shape[:2]
                cw, ch = int(iw * a.pos_crop), int(ih * a.pos_crop)
                cx, cy = (iw - cw) // 2, (ih - ch) // 2
                cv2.imwrite(os.path.join(img_dir, name), img[cy:cy + ch, cx:cx + cw],
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            else:
                shutil.copyfile(src_path, os.path.join(img_dir, name))
            rows.append({"file": name, "scientific_name": "__bird__",
                         "occurrence_key": r.get("occurrence_key", name)})
        print(f"  正样本 {region}: {min(a.pos_per_region, len(all_rows))}")

    # ② FP crop 负样本
    fp_manifest = os.path.join(FP_DIR, "manifest.json")
    if os.path.exists(fp_manifest):
        for m in json.load(open(fp_manifest, encoding="utf-8")):
            src = os.path.join(FP_DIR, m["file"])
            if os.path.exists(src):
                name = f"neg_fp_{len(rows):06d}.jpg"
                shutil.copyfile(src, os.path.join(img_dir, name))
                rows.append({"file": name, "scientific_name": "__not_bird__",
                             "occurrence_key": m["file"]})
        print(f"  负样本 FP crop: {sum(1 for r in rows if r['scientific_name']=='__not_bird__')}")

    # ③ 视频随机背景块（每帧 3 块，0.3-0.6 边长）
    for vi, video in enumerate(VIDEOS):
        if not os.path.exists(video):
            print(f"  [MISS] {video}")
            continue
        cap = cv2.VideoCapture(video)
        fid, kept = -1, 0
        while kept < a.bg_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            if fid % 7:
                continue
            h, w = frame.shape[:2]
            for _ in range(3):
                if kept >= a.bg_per_video:
                    break
                cw = int(w * rng.uniform(0.25, 0.55))
                ch = int(h * rng.uniform(0.25, 0.55))
                cx, cy = rng.randint(0, w - cw), rng.randint(0, h - ch)
                name = f"neg_bg{vi}_{len(rows):06d}.jpg"
                cv2.imwrite(os.path.join(img_dir, name), frame[cy:cy + ch, cx:cx + cw],
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({"file": name, "scientific_name": "__not_bird__",
                             "occurrence_key": name})
                kept += 1
        cap.release()
        print(f"  负样本 视频背景块 {os.path.basename(video)}: {kept}")

    # ④ COCO 非鸟图
    from bench_detector_a import load_bird_gt  # noqa: E402  复用 bird GT 排除表
    bird_files = set(load_bird_gt().keys())
    coco_all = [f for f in os.listdir(COCO_VAL) if f.endswith(".jpg")]
    coco_neg = [f for f in coco_all if f not in bird_files]
    rng.shuffle(coco_neg)
    for f in coco_neg[: a.coco_neg]:
        name = f"neg_coco_{len(rows):06d}.jpg"
        shutil.copyfile(os.path.join(COCO_VAL, f), os.path.join(img_dir, name))
        rows.append({"file": name, "scientific_name": "__not_bird__",
                     "occurrence_key": f})
    print(f"  负样本 COCO 非鸟: {min(a.coco_neg, len(coco_neg))}")

    with open(os.path.join(a.out, "dataset.csv"), "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["file", "scientific_name", "occurrence_key"])
        wtr.writeheader()
        wtr.writerows(rows)
    n_pos = sum(1 for r in rows if r["scientific_name"] == "__bird__")
    print(f"合计 {len(rows)} 张（鸟 {n_pos} / 非鸟 {len(rows) - n_pos}）→ {a.out}/dataset.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
r"""给下载页挑「每个物种用哪张训练集照片」。

挑的标准不是「好看」，是**可验证**：
  1. 检测器（App 同款 yolox 640 int8）在这张照片上真的出框，且分数够高；
  2. 区域分类器（该数据集目录对应的区域模型）跑出来的 **top-1 学名 == 文件名学名**；
  3. 分辨率够大（页面上的图片区有 490~1070 px 宽）。

第 2 条是自检：文件名带学名，等于给了标准答案；模型答对了，说明喂图口径没写错。
凡是不满足的一律不选——宁可少一张图，也不放一张「识别不出来」的。

跑完写 site/shots/manifest.json，给 make_screens.py 用。

Usage: python pick_photos.py [--limit 60] [--min-score 0.35]
"""
import argparse
import glob
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_identify import classify, detect_best, expand_crop  # noqa: E402

DATA = r"E:\code\code tools\data"
OUT = r"E:\code\observebird\site\shots\manifest.json"

# 页面上要出现的物种 -> (学名, 说明来自哪个页面的哪块图片区)
WANT = {
    "hong-er": "Pycnonotus_jocosus",       # 红耳鹎：图鉴卡片 + 详情大图
    "hongshui": "Phoenicurus_auroreus",    # 红尾水鸲：图鉴卡片
    "huixique": "Cyanopica_cyanus",        # 灰喜鹊：图鉴卡片 + 日志缩略图
    "maque": "Passer_montanus",            # 麻雀：日志缩略图
    "queshu": "Copsychus_saularis",        # 鹊鸲：取景页
}

REGION_OF = {  # 数据集目录 -> 区域模型（与 RegionClassifierRouter 同口径）
    "huazhong": "huazhong", "huadong": "huadong", "huanan_sw": "huanan_sw",
    "huabei": "huabei", "dongbei": "dongbei", "national": "huazhong",
}


def candidates(sci, limit):
    files = []
    for pat in glob.glob(os.path.join(DATA, "*", "images", sci + "_*.jpg")):
        region = os.path.basename(os.path.dirname(os.path.dirname(pat)))
        files.append((region, pat))
    out = []
    for region, p in files:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue
        if w < 700 or h < 500:
            continue
        out.append((region, p, w, h))
    out.sort(key=lambda t: -(t[2] * t[3]))
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-score", type=float, default=0.35)
    a = ap.parse_args()

    manifest = {}
    for key, sci in WANT.items():
        cands = candidates(sci, a.limit)
        tried = matched = 0
        best = None
        for region, path, w, h in cands:
            tried += 1
            det, wh = detect_best(path)
            if not det or det["score"] < a.min_score:
                continue
            rect = expand_crop(det["box"], wh)
            try:
                top5 = classify(Image.open(path).convert("RGB").crop(
                    tuple(int(v) for v in rect)), REGION_OF.get(region, "huazhong"))
            except Exception as e:
                print("  分类失败", os.path.basename(path), e)
                continue
            if not top5 or top5[0]["sci"].lower() != sci.lower():
                continue
            matched += 1
            cand = {"photo": path, "region": region, "sci": sci,
                    "zh": top5[0]["zh"], "img_wh": [w, h],
                    "det_score": round(det["score"], 4),
                    "det_box": [round(v, 1) for v in det["box"]],
                    "top5": [{"sci": t["sci"], "zh": t["zh"], "p": round(t["p"], 4)} for t in top5]}
            # 先要认对，再要框分高，最后要分辨率大
            rank = (cand["det_score"], w * h)
            if best is None or rank > (best["det_score"], best["img_wh"][0] * best["img_wh"][1]):
                best = cand
            if matched >= 12:
                break
        print("%-10s %-22s 候选 %d / 尝试 %d / top1 命中 %d -> %s" % (
            key, sci, len(cands), tried, matched,
            ("OK det=%.2f %s %dx%d" % (best["det_score"], best["photo"].split("\\")[-1],
                                       best["img_wh"][0], best["img_wh"][1])) if best else "无合格照片"))
        if best:
            manifest[key] = best

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("写出", OUT, "共", len(manifest), "个物种")


if __name__ == "__main__":
    main()
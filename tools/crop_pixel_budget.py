# -*- coding: utf-8 -*-
"""用已落盘的检测框，算「端上真正喂给分类器的那块」有多少像素 —— 检验 C（升分辨率重抓）的必要性。

背景（2026-10-09）：C 要花 15~25 小时 + ~22GB 磁盘重下高清图，理由是
「训练图短边只有 256，10 倍变焦下裁出来的小块是糊的」。
但**这个前提从没被量化过**：端上管线是
  检测框 → 外扩 4.0（BOX_EXPAND_RATIO，裁剪宽 = 5×框宽）→ 短边缩到 256 → 中心裁 224
所以关键不是原图多少像素，而是**裁剪出来的那块有多少像素、要放大多少倍**。

本脚本只做几何计算（读 boxes.jsonl + 读图片头拿真实尺寸，不解码像素），分钟级出结果。

⚠️ **必须钳制到画幅边界**：端上 `expand_box()` 会把外扩后的框夹进画幅内，
   而本脚本初版漏了这一步，算出「裁剪短边中位 289px」——比原图短边 256 还大，
   物理上不可能。**已修正：按每张图的真实宽高钳制。**

用法：python tools/crop_pixel_budget.py [--frac 2.0] [--max-n 6000]
"""
import argparse
import json
import os
from collections import defaultdict

BOXES = r"E:\code\code tools\data\crop_regime\boxes.jsonl"
DATA = r"E:\code\code tools\data"
PRE_SHORT = 256   # 端上：短边缩到 256
CROP = 224        # 再中心裁 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=2.0,
                    help="脚本口径外扩比例；端上 BOX_EXPAND_RATIO = 2×frac")
    ap.add_argument("--max-n", type=int, default=6000)
    a = ap.parse_args()

    from PIL import Image

    recs = [json.loads(l) for l in open(BOXES, encoding="utf-8")]
    by = defaultdict(list)
    for r in recs:
        if r.get("det"):
            by[r["region"]].append(r)
    # 均匀抽样，控制读图头开销
    for region in by:
        rs = by[region]
        if len(rs) > a.max_n // len(by):
            step = len(rs) // (a.max_n // len(by)) + 1
            by[region] = rs[::step]

    f = 1 + 2 * a.frac
    print(f"样本 {sum(len(v) for v in by.values())} 张；frac={a.frac}（端上 BOX_EXPAND_RATIO={a.frac*2:.0f}）")
    print(f"端上管线：裁剪宽 = {f:.0f}×框宽 → **钳进画幅** → 短边缩到 {PRE_SHORT} → 中心裁 {CROP}\n")

    print(f"{'区':<11}{'样本':>6}{'裁剪短边中位':>14}{'p10':>7}{'放大中位':>10}"
          f"{'放大>3倍':>11}")
    print("-" * 61)
    allsh, allup = [], []
    for region, rs in sorted(by.items()):
        shorts, upscales = [], []
        img_dir = os.path.join(DATA, region, "images")
        for r in rs:
            try:                                   # 只读图片头，不解码像素
                with Image.open(os.path.join(img_dir, r["file"])) as im:
                    W, H = im.size
            except Exception:
                continue
            x1, y1, x2, y2 = r["det"][1:]
            cw = min((x2 - x1) * f, W)              # ← 钳制（初版漏了这步）
            ch = min((y2 - y1) * f, H)
            s = min(cw, ch)
            shorts.append(s)
            upscales.append(PRE_SHORT / max(s, 1e-6))
        shorts.sort(); upscales.sort()
        n = len(shorts)
        if not n:
            continue
        allsh += shorts; allup += upscales
        heavy = sum(1 for u in upscales if u > 3.0)
        print(f"{region:<11}{n:>6}{shorts[n//2]:>13.0f}px{shorts[n//10]:>7.0f}"
              f"{upscales[n//2]:>9.1f}x{heavy:>7} ({heavy*100//n}%)")

    allsh.sort(); allup.sort(); n = len(allsh)
    heavy = sum(1 for u in allup if u > 3.0)
    print(f"\n全样本 n={n}: 裁剪短边 中位 {allsh[n//2]:.0f}px / p10 {allsh[n//10]:.0f}px / "
          f"p90 {allsh[9*n//10]:.0f}px")
    print(f"          放大倍数 中位 {allup[n//2]:.1f}x / p90 {allup[9*n//10]:.1f}x")
    print(f"          需要放大 >1x（短边不足 256px）的占比 "
          f"{sum(1 for u in allup if u > 1.0)*100//n}%；>3x 的 {heavy*100//n}%")
    print("\n读法：放大 >1x = 这块被插值撑大了，像素不够就是不够；"
          ">3x 基本等于只剩轮廓。")


if __name__ == "__main__":
    raise SystemExit(main())
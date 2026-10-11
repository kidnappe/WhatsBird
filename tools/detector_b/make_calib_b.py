# -*- coding: utf-8 -*-
"""B 量化校准集清单：华北随机抽样 + 难例样张（主文档 §阶段 4「校准集换域」）。

COCO 原校准 npz（calib_512.npz）不再用于 B——换域量化损失会被放大。
产出 calib_b_files.txt（绝对路径清单），交给 quantize_ort.py 的可选参数使用；
喂法口径（letterbox 114 + BGR 0-255）由 quantize_ort.py 负责，本脚本只管选图。

Usage: python make_calib_b.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _progress import update as prog_update  # noqa: E402  progress.json（监控台读）

TOOLS = r"E:\code\code tools"
HB_IMG = os.path.join(TOOLS, "data", "huabei", "images")
HARD_LIST = os.path.join(TOOLS, "data", "detector_b", "pseudo_label", "hard_examples.txt")
OUT = os.path.join(TOOLS, "data", "detector_b", "calib_b_files.txt")
N_HUABEI, N_HARD = 100, 60

hb = sorted(f for f in os.listdir(HB_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png")))
hb = [os.path.join(HB_IMG, f) for f in random.Random(42).sample(hb, min(N_HUABEI, len(hb)))]

hard = []
if os.path.exists(HARD_LIST):
    hard = [l.strip() for l in open(HARD_LIST, encoding="utf-8") if l.strip()]
    hard = [p for p in hard if os.path.exists(p)]
    random.Random(42).shuffle(hard)
    hard = hard[:N_HARD]

files, seen = [], set()
for p in hb + hard:
    rp = os.path.normpath(p)
    if rp not in seen:
        seen.add(rp)
        files.append(rp)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(files))
prog_update(calib_files=len(files))
print(f"校准清单 {len(files)} 张（华北 {len(hb)} + 难例 {len(hard)}）→ {OUT}")
if not hard:
    print("⚠️ 没找到 hard_examples.txt（伪标签还没跑完？）——先只出了华北部分，伪标签完成后重跑本脚本")

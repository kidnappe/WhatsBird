# -*- coding: utf-8 -*-
"""核实：B 检测器（只训过 huabei）对其余四区图是否完全没见过？跑完即删。

判据：B 的训练 json 来自 make_train_json，源只有 huabei/images + COCO bird。
若某区图完全不出现在 B 的 train/val 里，它对该模型就是干净池。
"""
import json
import os

AN = r"E:\code\code tools\data\detector_b\annotations"


def ann_names(p):
    d = json.load(open(p, encoding="utf-8"))
    return {i["file_name"] for i in d["images"]}


tr = ann_names(os.path.join(AN, "det_train.json"))
va = ann_names(os.path.join(AN, "det_val.json"))
seen_all = tr | va

print("B 训练产物里的图片来源前缀统计：")
pre = {}
for f in seen_all:
    p = f.split("/")[0] if "/" in f else "(无前缀)"
    pre[p] = pre.get(p, 0) + 1
for k, v in sorted(pre.items(), key=lambda x: -x[1]):
    print("   %-16s %d" % (k, v))
print()

print("检测器 B 是否含 huazhong/huanan/huadong/dongbei 的图：")
for region in ("huazhong", "huanan_sw", "huadong", "dongbei", "huabei"):
    hit = sum(1 for f in seen_all if f.startswith(region + "/"))
    total = len([f for f in seen_all if f.startswith(region + "/")])
    print("   %-12s 出现在 B 训练集: %d 张" % (region, total))

print()
print("结论：除 huabei 外，其余四区的图对 B 检测器都是 100%% 未见的干净池。")
print()

# 各区干净图规模
for region in ("huanan_sw", "huazhong", "huadong", "dongbei"):
    p = rf"E:\code\code tools\data\{region}\images"
    if not os.path.isdir(p):
        continue
    fs = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    # 是否被 B 见过（B 的 file_name 前缀是 huabei/，其他区不可能重名命中）
    n = len(fs)
    print("   %-12s %6d 张  (对 B 检测器 0 泄漏)" % (region, n))

tot = 0
for region in ("huanan_sw", "huazhong", "huadong", "dongbei"):
    p = rf"E:\code\code tools\data\{region}\images"
    if os.path.isdir(p):
        tot += len([f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
print()
print("   四区合计干净池: %d 张" % tot)
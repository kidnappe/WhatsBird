# -*- coding: utf-8 -*-
r"""拒识模型验收：FP crop 拒识率 vs 真鸟误拒率，并给出部署阈值建议。

评测集：
  ① negatives_fp/*.jpg —— 装机实测视频收割的检测器 FP crop（期望判 __not_bird__）
  ② 新鲜鸟图 —— 五区 dataset.csv 中**未进拒识训练集**的抽样（期望判 __bird__）
Usage: python eval_reject.py
"""
import csv
import os
import random
import sys

import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, r"E:\code\observebird\tools")

CKPT = r"E:\code\code tools\models\reject_mnv4_v1\best.pt"
FP_DIR = r"E:\code\code tools\data\reject_clf\negatives_fp"
REGIONS = ["huabei", "huazhong", "huanan_sw", "huadong", "dongbei"]
DATA = r"E:\code\code tools\data"
REJECT_DS = os.path.join(DATA, "reject_clf", "dataset.csv")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.224 + 0.005]
TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else CKPT
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from train import build_model
    net = build_model(ck["model_name"], len(ck["species"])).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    idx_of = {s: i for i, s in enumerate(ck["species"])}
    print(f"拒识模型 {ck['model_name']} best_top1={ck['top1']*100:.2f}% species={ck['species']}")

    def p_not_bird(img: Image.Image) -> float:
        with torch.no_grad():
            o = net(TF(img).unsqueeze(0).to(dev))
            return torch.softmax(o, 1)[0, idx_of["__not_bird__"]].item()

    # ① FP crops
    fp_scores = []
    for f in sorted(os.listdir(FP_DIR)):
        if f.endswith(".jpg"):
            img = Image.open(os.path.join(FP_DIR, f)).convert("RGB")
            fp_scores.append(p_not_bird(img))

    # ② 新鲜鸟图（不在拒识训练集里的）
    trained = set()
    with open(REJECT_DS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            trained.add(r["occurrence_key"])
    rng = random.Random(11)
    bird_scores = []
    for region in REGIONS:
        with open(os.path.join(DATA, region, "dataset.csv"), encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if r.get("occurrence_key", "") not in trained
                    and os.path.exists(os.path.join(DATA, region, "images", r["file"]))]
        rng.shuffle(rows)
        for r in rows[:200]:
            try:
                bird_scores.append(p_not_bird(Image.open(
                    os.path.join(DATA, region, "images", r["file"])).convert("RGB")))
            except Exception:
                pass

    print(f"\nFP crop（期望 not_bird）: n={len(fp_scores)}")
    print(f"  P(not_bird) 分布: 中位 {sorted(fp_scores)[len(fp_scores)//2]:.3f}  "
          f"min {min(fp_scores):.3f}  max {max(fp_scores):.3f}")
    print(f"新鲜鸟图（期望 bird）: n={len(bird_scores)}")
    bs = sorted(bird_scores)
    print(f"  P(not_bird) 分布: 中位 {bs[len(bs)//2]:.3f}  min {bs[0]:.3f}  max {bs[-1]:.3f}")

    print(f"\n{'阈值':>6} {'FP拒识率':>9} {'真鸟误拒率':>10}")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        rej = sum(1 for s in fp_scores if s >= th) / len(fp_scores)
        frr = sum(1 for s in bird_scores if s >= th) / len(bird_scores)
        print(f"{th:>6.2f} {rej*100:>8.1f}% {frr*100:>9.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

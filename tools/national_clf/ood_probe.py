# -*- coding: utf-8 -*-
r"""开集置信度诊断：区域模型对类目外鸟（141 缺口种）到底会不会「给不出答案」。

问题背景：级联兜底（低置信→上抛 cn373）的触发器是 max-softmax，但闭集分类器
对 OOD 输入普遍 overconfidence——本脚本用实测分布回答「阈值这条路还剩多少信号」。

口径：
  in-dist  = <region> 自己 dataset.csv 随机抽样（注意：来自训练集，置信度偏高，
             结论只会更保守——train 图都拉不开，val 更拉不开）
  OOD      = national_gap 缺口种图（任何五区类目外，模型绝对没见过）
  指标     = MSP(max softmax) / energy(logsumexp logits) / margin(top1-top2 prob)

Usage: python ood_probe.py --region dongbei --n 400
"""
import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"E:\code\observebird\tools")

MODELS = r"E:\code\code tools\models"
DATA = r"E:\code\code tools\data"


def load_rows(csv_path):
    return list(csv.DictReader(open(csv_path, encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="dongbei")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    mdir = a.model_dir or os.path.join(MODELS, f"{a.region}_mnv4")
    ck = torch.load(os.path.join(mdir, "best.pt"), map_location="cpu", weights_only=False)

    import timm
    model = timm.create_model(ck["model_name"], pretrained=False,
                              num_classes=len(ck["labels"]))
    model.load_state_dict(ck["model"])
    model.eval()
    from timm.data import resolve_data_config
    cfg = resolve_data_config({}, model=model)
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    size = cfg["input_size"][-1]

    rng = random.Random(0)

    def sample(rows, img_dir, n):
        rng.shuffle(rows)
        out = []
        for r in rows:
            p = os.path.join(img_dir, r["file"])
            if os.path.exists(p):
                out.append((p, r["scientific_name"]))
            if len(out) >= n:
                break
        return out

    in_rows = load_rows(os.path.join(DATA, a.region, "dataset.csv"))
    in_s = sample(in_rows, os.path.join(DATA, a.region, "images"), a.n)
    gap_rows = load_rows(os.path.join(DATA, "national_gap", "dataset.csv"))
    gap_s = sample(gap_rows, os.path.join(DATA, "national_gap", "images"), a.n)
    print(f"{a.region}: in-dist {len(in_s)} 张 / OOD(缺口种) {len(gap_s)} 张")

    def run(paths):
        msp, energy, margin = [], [], []
        with torch.no_grad():
            for i in range(0, len(paths), 32):
                imgs = []
                for p, _ in paths[i:i + 32]:
                    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (size, size)).astype(np.float32) / 255.0
                    im = (im - mean) / std
                    imgs.append(torch.from_numpy(im).permute(2, 0, 1))
                x = torch.stack(imgs)
                logits = model(x)
                prob = F.softmax(logits, dim=1)
                top2 = prob.topk(2, dim=1).values
                msp += top2[:, 0].tolist()
                margin += (top2[:, 0] - top2[:, 1]).tolist()
                energy += torch.logsumexp(logits, dim=1).tolist()
        return np.array(msp), np.array(energy), np.array(margin)

    m_in, e_in, g_in = run(in_s)
    m_od, e_od, g_od = run(gap_s)

    def q(arr, ps=(5, 50, 95)):
        return "/".join(f"p{p}={np.percentile(arr, p):.3f}" for p in ps)

    print("MSP     in:", q(m_in), " OOD:", q(m_od))
    print("energy  in:", q(e_in), " OOD:", q(e_od))
    print("margin  in:", q(g_in), " OOD:", q(g_od))

    def auroc(pos, neg):  # pos 应得分高
        s = np.concatenate([pos, neg])
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        r = np.argsort(np.argsort(s)) + 1
        n1, n0 = lab.sum(), (1 - lab).sum()
        return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    print(f"AUROC  MSP={auroc(m_in, m_od):.3f}  energy={auroc(e_in, e_od):.3f}  margin={auroc(g_in, g_od):.3f}")
    for pct in (1, 5, 10):
        th = np.percentile(m_in, pct)
        cover = (m_od < th).mean() * 100
        print(f"阈值=in-dist p{pct} ({th:.3f}): 上抛 {cover:.1f}% 的类目外鸟")
    # OOD 高置信占比——用户观察的直接验证
    print(f"OOD 中 MSP>0.9: {(m_od > 0.9).mean()*100:.1f}%  >0.99: {(m_od > 0.99).mean()*100:.1f}%")


if __name__ == "__main__":
    main()

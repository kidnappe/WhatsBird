# -*- coding: utf-8 -*-
r"""五区部署模型低置信上抛阈值标定（级联链路第一级，见 docs/handoff 讨论）。

口径（吸取 ood_probe.py 的教训）：
  - 模型 = 手机上实际部署的那批（huabei_mnv4_rec / huadong_300_rec / ...，见
    copy_models_to_android.py 的 FILES 表），不是同名老模型；
  - in-dist 样本 = 该模型训练时同 dataset+val_frac+seed 的 grouped_split val 侧
    （模型没见过，无训练自信偏置）；
  - OOD 样本 = 缺口种图(national_gap，天然五区皆无) + 邻区 val 里不在本区类表的种
    （"隔壁区的鸟"——GPS 漂移/旅拍的真实场景）；
  - 阈值取 in-dist val MSP 的 p5（误伤≤5%），同时给 p1/p10 两档备选。

纯 CPU 运行（torch 线程=4），不与训练抢 GPU。产出:
  tools/national_clf/region_thresholds.json   （Router 侧直接读的文件）
  终端汇总表

Usage: python -u calib_thresholds.py [--n-in 600] [--n-gap 400] [--n-cross 400]
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"E:\code\observebird\tools")
from train import grouped_split  # noqa: E402  与训练同一划分函数

torch.set_num_threads(4)  # 让出 CPU 给训练 dataloader

MODELS_DIR = r"E:\code\code tools\models"
GAP_CSV = r"E:\code\code tools\data\national_gap\dataset.csv"
GAP_IMG = r"E:\code\code tools\data\national_gap\images"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "region_thresholds.json")

# 与 copy_models_to_android.py 的部署表一致
DEPLOYED = {
    "huabei": "huabei_mnv4_rec",
    "huazhong": "huazhong_mnv4_25k",
    "huanan_sw": "huanan_sw_mnv4_rec",
    "huadong": "huadong_mnv4_300_rec",
    "dongbei": "dongbei_mnv4_rec",
}


def load_model(ck):
    import timm
    from timm.data import resolve_data_config
    model = timm.create_model(ck["model_name"], pretrained=False,
                              num_classes=len(ck["labels"]))
    model.load_state_dict(ck["model"])
    model.eval()
    cfg = resolve_data_config({}, model=model)
    return model, np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)


def msp_all(model, mean, std, paths, size=224):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            xs = []
            for p in paths[i:i + 32]:
                im = cv2.imread(p)
                if im is None:
                    continue
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (size, size)).astype(np.float32) / 255.0
                xs.append(torch.from_numpy((im - mean) / std).permute(2, 0, 1))
            out.extend(F.softmax(model(torch.stack(xs)), 1).max(1).values.tolist())
    return np.array(out)


def spread_by_species(by_sp, target, rng):
    """按物种轮转抽样，均衡覆盖（别全堆在优势种上）。"""
    picked = []
    order = sorted(by_sp)
    rng.shuffle(order)
    pools = {s: by_sp[s][:] for s in order}
    for s in pools.values():
        rng.shuffle(s)
    while len(picked) < target and any(pools[s] for s in order):
        for s in order:
            if pools[s] and len(picked) < target:
                picked.append(pools[s].pop())
    return picked


def region_val_paths(region, cfg, n, rng):
    rows = list(csv.DictReader(open(cfg["dataset"], encoding="utf-8")))
    _, va = grouped_split(rows, val_frac=cfg.get("val_frac", 0.2),
                          seed=cfg.get("seed", 0))
    img_dir = os.path.join(os.path.dirname(cfg["dataset"]), "images")
    by_sp = defaultdict(list)
    for r in va:
        p = os.path.join(img_dir, r["file"])
        if os.path.exists(p):
            by_sp[r["scientific_name"]].append(p)
    return spread_by_species(by_sp, n, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-in", type=int, default=600)
    ap.add_argument("--n-gap", type=int, default=400)
    ap.add_argument("--n-cross", type=int, default=400)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    import random
    rr = random.Random(0)

    gap_rows = list(csv.DictReader(open(GAP_CSV, encoding="utf-8")))
    gap_by_sp = defaultdict(list)
    for r in gap_rows:
        p = os.path.join(GAP_IMG, r["file"])
        if os.path.exists(p):
            gap_by_sp[r["scientific_name"]].append(p)

    out = {}
    for region, mdir in DEPLOYED.items():
        ck = torch.load(os.path.join(MODELS_DIR, mdir, "best.pt"),
                        map_location="cpu", weights_only=False)
        model, mean, std = load_model(ck)
        species = set(ck["labels"])

        in_p = region_val_paths(region, ck["train_cfg"], a.n_in, rr)
        gap_p = spread_by_species(gap_by_sp, a.n_gap, rr)
        # 邻区 val（排除本区认识的种）
        cross_p = []
        for other, odir in DEPLOYED.items():
            if other == region:
                continue
            ock = torch.load(os.path.join(MODELS_DIR, odir, "best.pt"),
                             map_location="cpu", weights_only=False)
            orows = list(csv.DictReader(open(ock["train_cfg"]["dataset"],
                                             encoding="utf-8")))
            _, ova = grouped_split(orows, val_frac=0.2, seed=ock["train_cfg"].get("seed", 0))
            oimg = os.path.join(os.path.dirname(ock["train_cfg"]["dataset"]), "images")
            pool = [os.path.join(oimg, r["file"]) for r in ova
                    if r["scientific_name"] not in species
                    and os.path.exists(os.path.join(oimg, r["file"]))]
            rr.shuffle(pool)
            cross_p += pool[: a.n_cross // 4]
            del ock
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        model, mean, std = None, None, None  # 释放后重载（省内存）
        ck2 = ck
        m, mean, std = load_model(ck2)
        v_in = msp_all(m, mean, std, in_p)
        v_gap = msp_all(m, mean, std, gap_p)
        v_cross = msp_all(m, mean, std, cross_p)
        del m, ck, ck2

        tau5 = float(np.percentile(v_in, 5))
        rec = {
            "model_dir": mdir, "val_top1": round(ck_top1(OUT_JSON, region, None) or 0, 4),
            "n_in": len(v_in), "n_gap": len(v_gap), "n_cross": len(v_cross),
            "tau_p1": round(float(np.percentile(v_in, 1)), 4),
            "tau_p5": round(tau5, 4),
            "tau_p10": round(float(np.percentile(v_in, 10)), 4),
            "throw_gap_p5": round(float((v_gap < tau5).mean()), 4),
            "throw_cross_p5": round(float((v_cross < tau5).mean()), 4),
            "throw_gap_p10": round(float((v_gap < np.percentile(v_in, 10)).mean()), 4),
            "msp_median_gap": round(float(np.median(v_gap)), 4),
            "msp_median_in": round(float(np.median(v_in)), 4),
        }
        out[region] = rec
        print(f"{region:10s} τ(p5)={rec['tau_p5']:.3f}  误伤5%时上抛率: 缺口种 "
              f"{rec['throw_gap_p5']*100:.0f}% / 邻区鸟 {rec['throw_cross_p5']*100:.0f}%"
              f"  (p10档缺口种 {rec['throw_gap_p10']*100:.0f}%)", flush=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"→ {OUT_JSON}")
    return 0


def ck_top1(_, region, __):
    """从部署表取模型 val top1（记录用，缺失返回 None）。"""
    try:
        import torch as t
        ck = t.load(os.path.join(MODELS_DIR, DEPLOYED[region], "best.pt"),
                    map_location="cpu", weights_only=False)
        return ck.get("top1")
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())

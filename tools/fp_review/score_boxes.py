# -*- coding: utf-8 -*-
r"""E-1：为「真实观鸟场景的 FP 到底多少」准备人工复核材料（2026-10-10）。

## 为什么做这个

`FP/图 0.86` 与验收线 `≤0.3` **都是 COCO 口径**（`docs/handoff/root/PRIORITIES-20261009.md` §10.1/§11.1），
而 COCO 的 FP 约**七成是群像图 GT 漏标真鸟的评测假象**。
真实观鸟场景的 FP **从来没被测过**——这是 E（检测器难例回流）做不做的唯一判据。

## 做法

用现成的 `data/crop_regime/boxes.jsonl`（20,000 条，五区齐全，det_th=0.45），
每条是一个**部署口径 crop**（frac 2.0 = 端上 `BOX_EXPAND_RATIO=4.0`），
喂**三个信号**给人看，由人给最终判决：

  ① `p_not_bird`  拒识门（birdnet_reject）判非鸟的概率 —— 唯一的二元信号
  ② `clf_top1` / `clf_prob` / `clf_match`  该区分类器 top1 学名 + 概率 + 是否等于该图 frozen_val 真值
  ③ `det_score`  检测器置信度

**分类器那一路是给复核人当参照的**：五区模型全是"只含鸟种"的分类器，
所以它永远会输出一个鸟种——**不能**用它判"是不是鸟"，但
「分类器高置信地喊出一个**别的**种」是"框歪了/拍到别的鸟"的强提示。

## 口径铁律（初版踩过，见 APP-LAYER-AUDIT §2.2.2）

端上 `InferenceEngine.expand` 每侧加 `宽×ratio/2`，`ratio = 2×frac`
⇒ frac=2.0 时**总宽 = 5× 框宽**。写成"每侧加 宽×frac/2"会只到 3×，crop 比真机小一截。

Usage:
    python tools/fp_review/score_boxes.py                 # 全量 20000
    python tools/fp_review/score_boxes.py --max-n 3000    # 抽样先试
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from PIL import Image
from torchvision import transforms as T

ROOT = r"E:\code\observebird"
sys.path.insert(0, os.path.join(ROOT, "tools"))

CKPT_REJECT = r"E:\code\code tools\models\reject_mnv4_v1\best.pt"
MODEL_ROOT = r"E:\code\code tools\models"
REGION_CKPT = {
    "huabei":    os.path.join(MODEL_ROOT, "huabei_mnv4_rec", "best.pt"),
    "huazhong":  os.path.join(MODEL_ROOT, "huazhong_mnv4_25k", "best.pt"),
    "huanan_sw": os.path.join(MODEL_ROOT, "huanan_sw_mnv4_rec", "best.pt"),
    "huadong":   os.path.join(MODEL_ROOT, "huadong_mnv4_300_rec", "best.pt"),
    "dongbei":   os.path.join(MODEL_ROOT, "dongbei_mnv4_rec", "best.pt"),
}
BOXES = r"E:\code\code tools\data\crop_regime\boxes.jsonl"
DATA = r"E:\code\code tools\data"
OUT_DIR = os.path.join(ROOT, "data", "fp_review")

FRAC = 2.0           # 脚本口径外扩；端上 BOX_EXPAND_RATIO = 2×frac = 4.0
PRE_SHORT, CROP = 256, 224

TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from train import build_model
    net = build_model(ck["model_name"], len(ck["species"])).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def expand_box(x0, y0, x1, y1, f, W, H):
    """每侧外扩 frac×边长并钳进画幅；总宽 = w×(1+2f)。frac=2.0 → 5w（= 端上 4.0）。"""
    w, h = x1 - x0, y1 - y0
    nx0, ny0 = max(0.0, x0 - w * f), max(0.0, y0 - h * f)
    nx1, ny1 = min(float(W), x1 + w * f), min(float(H), y1 + h * f)
    return (int(nx0), int(ny0), max(int(nx1), int(nx0) + 1),
            max(int(ny1), int(ny0) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--frac", type=float, default=FRAC,
                    help="脚本口径外扩比例；端上 BOX_EXPAND_RATIO = 2×frac")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)
    rej, rej_ck = build(CKPT_REJECT, dev)
    rej_idx = {s: i for i, s in enumerate(rej_ck["species"])}
    nb = rej_idx["__not_bird__"]
    print(f"拒识门 {rej_ck['model_name']} top1={rej_ck['top1']*100:.2f}%", flush=True)

    regs = {}
    for k, p in REGION_CKPT.items():
        if os.path.isfile(p):
            net, ck = build(p, dev)
            regs[k] = (net, ck)
            print(f"  {k:<10} {ck['model_name']} {len(ck['species'])} 类 "
                  f"top1={ck['top1']*100:.2f}%", flush=True)
        else:
            print(f"  {k:<10} 缺 ckpt: {p}", flush=True)

    recs = []
    with open(BOXES, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("det"):
                    recs.append(r)
    by = defaultdict(list)
    for r in recs:
        by[r["region"]].append(r)
    per = max(1, a.max_n // len(by))
    recs = [r for rs in by.values() for r in rs[::max(1, len(rs) // per)]]
    print(f"\n样本 {len(recs)} 条（五区各 ≤{per}），frac={a.frac} → 端上 ratio={a.frac*2:.0f}", flush=True)

    out, miss = [], 0
    t0 = time.time()
    buf = {"rej": []}                      # 输入缓冲：只放待推理的 tensor
    for k in regs:
        buf[k] = []
    # ⚠️ 每个分区缓冲要**记住自己对应的样本下标**（初版错位，见下方注释）
    idxbuf = {k: [] for k in regs}
    res = {}                               # 分类器：样本下标 -> (top1_sci, prob)，按需写入
    res_rej = []                           # 拒识门：单模型顺序追加，天然同序

    def flush():
        # ⚠️ 不能因为 rej 空就 return：各区缓冲是**按各自样本数**攒的，
        #    收尾时 rej 已空但各区仍有残余，不刷就会丢结果。
        if buf["rej"]:
            with torch.no_grad():
                res_rej.extend(
                    torch.softmax(rej(torch.stack(buf["rej"]).to(dev)), 1)[:, nb].tolist())
                buf["rej"].clear()
        for k, (net, ck) in regs.items():
            if not buf[k]:
                continue
            with torch.no_grad():
                p = torch.softmax(net(torch.stack(buf[k]).to(dev)), 1)
            top = p.max(1)
            for i in range(top.values.shape[0]):
                si = idxbuf[k][i]                  # ← 样本下标，逐条对齐
                res[si] = (ck["species"][int(top.indices[i])], float(top.values[i]))
            buf[k].clear()
            idxbuf[k].clear()

    for i, r in enumerate(recs):
        p = os.path.join(DATA, r["region"], "images", r["file"])
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                W, H = im.size
                x0, y0, x1, y1 = r["det"][1:]
                c = im.crop(expand_box(x0, y0, x1, y1, a.frac, W, H))
                buf["rej"].append(TF(c))
                si = len(out)
                # ⚠️⚠️ 只喂给**该样本自己那个区**的模型。
                #   初版写成 `for k in regs: buf[k].append(...)`，即每条样本被 5 个区的模型
                #   各预测一次，flush 时依次写 res[si] → **后写的覆盖先写的**。
                #   实测后果：华北 6/6 对、华南西南 0/6 对（落盘值全错，独立复算全对），
                #   整体 clf_match 被压到 60.5%（真值应约 32~40%）。
                rk = r["region"]
                if rk in regs:
                    buf[rk].append(TF(c))
                    idxbuf[rk].append(si)
                out.append({"region": r["region"], "file": r["file"], "sci": r["sci"],
                            "det_score": r["det"][0], "box": list(r["det"][1:])})
        except Exception:
            miss += 1
        if len(buf["rej"]) >= a.batch:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    flush()

    for o, pv in zip(out, res_rej):
        o["p_not_bird"] = pv
    for o, r in zip(out, (res[i] for i in range(len(out)))):
        if r is None:
            raise SystemExit(f"分类器结果缺失：{o['file']} —— 对齐逻辑又有问题")
        o["clf_top1"], o["clf_prob"] = r
        o["clf_match"] = bool(r[0].lower() == o["sci"].lower())

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"scored_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "frac": a.frac,
                   "device_ratio": a.frac * 2, "n": len(out), "miss": miss,
                   "source": BOXES, "rows": out}, f, ensure_ascii=False)
    print(f"\n落盘 {path}  n={len(out)} miss={miss}")

    n = len(out)
    hi = sum(1 for o in out if o["p_not_bird"] >= 0.8)
    mis = sum(1 for o in out if not o["clf_match"])
    print(f"\np_not_bird ≥0.8 的：{hi} 条（{hi/n*100:.2f}%）"
          f"  ← 拒识门现役工作点下的疑似 FP 上界")
    print(f"分类器 top1 ≠ 真值：{mis} 条（{mis/n*100:.1f}%）"
          f"  ← 复核人重点看这类（框歪 / 真值错 / 拍到别的鸟）")
    print(f"\n下一步：python tools/fp_review/serve_review.py（人审抽样，估真 FP/图）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
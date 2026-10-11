# -*- coding: utf-8 -*-
r"""拒识门在**部署裁剪口径**下的真鸟误拒率（2026-10-10）。

起因（docs/reports/APP-LAYER-AUDIT-20261010.md §2.2）：
`REJECT-GATE.md` 的验收数 **FP 拒识 99.1% / 真鸟误拒 0.6% @0.8** 测于 2026-10-07，
而 `BOX_EXPAND_RATIO` 是当天从 0.15 改到 4.0 的。看那组数的口径：

    负样本 = 装机视频收割的 FP crop 113 张 → harvest_negatives.py:71 是
             `frame[y1:y2, x1:x2]`，**紧贴检测框，没有任何外扩**
    正样本 = 五区 dataset.csv 的**整图**    → eval_reject.py:70-71 直接 Image.open
    端上实际 = **外扩 4.0 的 crop**（鸟占画面约 20%）→ ThreeStagePipeline.kt:35/152

三个分布两两不同 ⇒ 那组数描述的不是端上跑的那种图。本脚本补测正样本侧。

做法：复用 `data/crop_regime/boxes.jsonl`（20,000 条，frac=2.0 即端上 4.0，五区齐全，
**就是部署口径的 crop**），在**同一样本**上同时算两个口径，好让对比严格成立：

    A「整图」   = 原图整张 → Resize(256)+CenterCrop(224)   ← 旧验收的正样本口径
    B「部署crop」= 检测框外扩 frac=2.0 并**钳进画幅** → Resize(256)+CenterCrop(224)  ← 端上真实

预处理与模型构造**逐位复制** tools/detector_b/eval_reject.py，同一个 ckpt。

⚠️ 负样本侧本脚本测不了：FP crop 收割时就已裁掉上下文，而**原始视频已不在盘上**
   （全仓 0 个 mp4/mov/avi），无法重新外扩。负样本只能用 240 张背景块/COCO 非鸟整图近似，
   其上下文比「外扩后的 FP crop」小，不可替代。**本文只出正样本侧结论。**

用法：
    python tools/reject_gate_crop_check/check_reject_at_deploy_crop.py
    python tools/reject_gate_crop_check/check_reject_at_deploy_crop.py --max-n 2000
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

CKPT = r"E:\code\code tools\models\reject_mnv4_v1\best.pt"
BOXES = r"E:\code\code tools\data\crop_regime\boxes.jsonl"
DATA = r"E:\code\code tools\data"
OUT = os.path.join(ROOT, "data", "reject_crop_check")

FRAC = 2.0          # 脚本口径外扩比例；端上 BOX_EXPAND_RATIO = 2×frac = 4.0
PRE_SHORT = 256     # 端上：短边缩到 256，再中心裁 224

TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_model(dev):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    from train import build_model
    net = build_model(ck["model_name"], len(ck["species"])).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, {s: i for i, s in enumerate(ck["species"])}, ck


def expand(x0, y0, x1, y1, f, W, H):
    """四边各外扩 **frac × 边长**，**钳进画幅**；总宽 = w × (1 + 2×frac)。

    ⚠️ 口径铁律（初版踩过这个坑，见本文档 §2.2.1 的更正）：
      端上 `InferenceEngine.expand` 每侧加 `宽×ratio/2`，总宽 = w×(1+ratio)；
      评测脚本口径 frac 的换算是 **device BOX_EXPAND_RATIO = 2 × frac**
      （CROP-REGIME.md §2 换算铁律），故 frac=2.0 时 **device ratio=4.0、总宽=5w**。
      `crop_pixel_budget.py` 用的是同一个式子：f = 1 + 2*frac。
      **每侧加 宽×frac/2 是错的**（frac=2.0 时只到 3w，crop 比真机小一截）。
    """
    w, h = x1 - x0, y1 - y0
    k = 1.0 + 2.0 * f
    nx0, ny0 = max(0.0, x0 - w * f), max(0.0, y0 - h * f)
    nx1, ny1 = min(float(W), x1 + w * f), min(float(H), y1 + h * f)
    nx1 = max(nx1, nx0 + 1.0)
    ny1 = max(ny1, ny0 + 1.0)
    assert abs((nx1 - nx0) - w * k) < 1e-6 or nx0 == 0.0 or nx1 == float(W)
    return int(nx0), int(ny0), int(nx1), int(ny1)


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--frac", type=float, default=FRAC)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
    net, idx_of, ck = load_model(dev)
    print(f"拒识模型 {ck['model_name']} best_top1={ck['top1']*100:.2f}% species={ck['species']}\n")

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
    print(f"样本 {len(recs)} 条（五区各 ≤{per}），frac={a.frac} → 端上 BOX_EXPAND_RATIO={a.frac*2:.0f}")

    nnb = idx_of["__not_bird__"]
    whole, crop, meta = [], [], []
    miss = 0
    t0 = time.time()
    bw, bc = [], []          # 两路各一个缓冲：bw 收整图，bc 收部署 crop

    def push(buf, which, img):
        buf.append(TF(img))
        if len(buf) < a.batch:
            return
        with torch.no_grad():
            o = net(torch.stack(buf).to(dev))
            (whole if which == "w" else crop).extend(
                torch.softmax(o, 1)[:, nnb].tolist())
        buf.clear()

    for i, r in enumerate(recs):
        p = os.path.join(DATA, r["region"], "images", r["file"])
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                W, H = im.size
                push(bw, "w", im)                              # 整图（旧验收口径）
                x0, y0, x1, y1 = r["det"][1:]
                cx0, cy0, cx1, cy1 = expand(x0, y0, x1, y1, a.frac, W, H)
                push(bc, "c", im.crop((cx0, cy0, cx1, cy1)))    # 部署口径 crop（端上真实）
                meta.append((r["region"], r["file"], r["det"][0],
                             min(cx1 - cx0, cy1 - cy0)))
        except Exception:
            miss += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    for buf, which in ((bw, "w"), (bc, "c")):
        if buf:
            with torch.no_grad():
                o = net(torch.stack(buf).to(dev))
                (whole if which == "w" else crop).extend(
                    torch.softmax(o, 1)[:, nnb].tolist())
            buf.clear()
    assert len(whole) == len(crop) == len(meta), (len(whole), len(crop), len(meta))

    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(OUT, f"reject_deploy_crop_{stamp}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "ckpt": CKPT, "boxes": BOXES, "n": len(meta), "frac": a.frac,
            "device_ratio": a.frac * 2, "miss": miss,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "正样本侧（检出框 crop）测的是拒识门在部署裁剪口径下的真鸟误拒率；"
                    "负样本侧无法复原（原始视频不在盘上），未测。",
            "rows": [{"region": m[0], "file": m[1], "det_score": m[2], "crop_short": m[3],
                       "p_not_bird_whole": w, "p_not_bird_crop": c}
                      for m, w, c in zip(meta, whole, crop)],
        }, f, ensure_ascii=False)
    print(f"\n落盘 {out_json}（逐条分数，失败样本已剔除；miss={miss}）")

    print(f"\n{'阈值':>6}{'整图误拒率':>14}{'部署crop误拒率':>16}{'Δ':>10}")
    rows = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        w_rate = sum(1 for s in whole if s >= th) / len(whole)
        c_rate = sum(1 for s in crop if s >= th) / len(crop)
        rows.append((th, w_rate, c_rate))
        print(f"{th:>6.2f}{w_rate*100:>13.2f}%{c_rate*100:>15.2f}%{(c_rate-w_rate)*100:>+9.2f}pp")

    print(f"\nP(not_bird) 分布：")
    for name, xs in (("整图（旧验收口径）", whole), ("部署 crop（端上真实）", crop)):
        print(f"  {name:<22} 中位 {pct(xs,0.5):.3f}  p90 {pct(xs,0.9):.3f}  "
              f"p99 {pct(xs,0.99):.3f}  max {max(xs):.3f}")

    print("\n分区（阈值 0.8 = App 现役 REJECT_THRESHOLD）：")
    agg = defaultdict(lambda: [0, 0, 0])
    for (reg, _, _, _), w, c in zip(meta, whole, crop):
        agg[reg][0] += 1
        agg[reg][1] += (w >= 0.8)
        agg[reg][2] += (c >= 0.8)
    for reg in sorted(agg):
        n, wn, cn = agg[reg]
        print(f"  {reg:<10} n={n:<6} 整图 {wn/n*100:>6.2f}%   部署crop {cn/n*100:>6.2f}%")

    print("\n按裁剪短边分桶（阈值 0.8）——误拒是否集中在小 crop 上：")
    buckets = [(0, 64), (64, 128), (128, 192), (192, 256), (256, 10 ** 9)]
    print(f"  {'裁剪短边':<14}{'n':>7}{'占比':>9}{'整图误拒':>11}{'部署crop误拒':>15}")
    for lo, hi in buckets:
        idxs = [i for i, m in enumerate(meta) if lo <= m[3] < hi]
        if not idxs:
            continue
        n = len(idxs)
        w_rate = sum(1 for i in idxs if whole[i] >= 0.8) / n
        c_rate = sum(1 for i in idxs if crop[i] >= 0.8) / n
        rng = f"{lo}~{hi}px" if hi < 10 ** 9 else f"≥{lo}px"
        print(f"  {rng:<14}{n:>7}{n/len(meta)*100:>8.1f}%{w_rate*100:>10.2f}%{c_rate*100:>14.2f}%")

    small = [i for i, m in enumerate(meta) if m[3] < 128]
    if small:
        n = len(small)
        c_rate = sum(1 for i in small if crop[i] >= 0.8) / n
        rest = [i for i in range(len(meta)) if meta[i][3] >= 128]
        r_rate = sum(1 for i in rest if crop[i] >= 0.8) / max(1, len(rest))
        print(f"\n  → 裁剪短边 <128px 的 {n} 条（占 {n/len(meta)*100:.1f}%）误拒率 {c_rate*100:.2f}%；"
              f"其余 {len(rest)} 条 {r_rate*100:.2f}%")
        if c_rate > 3 * max(r_rate, 1e-4):
            print("  → **误拒几乎全部集中在小 crop 上**。小 crop 正是要放大很多倍才够 256 的那批，")
            print("    与 crop_pixel_budget.py 的「43%% 需放大 / 8%% 放大超 3 倍」是同一批样本。")

    print("\n最危险的样本（部署 crop 下 P(not_bird) 最高的 10 条，端上会被拒识门灭名）：")
    top = sorted(zip(crop, meta), key=lambda t: -t[0])[:10]
    for s, (reg, fn, ds, cs) in top:
        print(f"  {s:.3f}  {reg:<10} det={ds:.2f} 裁剪短边={cs:.0f}px  {fn}")

    th, w_rate, c_rate = next((r for r in rows if abs(r[0] - 0.8) < 1e-9), rows[-1])
    print("\n=== 结论 ===")
    print(f"阈值 0.8：整图口径误拒 {w_rate*100:.2f}%（旧验收 0.6%，不同样本，仅供量级参照）"
          f" / 部署 crop 口径 {c_rate*100:.2f}%")
    print(f"Δ = {(c_rate-w_rate)*100:+.2f}pp —— "
          + ("部署 crop **更难**被误拒，外扩没有带来额外误拒风险。"
             if c_rate <= w_rate else
             "部署 crop **更容易**被误拒，外扩带来了额外误拒风险，需要处置。"))
    print("负样本侧未测（原始视频不在盘上）—— FP 拒识率在部署口径下仍是未知数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
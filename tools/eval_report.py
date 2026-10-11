"""Evaluate a trained checkpoint, with and without geomodel reranking.

The point of this script is the comparison, not the absolute number: it reports
top-1/3/5 for the image model alone and after multiplying by the geomodel
occurrence probability for each image's own lat/lon and week-of-year.

Because GBIF records carry coordinates and dates, every validation image has a
real geo context -- no synthetic locations.
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset
from train import split_by_list  # 同一份切分口径，见 tools/freeze_val.py
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# geomodel 定位：包内布局用 <repo>/tools/geomodel（外包机器），主开发机回退到
# code tools 下的独立目录。硬编码单一路径会在另一侧必然失败。
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOMODEL = os.path.join(_REPO, "tools", "geomodel")
if not os.path.isdir(GEOMODEL):
    GEOMODEL = r"E:\code\code tools\geomodel"


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def week_of(datestr):
    """'YYYY-MM-DD' -> BirdNET-style 1..48 week (4 weeks per month)."""
    try:
        y, m, d = (int(x) for x in datestr.split("-"))
        return min(48, max(1, (m - 1) * 4 + (d - 1) // 7 + 1))
    except Exception:
        return 20


def week_of_row(r):
    """GBIF rows carry year/month only, so pin the month's midpoint (day 15).

    Reading a nonexistent 'date' column silently fell back to week 20 for every
    image, which reranks January photos with a mid-May species prior.
    """
    if r.get("date"):
        return week_of(r["date"])
    try:
        m = int(r["month"])
    except (KeyError, TypeError, ValueError):
        return 24
    return min(48, max(1, (m - 1) * 4 + 3))


class EvalSet(Dataset):
    def __init__(self, rows, labels, img_dir):
        self.rows, self.labels, self.img_dir = rows, labels, img_dir
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = Image.open(os.path.join(self.img_dir, r["file"])).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        return self.tf(img), self.labels[r["scientific_name"]]


def grouped_split(rows, val_frac=0.2, seed=0):
    import random
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["scientific_name"]][r.get("occurrence_key") or r["file"]].append(r)
    train, val = [], []
    rng = random.Random(seed)
    for sci, groups in by.items():
        keys = sorted(groups)
        rng.shuffle(keys)
        n = max(1, int(len(keys) * val_frac))
        for i, k in enumerate(keys):
            (val if i < n else train).extend(groups[k])
    return train, val


def build_model(ckpt):
    name = ckpt["model_name"]
    if name == "birdnet11m":
        import sys
        sys.path.insert(0, r"E:\code\code tools\bird-cn-373\code\Bird-Classification-main")
        from _BirdNet import BirdNet
        net = BirdNet(dims=(64, 128, 256, 512), layers=(3, 4, 6, 3),
                      class_num=len(ckpt["species"]))
    else:
        import timm
        net = timm.create_model(name, pretrained=False,
                                num_classes=len(ckpt["species"]))
    net.load_state_dict(ckpt["model"])
    return net.eval()


def prior_weight(G, alpha, scale):
    """geomodel 出现概率 -> 乘性权重。G 中的 NaN = 该图无坐标 / 该种无 geomodel 先验，
    权重取 1（不加分也不扣分），而不是退回索引 0 借到别的种的概率。

    prob：直接用绝对概率。整行同乘常数不影响行内排序，所以"量纲随地点变"本身无害；
          有害的是每行的**分散度**不同 —— alpha 的含义因此随区漂移，且大 alpha 近乎一票否决。
    rank：换成该地点内部的百分位秩（并列取平均秩），分散度被拉平，alpha 跨区稳定，
          代价是最低秩只能到 1/n，压不过 prob 的 1e-6 地板（实测峰值低于 prob，见 docs/handoff/root/RERANK-AB.md）。
    """
    W = np.ones_like(G)
    finite = np.isfinite(G)
    rows, cols = finite.any(axis=1), finite.any(axis=0)
    if not (rows.any() and cols.any()):
        return W
    sel = np.ix_(rows, cols)
    sub = G[sel]
    if scale == "prob":
        W[sel] = np.power(np.clip(sub, 1e-6, None), alpha)
    else:
        W[sel] = np.power(rankdata(sub, axis=1) / sub.shape[1], alpha)
    return W


def geo_mask(G, pct, min_keep=5):
    """按地点/季节把候选集缩小：每行只保留 geomodel 概率 >= 该行 pct 分位的类。

    等价于"临时把 150 类变成 150*(1-pct) 类"，不用重训。pct<=0 表示不掩码。
    两道保险：整行无坐标不掩；掩下来剩的类不足 min_keep 个也不掩（Top-5 填不满）。
    """
    if pct <= 0:
        return np.ones_like(G)
    # 整行无坐标时 nanquantile 会因 "All-NaN slice" 告警，值本身被后面的覆盖逻辑丢掉
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        thr = np.nanquantile(G, pct, axis=1, keepdims=True)
    # 无先验的种（整列 NaN）不参与掩码——不能因为"不知道"就把它排除
    M = np.where(~np.isfinite(G) | (G >= thr), 1.0, 0.0)
    # 无坐标的整行 NaN：thr 也是 NaN，不该被当成全 0
    M[~np.isfinite(G).any(axis=1)] = 1.0
    M[M.sum(axis=1) < min_keep] = 1.0
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--prior-scale", nargs="+", choices=["prob", "rank"], default=["prob"],
                    help="prob= geomodel 原始概率（默认，旧行为）；rank= 地点内百分位秩。"
                         "可一次传多个做 A/B")
    ap.add_argument("--val-list", default=None,
                    help="frozen_val.csv：用冻结的验证观察集，别按当前全量重切")
    ap.add_argument("--mask-pct", type=float, nargs="*", default=[0.0],
                    help="软掩码分位：把该行 geomodel 概率低于该分位的类排除出候选，"
                         "可传多个（0=不掩码）")
    ap.add_argument("--prior-table", default=None,
                    help="distill_prior_table.py 产出的分区×月份先验表(JSON)。"
                         "给了就用查表（docs/handoff/root/RERANK-DEVICE.md 方案 A），不再逐图跑 geomodel ONNX")
    ap.add_argument("--prior-lookup", choices=["month", "nearest"], default="month",
                    help="查表方式：month=只按图的月份取列（区中心/点均值，端上无 GPS 的"
                         "真实形态）；nearest=图坐标取最近采样点（模拟端上有 GPS）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    log(f"checkpoint {a.ckpt}")
    log(f"  model={ckpt['model_name']}  params={ckpt['params']/1e6:.2f}M  "
        f"species={len(ckpt['species'])}  训练时 top1={ckpt['top1']*100:.2f}%")

    img_dir = os.path.join(os.path.dirname(a.dataset), "images")
    rows = []
    with open(a.dataset, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if os.path.exists(os.path.join(img_dir, r["file"])):
                rows.append(r)
    species = ckpt["species"]
    labels = {s: i for i, s in enumerate(species)}
    if a.val_list:
        _, va, st = split_by_list(rows, a.val_list)
        log(f"用冻结验证集 {os.path.basename(a.val_list)}: 清单 {st['frozen']} 张，"
            f"本次命中 {st['present']} 张，清单里已不在 dataset.csv 的 {st['gone_from_dataset']} 张")
    else:
        _, va = grouped_split(rows, ckpt.get("train_cfg", {}).get("val_frac", 0.2))
    va = [r for r in va if r["scientific_name"] in labels]
    log(f"验证集 {len(va)} 张 / {len({r['scientific_name'] for r in va})} 种")

    net = build_model(ckpt).to(dev)
    ds = EvalSet(va, labels, img_dir)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=4)

    logits_all, labels_all = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(dev)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                o = net(x)
            logits_all.append(o.float().cpu().numpy())
            labels_all.append(y.numpy())
    L = np.concatenate(logits_all)
    Y = np.concatenate(labels_all)
    log(f"推理 {len(L)} 张 用时 {time.time()-t0:.1f}s")

    def topk(L, ks=(1, 3, 5)):
        o = np.argsort(-L, axis=1)
        return {k: float(np.mean([Y[i] in o[i, :k] for i in range(len(Y))]))
                for k in ks}

    def softmax(L):
        z = L - L.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    base_probs = softmax(L)
    res = {"checkpoint": a.ckpt, "model": ckpt["model_name"],
           "params": ckpt["params"], "n_species": len(species),
           "n_val": len(Y)}

    log("\n=== 图像模型（无重排）===")
    m = topk(L)
    res["image_only"] = m
    for k, v in m.items():
        log(f"  top{k} = {v*100:6.2f}%")

    # ---- geomodel rerank using each image's own coordinates ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    gs = ort.InferenceSession(
        os.path.join(GEOMODEL, "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx"),
        so, providers=["CPUExecutionProvider"])
    gname = gs.get_inputs()[0].name

    with open(os.path.join(GEOMODEL, "BirdNET+_Geomodel_V3.0.4_Global_14K_Labels.txt"),
              encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    gidx = {}
    for i, line in enumerate(lines):
        p = line.split("\t")
        if len(p) >= 2:
            gidx.setdefault(p[1].strip().lower(), i)

    # map our species -> geomodel output index。用 checkpoint 自带的 species 列表
    # 直接匹配，与具体区域清单解耦（旧版固定读华北清单，其他区域会整表回退索引 0）。
    sci_of_label = {}
    for s in species:
        j = gidx.get(s.lower())
        if j is not None:
            sci_of_label[s] = j
    missing = [s for s in species if s not in sci_of_label]
    if missing:
        log(f"  注意: {len(missing)} 种未匹配到 geomodel 索引 (重排时权重按 1，不借别人的概率)")

    # G = 每张图自己的 (lat, lon, week) 下各种的 geomodel 出现概率；NaN = 无先验
    G = np.full_like(base_probs, np.nan)
    if a.prior_table:
        # ---- 方案 A：查表（docs/handoff/root/RERANK-DEVICE.md）——类表与表逐项断言，同源同序 ----
        tj = json.load(open(a.prior_table, encoding="utf-8"))
        if list(tj["species"]) != list(species):
            sys.exit(f"先验表物种顺序与 checkpoint 不一致 "
                     f"({len(tj['species'])} vs {len(species)})——表与类表必须同源同序")
        pts = np.array(tj["points"], dtype=float)
        by_month = {int(m): np.array(v, dtype=float)   # [npts, ncls]
                    for m, v in tj["prior"].items()}
        n_geo = 0
        for i, r in enumerate(va):
            try:
                m = int(r["month"])
            except (ValueError, KeyError, TypeError):
                continue
            if m not in by_month:
                continue
            arr = by_month[m]
            if a.prior_lookup == "nearest" and len(pts) > 1:
                try:
                    la, lo = float(r["lat"]), float(r["lon"])
                    j = int(np.argmin((pts[:, 0] - la) ** 2 + (pts[:, 1] - lo) ** 2))
                except (ValueError, KeyError, TypeError):
                    j = 0
                G[i, :] = arr[j]
            else:                                       # month：端上无 GPS 的真实形态
                G[i, :] = arr.mean(axis=0)
            n_geo += 1
        log(f"  [查表] {os.path.basename(a.prior_table)} source={tj['source']} "
            f"grid={tj['grid']} lookup={a.prior_lookup} "
            f"→ {n_geo}/{len(va)} 张有先验")
    else:
        n_geo = 0
        for i, r in enumerate(va):
            try:
                lat = float(r["lat"])
                lon = float(r["lon"])
            except (ValueError, KeyError, TypeError):
                continue
            wk = week_of_row(r)
            p = gs.run(None, {gname: np.array([[lat, lon, wk]],
                                                 dtype=np.float32)})[0][0]
            for c, s in enumerate(species):
                j = sci_of_label.get(s)
                if j is not None:
                    G[i, c] = p[j]
            n_geo += 1
        log(f"  使用了 {n_geo}/{len(va)} 张的真实坐标做重排")

    res["geomodel"] = {}
    scores = {}
    log("\n=== geomodel 重排后 ===")
    for scale in a.prior_scale:
        for pct in a.mask_pct:
            M = geo_mask(G, pct)
            kept = M.sum(axis=1)
            # 真种被掩掉多少 —— 掩码之所以伤，是因为 geomodel 对"这张图真是这个种"
            # 经常给很低的概率，排除法直接把正确答案移出候选
            keep_true = float(M[np.arange(len(Y)), Y].mean()) if pct else 1.0
            for alpha in a.alphas:
                F = base_probs * prior_weight(G, alpha, scale) * M
                F = F / F.sum(1, keepdims=True)
                m = topk(F)
                key = (f"alpha={alpha}" if pct == 0 and scale == "prob"
                       else f"{scale}@{alpha}" + (f"|mask{pct}" if pct else ""))
                res["geomodel"][key] = m
                scores[key] = (scale, alpha, m, pct)
                log(f"  {scale:<5} alpha={alpha:<4} mask={pct:<5} "
                    f"(每行均留 {kept.mean():5.1f} 类, 真种留存 {keep_true*100:5.1f}%) "
                    f"top1={m[1]*100:6.2f}%  "
                    f"top3={m[3]*100:6.2f}%  top5={m[5]*100:6.2f}%  "
                    f"Δtop1 {(m[1]-res['image_only'][1])*100:+5.2f}  "
                    f"Δtop5 {(m[5]-res['image_only'][5])*100:+5.2f}")
            res.setdefault("mask_kept", {})[str(pct)] = [
                float(kept.mean()), float(np.percentile(kept, 10)), keep_true]

    # per-species delta at the best (scale, alpha, mask)
    best_key = max(scores, key=lambda k: scores[k][2][1])
    best_scale, best_a, _, best_pct = scores[best_key]
    F = base_probs * prior_weight(G, best_a, best_scale) * geo_mask(G, best_pct)
    order_b = np.argsort(-L, axis=1)
    order_f = np.argsort(-F, axis=1)
    per = defaultdict(lambda: [0, 0, 0])
    for i in range(len(Y)):
        s = va[i]["scientific_name"]
        per[s][0] += 1
        per[s][1] += int(Y[i] in order_b[i, :1])
        per[s][2] += int(Y[i] in order_f[i, :1])
    rows_out = sorted(per.items(), key=lambda x: x[1][2] - x[1][1])
    log(f"\n=== 逐种 top1 变化 (best {best_scale} alpha={best_a} mask={best_pct}) ===")
    log(f"  {'species':<26}{'中文':<10}{'n':>4}{'前':>7}{'后':>7}{'Δ':>7}")
    gain, loss = [], []
    for s, (n, b, f) in rows_out:
        d = f - b
        zh = next((r["common_name"] for r in va if r["scientific_name"] == s), "")
        log(f"  {s:<26}{zh:<10}{n:>4}{b/n*100:6.1f}%{f/n*100:6.1f}%{(f-b)/n*100:+6.1f}")
        (gain if d > 0 else loss if d < 0 else []).append((s, zh, d))
    if gain:
        log("\n  受益最大的种:")
        for s, zh, d in sorted(gain, key=lambda x: -x[2])[:8]:
            log(f"     {s:<26}{zh:<10}{d:+d} 张")
    if loss:
        log("\n  受损最大的种:")
        for s, zh, d in sorted(loss, key=lambda x: x[2])[:8]:
            log(f"     {s:<26}{zh:<10}{d:+d} 张")
    res["best_alpha"] = best_a
    res["best_prior_scale"] = best_scale
    res["best_mask_pct"] = best_pct
    res["per_species"] = {s: {"n": v[0], "image_only_top1": v[1],
                              "rerank_top1": v[2]} for s, v in per.items()}

    out = a.out or os.path.splitext(a.ckpt)[0] + "_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"\n-> {out}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
r"""全国兜底 v2（384 类）验收评测：与 cn373 基线同题对比。

背景：v2 训练完成（models/national_mnv4_v1/best.pt，ep28 val top1 61.98%，
384 类/限流后 122,003 张）。但 v2 的 val 集与 cn373 验收线（c01d8b8：top1
59.25%/top3 69.67%/top5 73.69%，n=2265）不是同一批图——验收必须同题：

  A) 基线同题复现：--no-gap 重建 244 类 era 的 merged CSV（五区源文件未动，合并
     与 grouped_split 均确定性），按基线脚本口径【不限流】直切 val(29,409)，复现其
     抽样集（cap 2500/seed 3，应得 ~2265 张）。cn373 与 v2 各跑一遍同一批图；
     cn373 结果须落在记录值 ±0.5pp 内，否则视为划分未复现、结论作废。
  B) 新题（含缺口种）：384 类 era 同样不限流直切 val(35,886)，同抽样法抽 ~2500 张
     跑 cn373/v2 对比，子集分五区并集/gap 种分开报。
     gap 140 种全部在 labels_birdnet373.txt 前 373 行内，两模型同题可测。
  C) v2 训练曲线复现：注意 train.py 是【先限流(500/种)再划分】——训练 best 61.98%
     对应的是限流后 val(24,275)，与 A/B 口径不同批图，单独验证。
  D) 验收同题（正口径）：从限流后真 val 分层抽样，两模型逐图同题，v2 零泄漏。
     ⚠️ 泄漏教训（首轮实跑发现）：A/B 用未限流划分抽样，而 train.py 先限流再划分——
     抽样集里落在 capped train 的图对 v2 是见过题（实测 v2 在 A 题"75.36%"即此虚高）。
     cn373 系外部模型无泄漏，其记录线 59.25% 本身仍有效。验收结论以 D 为准。

注：v2 驱动的心跳镜像把 data/national/dataset.csv 覆写成了缺口补抓文件（32,637 行），
本脚本先重跑合并恢复（副作用：监控台 national 数据也恢复为合并口径，属预期终态）。

产出（models/national_v2_eval/）：summary.json、per_species_v2_fullval.csv、
cn373_cache.jsonl / v2_cache.jsonl（断点续跑缓存，换 checkpoint 或改口径时删除重测）。
publish 步骤把 D 两个单点写成 models/national_v2_head2head*/history.json
进监控台历史对比页（目录名须以地区前缀 national 开头，见 train_dashboard.collect_runs）。

Usage: python -u eval_national_v2.py [--cap 2500] [--threads 8] [--dry]
  --dry 只跑恢复+复现校验+v2 GPU 全量（跳过 cn373 慢线），先确认 n=2265 复现。
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms as T

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, r"E:\code\observebird\tools")
from train import grouped_split  # noqa: E402  与训练同一划分函数/种子

ROOT = r"E:\code\observebird"
DATA = r"E:\code\code tools\data"
PY = sys.executable
MERGE = os.path.join(ROOT, "tools", "national_clf", "build_national_merged.py")
NAT_CSV = os.path.join(DATA, "national", "dataset.csv")
EVAL244_CSV = os.path.join(DATA, "national_eval244", "dataset.csv")
BEST_PT = r"E:\code\code tools\models\national_mnv4_v1\best.pt"
CN373_TFLITE = os.path.join(ROOT, r"android\app\src\main\assets\models\birdnet_373_float32.tflite")
LABELS373 = os.path.join(ROOT, r"android\app\src\main\assets\models\labels_birdnet373.txt")
GAP_SPECIES = os.path.join(ROOT, "tools", "national_clf", "species_gap.csv")
OUT_DIR = r"E:\code\code tools\models\national_v2_eval"
REGIONS = ["huabei", "huazhong", "huanan_sw", "huadong", "dongbei"]
BASELINE_RECORDED = (0.5925, 0.6967, 0.7369, 2265)  # c01d8b8 记录的 cn373 验收线
TRAIN_RECORDED = 0.6198  # v2 训练曲线 best top1（train.log ep28，限流后口径）

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def die(m):
    log("FATAL: " + m)
    sys.exit(1)


def restore_datasets():
    """重跑合并：恢复被心跳镜像覆写的 data/national/dataset.csv，
    另建 244-era 副本 national_eval244/dataset.csv（--no-gap）。"""
    for extra, tag in ((["--no-gap"], "244-era 副本"), ([], "v2 全量")):
        out = EVAL244_CSV if extra else NAT_CSV
        cmd = [PY, "-u", MERGE, "--out", os.path.dirname(out)] + extra
        log(f"恢复合并数据集（{tag or 'v2 全量'}）…")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = (r.stdout or "").strip().splitlines()[-2:]
        for t_ in tail:
            log("  " + t_)
        if r.returncode != 0:
            die(f"merge 失败 exit={r.returncode}\n{r.stderr}")
    return load_rows(NAT_CSV), load_rows(EVAL244_CSV)


def load_rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def stratified_sample(va, sp2idx, cap, seed=3):
    """与 eval_cn373_baseline.py 完全一致的抽样（同顺序/同 rng）。"""
    by_sp = defaultdict(list)
    for r in va:
        if sp2idx.get(r["scientific_name"], -1) >= 0:
            by_sp[r["scientific_name"]].append(r)
    rng = random.Random(seed)
    per = max(1, cap // max(1, len(by_sp)))
    sample = []
    for s in sorted(by_sp):
        rs = by_sp[s]
        rng.shuffle(rs)
        sample += [(s, r) for r in rs[:per]]
    return by_sp, sample


def cap_rows(rows, mx=500):
    """复刻 train.py 的 --max-per-species：按文件序每种种留前 mx 张（划分前限流）。
    注意训练 val 是从限流行切出来的，与基线脚本（不限流直切）不是同一批图——
    A/B 用不限流口径对齐 cn373 记录线，C 用限流口径复现训练曲线。"""
    keep = defaultdict(int)
    out = []
    for r in rows:
        s = r["scientific_name"]
        if keep[s] < mx:
            keep[s] += 1
            out.append(r)
    return out


def build_sp2idx():
    """cn373 类下标：五区 dataset 的 birdnet_index 列（基线口径）优先；
    gap 种取自 labels_birdnet373.txt 行号。两源重叠处校验一致性。"""
    sp2idx = {}
    for region in REGIONS:
        for r in csv.DictReader(open(os.path.join(DATA, region, "dataset.csv"), encoding="utf-8")):
            s = r["scientific_name"]
            if s not in sp2idx:
                try:
                    sp2idx[s] = int(r["birdnet_index"])
                except (ValueError, KeyError):
                    sp2idx[s] = -1
    file_idx = {}
    for i, line in enumerate(l.strip() for l in open(LABELS373, encoding="utf-8") if l.strip()):
        file_idx[line.split("\t")[0]] = i
    gap = {r["scientific_name"] for r in csv.DictReader(open(GAP_SPECIES, encoding="utf-8"))}
    mism = [s for s in sp2idx if sp2idx[s] >= 0 and s in file_idx and sp2idx[s] != file_idx[s]]
    if mism:
        die(f"birdnet_index 列与 labels_birdnet373 行号不一致 {len(mism)} 种：{mism[:5]}")
    for s in gap:
        if s not in file_idx:
            die(f"gap 种 {s} 不在 cn373 标签表")
        sp2idx.setdefault(s, file_idx[s])
    log(f"sp2idx: 五区映射可评 {sum(1 for v in sp2idx.values() if v >= 0 and v < 373)} 种"
        f"（含 gap 补 {len(gap)}），labels 表 {len(file_idx)} 行")
    return sp2idx, gap, file_idx


# ---------- v2 (torch, GPU) ----------

class _V2Set(torch.utils.data.Dataset):
    """items: [(绝对路径, 类下标)]。解码/增广在 DataLoader worker 里做。"""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, li = self.items[i]
        return i, TF(Image.open(p).convert("RGB")), li


@torch.no_grad()
def eval_v2(rows, bs=128):
    """返回 path -> {"gt":种名, "r":gt 名次(1-5，>5 记 6), "t1":top1 种名}。
    逐图缓存 v2_cache.jsonl（换 checkpoint 后请删除缓存文件再跑）。"""
    cache_p = os.path.join(OUT_DIR, "v2_cache.jsonl")
    res = {}
    if os.path.exists(cache_p):
        with open(cache_p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                res[d["p"]] = d
        log(f"  v2 缓存命中 {len(res)} 张")
    todo = [r for r in rows if os.path.normpath(r["file"]) not in res]
    if not todo:
        log("  v2 全部命中缓存")
        return res
    ckpt = torch.load(BEST_PT, map_location="cpu")
    species, labels = ckpt["species"], ckpt["labels"]
    net = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(species))
    net.load_state_dict(ckpt["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev).eval()
    inv = {i: s for s, i in labels.items()}
    all_items = [(os.path.normpath(r["file"]), labels[r["scientific_name"]]) for r in todo]
    items = [(p, li) for p, li in all_items if os.path.exists(p)]
    if len(items) < len(all_items):
        log(f"  v2 缺图跳过 {len(all_items) - len(items)} 张")
    dl = torch.utils.data.DataLoader(_V2Set(items), batch_size=bs, num_workers=4)
    t0 = time.time()
    with open(cache_p, "a", encoding="utf-8") as cf:
        for n, (idxs, xs, lis) in enumerate(dl, 1):
            with torch.autocast("cuda", enabled=(dev.type == "cuda")):  # 与训练 val 同口径
                logits = net(xs.to(dev, non_blocking=True)).float()
            top5 = logits.argsort(dim=1, descending=True)[:, :5].cpu().numpy()
            for i, li, row5 in zip(idxs.tolist(), lis.tolist(), top5):
                rank = 6
                for k, cls in enumerate(row5):
                    if cls == li:
                        rank = k + 1
                        break
                d = {"p": items[i][0], "gt": inv[li], "r": int(rank),
                     "t1": inv[int(row5[0])]}
                res[d["p"]] = d
                cf.write(json.dumps(d, ensure_ascii=False) + "\n")
            cf.flush()
            if n % 40 == 0:
                done = n * bs
                eta = (len(items) - done) / max(1e-9, done / (time.time() - t0))
                log(f"  v2 {done}/{len(items)}（余约 {eta/60:.1f} 分）")
    log(f"  v2 完成：新测 {len(items)} 张（GPU {dev}）")
    return res


# ---------- cn373 (litert, CPU) ----------

def eval_cn373(items, sp2idx, threads):
    """items: [(path, species)] 去重后的并集。jsonl 缓存断点续跑。
    返回 path -> {"gt","r","t1"}。"""
    from ai_edge_litert.interpreter import Interpreter
    cache_p = os.path.join(OUT_DIR, "cn373_cache.jsonl")
    res = {}
    if os.path.exists(cache_p):
        with open(cache_p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                res[d["p"]] = d
        log(f"  cn373 缓存命中 {len(res)} 张")
    todo = [(p, s) for p, s in items if p not in res]
    if not todo:
        log("  cn373 全部命中缓存")
        return res
    it = Interpreter(model_path=CN373_TFLITE, num_threads=threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    inv373 = [l.split("\t")[0] for l in
              (x.strip() for x in open(LABELS373, encoding="utf-8")) if l][:373]
    n_skipped = 0
    t0 = time.time()
    with open(cache_p, "a", encoding="utf-8") as cf:
        for n, (p, s) in enumerate(todo, 1):
            gt = sp2idx[s]
            try:
                img = Image.open(p).convert("RGB")
            except OSError:
                n_skipped += 1
                continue
            x = (TF(img).numpy().transpose(1, 2, 0)[None]).astype(np.float32)
            it.set_tensor(inp["index"], x)
            it.invoke()
            logits = it.get_tensor(out["index"])[0][:373]
            top5 = np.argsort(logits)[::-1][:5]
            rank = 6
            for k, cls in enumerate(top5):
                if cls == gt:
                    rank = int(k) + 1
                    break
            d = {"p": p, "gt": s, "r": int(rank), "t1": inv373[int(top5[0])]}
            res[p] = d
            cf.write(json.dumps(d, ensure_ascii=False) + "\n")
            cf.flush()
            if n % 300 == 0:
                eta = (len(todo) - n) / max(1e-9, n / (time.time() - t0))
                log(f"  cn373 {n}/{len(todo)}（余约 {eta/60:.1f} 分）")
    log(f"  cn373 完成：新测 {len(todo)-n_skipped} 张，缺图 {n_skipped}")
    return res


def agg(records):
    """records: 同题两模型的 [{r}...] 之一。返回 (top1,top3,top5,n)。"""
    n = len(records)
    if not n:
        return (0.0, 0.0, 0.0, 0)
    r1 = sum(x["r"] == 1 for x in records)
    r3 = sum(x["r"] <= 3 for x in records)
    r5 = sum(x["r"] <= 5 for x in records)
    return (r1 / n, r3 / n, r5 / n, n)


def fmt(t):
    return f"top1 {t[0]*100:5.2f}%  top3 {t[1]*100:5.2f}%  top5 {t[2]*100:5.2f}%  (n={t[3]})"


def publish_to_dashboard(summary):
    """把 D 口径（验收同题：v2 真 val 抽样、两模型逐图同题、零泄漏）写成单点 run
    进监控台历史对比页。机制（train_dashboard.collect_runs）：扫 models_root/*/history.json，
    目录名以地区前缀开头才进该区对比页——national_* 才显示在 national 页。
    格式对齐 national_cn373_baseline（epoch=1 单点，args.note 记口径）。"""
    d = summary["D_head2head"]
    models_root = os.path.dirname(os.path.normpath(OUT_DIR))
    proto = (f"D 验收同题：两模型逐图同答 v2 真 val（限流后划分）分层抽样 {d['n']} 张"
             f"（cn373 可评种、cap2500/seed3），v2 零泄漏。"
             f"分桶与泄漏诊断见 {os.path.join(OUT_DIR, 'summary.json')}")
    entries = [
        ("national_v2_head2head", d["v2"]["all"], 2980000,
         BEST_PT + "（mobilenetv4_conv_small，384 类=五区并集244+缺口补抓140）",
         proto),
        ("national_v2_head2head_cn373", d["cn373"]["all"], 11200000,
         "cn373 预装兜底（BirdNet ConvNeXt 11.2M）同题参考线",
         proto),
    ]
    for name, acc, params, modelname, note in entries:
        d2 = os.path.join(models_root, name)
        os.makedirs(d2, exist_ok=True)
        hist = {
            "args": {"model": modelname, "note": note},
            "history": [{"epoch": 1, "top1": acc[0], "top3": acc[1], "top5": acc[2],
                         "note": "单点补评（验收同题口径，非训练曲线）"}],
            "best_top1": acc[0], "params": params,
            "n_images": d["n"], "n_species": d.get("n_species", 368),
        }
        p = os.path.join(d2, "history.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=1)
        log(f"发布 → {p}  (top1 {acc[0]*100:.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--dry", action="store_true", help="只跑恢复/复现校验/v2 全量，跳过 cn373")
    ap.add_argument("--publish-only", action="store_true",
                    help="不评测，读已有 summary.json 发布历史对比页单点条目")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if a.publish_only:
        sp = os.path.join(OUT_DIR, "summary.json")
        if not os.path.exists(sp):
            die(f"{sp} 不存在——先跑完整评测")
        publish_to_dashboard(json.load(open(sp, encoding="utf-8")))
        return 0

    rows384, rows244 = restore_datasets()
    sp384 = {r["scientific_name"] for r in rows384}
    sp244 = {r["scientific_name"] for r in rows244}
    assert (len(rows244), len(sp244)) == (147978, 244), \
        "244-era 合并未复现——五区源动了，同题对比不成立"
    assert (len(rows384), len(sp384)) == (180615, 384), \
        "384-era 合并未复现——缺口补抓源动了"
    # A/B 口径：不限流直切（与 eval_cn373_baseline.py 一致，对齐 cn373 记录线）
    _, va384 = grouped_split(rows384, val_frac=0.2, seed=0)
    _, va244 = grouped_split(rows244, val_frac=0.2, seed=0)
    # C 口径：与 train.py 相同——先限流再划分（须对上 train.log 的 122,003/24,275）
    rows384c = cap_rows(rows384)
    _, va384c = grouped_split(rows384c, val_frac=0.2, seed=0)
    assert (len(rows384c), len(va384c)) == (122003, 24275), \
        f"训练 val 未复现（限流后 {len(rows384c)} / val {len(va384c)} ≠ 122003/24275）"
    log(f"复现校验：244-era {len(rows244)} 张 val={len(va244)}；"
        f"384-era {len(rows384)} 张 val={len(va384)}；限流后 val={len(va384c)}")

    sp2idx, gap, _ = build_sp2idx()
    by244, samp244 = stratified_sample(va244, sp2idx, a.cap)
    by384, samp384 = stratified_sample(va384, sp2idx, a.cap)
    byD, sampD = stratified_sample(va384c, sp2idx, a.cap)
    # 泄漏集：capped train 的图。A/B 抽样出自未限流划分——v2 训练用的是先限流再划分，
    # 两者交集里落在 capped train 的图对 v2 是见过题（cn373 无此问题）。
    # D（限流后真 val 抽样）按构造零泄漏，才是同题验收的正口径。
    trainset = ({os.path.normpath(r["file"]) for r in rows384c}
                - {os.path.normpath(r["file"]) for r in va384c})
    log(f"A) 基线同题抽样：可评 {len(by244)}/244 种，n={len(samp244)}（记录值 2265）")
    log(f"B) 新题抽样（不限流划分）：可评 {len(by384)}/384 种，n={len(samp384)}")
    log(f"D) 验收同题抽样（限流后真 val）：可评 {len(byD)}/384 种，n={len(sampD)}")
    if abs(len(samp244) - BASELINE_RECORDED[3]) > 30:
        die(f"抽样集 n 与基线偏差过大（{len(samp244)} vs 2265）——同题不成立")

    # v2：全量 384-era val（基线/训练两口径都要覆盖）+ 244-era val（基线同题用）
    uniq = {}
    for r in va384:
        uniq[os.path.normpath(r["file"])] = r
    for r in va384c:
        uniq.setdefault(os.path.normpath(r["file"]), r)
    for r in va244:
        uniq.setdefault(os.path.normpath(r["file"]), r)
    v2 = eval_v2(list(uniq.values()))

    # cn373：三抽样集并集（缓存增量，只测新图）
    cn_items = sorted({(os.path.normpath(r["file"]), s) for s, r in samp244}
                      | {(os.path.normpath(r["file"]), s) for s, r in samp384}
                      | {(os.path.normpath(r["file"]), s) for s, r in sampD})
    if a.dry:
        log(f"--dry：跳过 cn373（待测 {len(cn_items)} 张）。v2 训练口径全量 val 指标：")
        full = [v2[os.path.normpath(r["file"])] for r in va384c
                if os.path.normpath(r["file"]) in v2]
        t = agg(full)
        log(f"  全体 {fmt(t)}  [训练记录 61.98%，复现偏差 {abs(t[0]-TRAIN_RECORDED)*100:.2f}pp]")
        gsub = [x for x in full if x["gt"] in gap]
        log(f"  gap 子集 {fmt(agg(gsub))}")
        log(f"  并集子集 {fmt(agg([x for x in full if x['gt'] not in gap]))}")
        return 0
    cn = eval_cn373(cn_items, sp2idx, a.threads)

    def pair(sample, clean=False):
        """clean=True：剔除 v2 泄漏图（落在 capped train 的），两模型同题剔除。"""
        recs_cn, recs_v2, n_leak = [], [], 0
        for s, r in sample:
            p = os.path.normpath(r["file"])
            if clean and p in trainset:
                n_leak += 1
                continue
            if p in cn and p in v2:
                recs_cn.append(cn[p])
                recs_v2.append(v2[p])
        return recs_cn, recs_v2, n_leak

    def subset(recs, cond):
        return agg([x for x in recs if cond(x)])

    # A) 基线同题（cn373 记录线复现门；v2 在此题有泄漏，只作诊断不作验收）
    a_cn, a_v2, _ = pair(samp244)
    tc, tv = agg(a_cn), agg(a_v2)
    drift = abs(tc[0] - BASELINE_RECORDED[0]) * 100
    log(f"A) 基线同题（244-era 不限流抽样，n={tv[3]}）")
    log(f"   cn373 : {fmt(tc)}  [记录线 {BASELINE_RECORDED[0]*100:.2f}%，"
        f"复现偏差 {drift:.2f}pp]")
    if drift > 0.5:
        die("cn373 未复现基线记录值——划分/抽样未对齐，结论作废")
    ac_cn, ac_v2, a_leak = pair(samp244, clean=True)
    log(f"   （A2 去泄漏：剔 {a_leak} 张 v2 训练已见图，剩 n={agg(ac_v2)[3]}）")
    log(f"    cn373 : {fmt(agg(ac_cn))}")
    log(f"    v2    : {fmt(agg(ac_v2))}  [含泄漏原样为 {fmt(tv)}——虚高，勿引]")

    # B) 新题不限流抽样（同样含 v2 泄漏：不限流/限流两划分 val 成员本就不同批，
    #    图落 capped train 即为见过题——泄漏与补抓量无关，A/B 只作诊断，验收看 D）
    b_cn, b_v2, _ = pair(samp384)
    log(f"B) 新题抽样（不限流，n={agg(b_v2)[3]}）")
    for tag, recs in (("cn373", b_cn), ("v2", b_v2)):
        log(f"   {tag} 全体     : {fmt(agg(recs))}")
        log(f"   {tag} 并集244 : {fmt(subset(recs, lambda x: x['gt'] not in gap))}")
        log(f"   {tag} gap140  : {fmt(subset(recs, lambda x: x['gt'] in gap))}")
    bc_cn, bc_v2, b_leak = pair(samp384, clean=True)
    log(f"   （B 去泄漏：剔 {b_leak} 张，剩 n={agg(bc_v2)[3]}）")
    log(f"    cn373 gap : {fmt(subset(bc_cn, lambda x: x['gt'] in gap))}    "
        f"v2 gap : {fmt(subset(bc_v2, lambda x: x['gt'] in gap))}")
    b_leak_gap = sum(1 for s, r in samp384 if r["scientific_name"] in gap
                     and os.path.normpath(r["file"]) in trainset)
    log(f"    （B gap 图中 {b_leak_gap} 张在泄漏集——不限流/限流两划分的 val 成员本就不同批，"
        f"gap 桶也不可信，验收只看 D）")

    # D) 验收同题：限流后真 val 抽样，两模型同题、v2 零泄漏（按构造）
    d_cn, d_v2, d_leak = pair(sampD, clean=True)
    if d_leak:
        die(f"D 抽样出现泄漏 {d_leak} 张——口径理解有误，停跑核查")
    tcD, tvD = agg(d_cn), agg(d_v2)
    log(f"D) 验收同题（v2 真 val 抽样，n={tvD[3]}）")
    log(f"   cn373 : {fmt(tcD)}    union {fmt(subset(d_cn, lambda x: x['gt'] not in gap))}    "
        f"gap {fmt(subset(d_cn, lambda x: x['gt'] in gap))}")
    log(f"   v2    : {fmt(tvD)}    union {fmt(subset(d_v2, lambda x: x['gt'] not in gap))}    "
        f"gap {fmt(subset(d_v2, lambda x: x['gt'] in gap))}")

    # C) v2 全量 val（n=24,275，训练曲线同口径：先限流再划分）
    full = [v2[os.path.normpath(r["file"])] for r in va384c
            if os.path.normpath(r["file"]) in v2]
    c_all, c_gap = agg(full), agg([x for x in full if x["gt"] in gap])
    c_uni = agg([x for x in full if x["gt"] not in gap])
    log(f"C) v2 全量 val：全体 {fmt(c_all)}  [训练记录 top1 61.98%，"
        f"复现偏差 {abs(c_all[0]-TRAIN_RECORDED)*100:.2f}pp]")
    log(f"   并集244 {fmt(c_uni)}    gap140 {fmt(c_gap)}")

    # 逐种明细（全量 val 上 v2；cn373 只有抽样数，标 n 供参考）
    per = defaultdict(lambda: {"v2_n": 0, "v2_r1": 0, "cn_n": 0, "cn_r1": 0})
    for x in full:
        d = per[x["gt"]]
        d["v2_n"] += 1
        d["v2_r1"] += x["r"] == 1
    for p, d in cn.items():
        e = per[d["gt"]]
        e["cn_n"] += 1
        e["cn_r1"] += d["r"] == 1
    csv_p = os.path.join(OUT_DIR, "per_species_v2_fullval.csv")
    with open(csv_p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scientific_name", "group", "v2_val_n", "v2_top1", "cn373_sample_n", "cn373_top1"])
        for s in sorted(per):
            d = per[s]
            w.writerow([s, "gap" if s in gap else "union", d["v2_n"],
                        f"{d['v2_r1']/max(1,d['v2_n']):.4f}", d["cn_n"],
                        f"{d['cn_r1']/max(1,d['cn_n']):.4f}" if d["cn_n"] else ""])
    summary = {
        "protocol": "A=基线抽样复现(含 v2 泄漏诊断) B=新题不限流抽样(同含泄漏,gap 桶基本干净) "
                    "C=v2 全量限流后 val D=验收同题(真 val 抽样,零泄漏,正口径)",
        "leak_diag": {"capped_train_n": len(trainset), "A_excluded": a_leak,
                      "B_excluded": b_leak, "B_gap_excluded": b_leak_gap},
        "A_baseline_head2head": {"n": tv[3], "cn373": tc[:3], "v2": tv[:3],
                                  "cn373_recorded": BASELINE_RECORDED[:3],
                                  "v2_clean": ac_v2[:3], "n_clean": agg(ac_v2)[3]},
        "B_new_task_sample": {"n": agg(b_v2)[3], "cn373": agg(b_cn)[:3], "v2": agg(b_v2)[:3],
                              "v2_gap": subset(b_v2, lambda x: x["gt"] in gap)[:3],
                              "cn373_gap": subset(b_cn, lambda x: x["gt"] in gap)[:3]},
        "C_v2_full_val": {"all": c_all[:3], "union244": c_uni[:3], "gap140": c_gap[:3],
                           "n": c_all[3], "train_recorded_top1": TRAIN_RECORDED},
        "D_head2head": {"n": tvD[3], "n_species": len(byD),
                         "cn373": {"all": tcD[:3],
                                    "union": subset(d_cn, lambda x: x['gt'] not in gap)[:3],
                                    "gap": subset(d_cn, lambda x: x['gt'] in gap)[:3]},
                         "v2": {"all": tvD[:3],
                                 "union": subset(d_v2, lambda x: x['gt'] not in gap)[:3],
                                 "gap": subset(d_v2, lambda x: x['gt'] in gap)[:3]}},
        "model": BEST_PT, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    log(f"→ {OUT_DIR}\\summary.json / per_species_v2_fullval.csv")
    publish_to_dashboard(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

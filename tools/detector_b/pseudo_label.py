# -*- coding: utf-8 -*-
"""B 路线伪标签生产（docs/handoff/detector_b/HANDOFF.md §3 阶段 1 的本机版）。

现役 yolox_s_512_dec.onnx（COCO 80 类）在华北网图上打框，四条自举盲区对策
（主文档 §3 阶段 1，缺一不可）：
  ① 低阈值 0.2 落盘 —— 训练标签要的是框位置不是置信度，低分框人审后再定去留
  ② 分类器交叉验证 —— 每个伪框 crop 喂华北分类器（best.pt），top1≥0.3 保留、
     <0.3 标 low_conf（不删，人审看）
  ③ 难例挖掘 —— 全帧 0 检出的图跑 centerRescan(0.5)：中心 1/2 区域 1:1 裁剪重检
     （等效 2× 变焦，复刻 ThreeStagePipeline.centerRescan），检出框加偏移映射回全帧
  ④ COCO bird tiny/small 过采样 —— 数据侧由 make_coco_bird_subset.py +
     make_train_json.py 负责，不在本脚本

输出 OUT = E:\\code\\code tools\\data\\detector_b\\pseudo_label\\：
  labels/<stem>.txt    YOLO txt（class 0，cx cy w h 归一化），含 rescan 回映射框
  pseudo_dets.jsonl    每图一行：file,w,h,dets[{score,x1,y1,x2,y2,src,clf_label,clf_prob,low_conf}]
  stats.json           框数 / 难例 / 低置信 / 分类器分布
  hard_examples.txt    难例（rescan 命中）图绝对路径 —— 量化校准集用（make_calib_b.py 读它）
  quality/*.jpg        抽检样张：分类器每类 top2 + 低分样张 + 难例样张 + 全帧 0 检出样张

断点续传：labels/<stem>.txt 已存在的图直接跳过。jsonl 是 append 写——若清空 labels/ 重跑，
须同时删掉旧 jsonl，否则 make_train_json.py 读到重复行（按 file 去重、后行覆盖，能兜底）。

推理口径与 bench_detector_a.py 完全一致：letterbox(114) + BGR 0-255 不归一化 + NMS 0.45。

Usage:
  python pseudo_label.py --limit 40    # 冒烟
  python pseudo_label.py               # 全量 ~33k 图（CPU 8 线程约 1.5h）
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

TOOLS = r"E:\code\code tools"
OBS_TOOLS = r"E:\code\observebird\tools"
DATA = os.path.join(TOOLS, "data", "huabei")
OUT = os.path.join(TOOLS, "data", "detector_b", "pseudo_label")
DET_ONNX = "yolox_s_512_dec.onnx"  # YoloxRunner 按 detectors/yolox/ 下的文件名取
CLF_CKPT = os.path.join(TOOLS, "models", "huabei_mnv4_rec", "best.pt")

STORE_TH = 0.20       # ① 低阈值落盘（= bench_detector_a.STORE_TH）
CLF_KEEP_TH = 0.30    # ② 分类器验证线：top1≥0.3 保留，否则 low_conf
CLF_MARGIN = 1.0      # ② crop 前框外扩倍数（每侧加 1×框宽/高）。实测关键：
                      #   紧框特写是分类器的 OOD（一致率 10%、p≈0.2），外扩后鸟占画面
                      #   ~1/3 ≈ 训练分布，一致率 55%、p≈0.60（对齐其 val top1）
RESCAN_REGION = 0.5   # ③ 中心区域边长占比（= App centerRescan(0.5)）
MIN_BOX_PX = 4        # 退化框剔除

sys.path.insert(0, OBS_TOOLS)
from bench_detector_a import YoloxRunner  # noqa: E402  letterbox114+BGR0-255+NMS0.45 同口径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _progress import update as prog_update  # noqa: E402  progress.json（监控台读）


# ---------------------------------------------------------------- 分类器交叉验证（对策②）

class CropClassifier:
    """华北分类器（train.py 产物 best.pt）给伪框验真。预处理 = train.py tf_val 口径
    （Resize 短边 256 + CenterCrop 224 + RGB + ImageNet 归一化）。"""

    def __init__(self, ckpt_path):
        import torch
        import timm
        torch.set_num_threads(8)
        self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.labels = list(ck["labels"])
        arch = ck.get("model_name", "mobilenetv4_conv_small.e2400_r224_in1k")
        self.net = timm.create_model(arch, num_classes=len(self.labels))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        print(f"分类器: {arch} / {len(self.labels)} 类 ← {os.path.basename(ckpt_path)}")

    def _prep(self, bgr):
        h, w = bgr.shape[:2]
        if h < 2 or w < 2:
            return None
        s = min(256.0 / min(h, w), 640.0 / max(h, w))  # 上限只拦极端长条 crop
        nh, nw = max(224, round(h * s)), max(224, round(w * s))
        r = cv2.resize(bgr, (nw, nh))
        t, l = (nh - 224) // 2, (nw - 224) // 2
        c = r[t:t + 224, l:l + 224].astype(np.float32) / 255.0
        c = (c[:, :, ::-1] - self.mean) / self.std  # BGR→RGB + 归一化
        return c.transpose(2, 0, 1)

    def predict(self, crops):
        """输入 BGR crop 列表 → [(label, prob), ...]。空/坏 crop 给 ("", 0.0)。"""
        out = []
        xs, idxs = [], []
        for i, c in enumerate(crops):
            x = self._prep(c) if c is not None else None
            if x is None:
                out.append(("", 0.0))
            else:
                xs.append(x)
                idxs.append(i)
        if xs:
            x = self.torch.from_numpy(np.stack(xs))
            with self.torch.no_grad():
                p = self.torch.softmax(self.net(x), dim=1).cpu().numpy()
            for j, i in enumerate(idxs):
                k = int(p[j].argmax())
                out.insert(i, (self.labels[k], float(p[j, k])))
        return out


# ---------------------------------------------------------------- 难例挖掘（对策③）

def center_rescan(runner, bgr):
    """复刻 ThreeStagePipeline.centerRescan(0.5)：中心 region×区域 1:1 裁剪重检。
    裁剪是原始像素的 1:1 子区域（只是进 512 输入时等效放大 2 倍），坐标映射 = 加偏移。"""
    h, w = bgr.shape[:2]
    l, t = int(w * (1 - RESCAN_REGION) / 2), int(h * (1 - RESCAN_REGION) / 2)
    roi_w, roi_h = min(int(w * RESCAN_REGION), w - l), min(int(h * RESCAN_REGION), h - t)
    if roi_w < 32 or roi_h < 32:
        return []
    crop = bgr[t:t + roi_h, l:l + roi_w]
    dets = runner.detect(crop, score_th=STORE_TH)
    return [(s, x1 + l, y1 + t, x2 + l, y2 + t) for s, x1, y1, x2, y2 in dets]


# ---------------------------------------------------------------- 主流程

def viz_only(seed=42, fresh=False):
    """从 pseudo_dets.jsonl 重建 quality/ 抽检样张（不重跑推理）。
    抽样按风险分带（标签清洗口径，2026-10-06 起，替代旧的"每类 top2"）：
      low(图最大框分<0.3) 40 · mid(0.3-0.5) 60 · high(>=0.5) 30 ·
      hard(rescan 救回) 12 · zero(全帧空图) 12
    按种抽检已废除——检测训练只学框位置，分类器种名对错无关训练。
    --fresh：排除 review.json 已判过的源图（出"下一批"用）。
    只动 quality/，不碰 labels/jsonl/stats/hard_examples/progress。"""
    import random
    import shutil
    bands = {"low": [], "mid": [], "high": []}
    hard_pf, zero_files = {}, []
    # 已判源图（--fresh 用）：判定键是样张文件名，按候选前缀对回 jsonl 源 stem
    judged_stems = set()
    if fresh:
        stems_all = set()
        try:
            with open(os.path.join(OUT, "pseudo_dets.jsonl"), encoding="utf-8") as f:
                for line in f:
                    try:
                        stems_all.add(os.path.splitext(json.loads(line)["file"])[0])
                    except Exception:
                        continue
            rv = json.load(open(os.path.join(OUT, "review.json"), encoding="utf-8"))
            for key in rv.get("verdicts", {}):
                toks = os.path.splitext(key)[0].split("_")
                for k in range(1, len(toks)):
                    cand = "_".join(toks[k:])
                    if cand in stems_all:
                        judged_stems.add(cand)
                        break
        except Exception:
            pass
    with open(os.path.join(OUT, "pseudo_dets.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            fn, ds = j["file"], j.get("dets") or []
            stem = os.path.splitext(fn)[0]
            ds = j.get("dets") or []
            if fresh and stem in judged_stems:
                continue
            if not ds:
                if len(zero_files) < 12:
                    zero_files.append(fn)
                continue
            mx = max(d["score"] for d in ds)
            key = "low" if mx < 0.3 else "mid" if mx < 0.5 else "high"
            bands[key].append((mx, fn))
            for d in ds:
                if d.get("src") == "rescan":
                    hard_pf[fn] = max(hard_pf.get(fn, 0.0), d["score"])
    qdir = os.path.join(OUT, "quality")
    shutil.rmtree(qdir, ignore_errors=True)
    os.makedirs(qdir, exist_ok=True)
    dets_of = {}
    with open(os.path.join(OUT, "pseudo_dets.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("dets"):
                dets_of[j["file"]] = j["dets"]

    def draw(fn, tag):
        bgr = cv2.imread(os.path.join(DATA, "images", fn))
        if bgr is None:
            return
        for e in dets_of.get(fn, []):
            color = (0, 255, 255) if e["src"] == "rescan" else \
                    (0, 165, 255) if e["low_conf"] else (0, 200, 0)
            p1, p2 = (int(e["x1"]), int(e["y1"])), (int(e["x2"]), int(e["y2"]))
            cv2.rectangle(bgr, p1, p2, color, 2)
            cv2.putText(bgr, f"{e['clf_label'][:20]} {e['score']:.2f}",
                        (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imwrite(os.path.join(qdir, f"{tag}_{os.path.splitext(fn)[0]}.jpg"), bgr)

    rng = random.Random(seed)
    seen = set()
    n_low = n_mid = n_high = 0
    for key, quota in (("low", 40), ("mid", 60), ("high", 30)):
        lst = bands[key][:]
        rng.shuffle(lst)
        for mx, fn in lst[:quota]:
            if fn in seen:
                continue
            seen.add(fn)
            draw(fn, f"{key}_{mx:.2f}")
            n_low += key == "low"; n_mid += key == "mid"; n_high += key == "high"
    for fn, s in sorted(hard_pf.items(), key=lambda kv: -kv[1])[:12]:
        if fn in seen:
            continue
        seen.add(fn)
        draw(fn, f"hard_{s:.2f}")
    for fn in zero_files:
        if fn in seen:
            continue
        seen.add(fn)
        draw(fn, "zero")
    print(f"quality/ 重建完成：{len(os.listdir(qdir))} 张"
          f"（低置信 {n_low} + 中置信 {n_mid} + 高置信抽检 {n_high}"
          f" + 难例 12 + 空图 {len(zero_files)}；分带母体 low {len(bands['low'])}"
          f" / mid {len(bands['mid'])} / high {len(bands['high'])}）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（冒烟）")
    ap.add_argument("--force", action="store_true", help="忽略 train.py 占用检查")
    ap.add_argument("--viz-only", action="store_true",
                    help="只从 jsonl 重建 quality/ 抽检样张，不重跑推理")
    ap.add_argument("--seed", type=int, default=42,
                    help="分带抽样种子（换一批新样张就换种子）")
    ap.add_argument("--fresh", action="store_true",
                    help="排除 review.json 里已判过的源图，只出没看过的")
    args = ap.parse_args()
    if args.viz_only:
        return viz_only(args.seed, args.fresh)

    # 与项目铁律一致：训练在跑就不动（外部 driver 会等 PIPELINE_DONE 再拉起本脚本）
    if not args.force:
        busy = os.popen(
            'powershell -NoProfile -Command "Get-CimInstance Win32_Process '
            "-Filter \\\"Name='python.exe'\\\" | Select-Object -ExpandProperty CommandLine\""
        ).read()
        if "train.py" in busy or "auto_region_pipeline" in busy or "pipeline_region" in busy:
            print("⚠️ 有训练链在跑（train.py），拒绝启动。等它结束或 --force。")
            return 2

    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "quality"), exist_ok=True)
    txt_dir = os.path.join(OUT, "labels")
    jsonl_path = os.path.join(OUT, "pseudo_dets.jsonl")

    with open(os.path.join(DATA, "dataset.csv"), encoding="utf-8-sig", newline="") as f:
        files = [r["file"] for r in csv.DictReader(f) if r.get("file")]
    if args.limit:
        files = files[:args.limit]
    print(f"dataset.csv: {len(files)} 张图")
    # 断点续传：把 jsonl 里已处理图的计数补回 st——否则最终 stats/监控台只含
    # 本轮新处理的部分，框总数比真实少一大截（实测 10.4k 张续传后只报 807 框）
    seed = {"box": 0, "low_conf": 0, "rescan_box": 0, "img_zero": 0, "img_det": 0}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                ds = j.get("dets") or []
                if not ds:
                    seed["img_zero"] += 1
                else:
                    seed["img_det"] += 1
                for d in ds:
                    seed["box"] += 1
                    seed["low_conf"] += int(d.get("low_conf"))
                    seed["rescan_box"] += int(d.get("src") == "rescan")
    prog_update(stage="pseudo_label", img_total=len(files), img_done=0,
                box=seed["box"], low_conf=seed["low_conf"],
                rescan_box=seed["rescan_box"], img_zero=seed["img_zero"],
                pl_done=False)

    runner = YoloxRunner("s512", DET_ONNX)
    clf = CropClassifier(CLF_CKPT)

    st = {"img": 0, "skip": 0, "bad": 0, "img_det": seed["img_det"],
          "img_zero": seed["img_zero"],
          "box": seed["box"], "box_rescan": seed["rescan_box"],
          "low_conf": seed["low_conf"], "clf_no_crop": 0}
    score_hist = {"0.20-0.30": 0, "0.30-0.50": 0, ">=0.50": 0}
    by_species = {}
    species_samples = {}   # label -> [(best_score, file)]
    low_samples = []       # (min_score, file)
    hard_samples = []      # (best_rescan_score, file)
    zero_files = []
    hard_paths = set()

    jsonl = open(jsonl_path, "a", encoding="utf-8")
    t0 = time.time()
    for k, fn in enumerate(files):
        stem = os.path.splitext(fn)[0]
        if os.path.exists(os.path.join(txt_dir, stem + ".txt")):
            st["skip"] += 1
            continue
        bgr = cv2.imread(os.path.join(DATA, "images", fn))
        if bgr is None:
            st["bad"] += 1
            continue
        h, w = bgr.shape[:2]
        dets = runner.detect(bgr, score_th=STORE_TH)
        srcs = ["base"] * len(dets)
        if not dets:  # ③ 全帧 0 检出 → 难例挖掘
            rdets = center_rescan(runner, bgr)
            if rdets:
                dets, srcs = rdets, ["rescan"] * len(rdets)
                hard_paths.add(os.path.join(DATA, "images", fn))
                hard_samples.append((max(d[0] for d in dets), fn))
                st["box_rescan"] += len(dets)
            else:
                st["img_zero"] += 1
                if len(zero_files) < 12:
                    zero_files.append(fn)
        if dets:
            st["img_det"] += 1

        # ② 分类器交叉验证（整图 crop 一次批量前向；框按 CLF_MARGIN 外扩给上下文）
        crops = []
        for s, x1, y1, x2, y2 in dets:
            bw, bh = x2 - x1, y2 - y1
            l, t = max(0, int(x1 - CLF_MARGIN * bw)), max(0, int(y1 - CLF_MARGIN * bh))
            r2, b2 = min(w, int(x2 + CLF_MARGIN * bw)), min(h, int(y2 + CLF_MARGIN * bh))
            c = bgr[t:b2, l:r2]
            crops.append(c if c.size else None)
        preds = clf.predict(crops)

        entries = []
        with open(os.path.join(txt_dir, stem + ".txt"), "w") as f:
            for i, ((s, x1, y1, x2, y2), src) in enumerate(zip(dets, srcs)):
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                    continue
                lab, pr = preds[i] if i < len(preds) else ("", 0.0)
                low = pr < CLF_KEEP_TH
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                entries.append({"score": round(s, 4), "x1": round(x1, 1), "y1": round(y1, 1),
                                "x2": round(x2, 1), "y2": round(y2, 1), "src": src,
                                "clf_label": lab, "clf_prob": round(pr, 4), "low_conf": low})
                st["box"] += 1
                st["low_conf"] += int(low)
                score_hist["0.20-0.30" if s < 0.3 else "0.30-0.50" if s < 0.5 else ">=0.50"] += 1
                if s < 0.3 and len(low_samples) < 500:
                    low_samples.append((s, fn))
                if lab:
                    by_species[lab] = by_species.get(lab, 0) + 1
                    bucket = species_samples.setdefault(lab, [])
                    bucket.append((s, fn))
        jsonl.write(json.dumps({"file": fn, "w": w, "h": h, "dets": entries},
                               ensure_ascii=False) + "\n")
        st["img"] += 1
        if (k + 1) % 200 == 0:
            jsonl.flush()
            el = time.time() - t0
            # 速率只按新处理图算：把续传跳过的算进来会把 ETA 压成假乐观
            # （实测断点续传后显示 2.3 分钟，真实还要 40 分钟）
            ms = el / max(st["img"] + st["bad"], 1) * 1000
            prog_update(img_done=k + 1, box=st["box"], low_conf=st["low_conf"],
                        rescan_box=st["box_rescan"], img_zero=st["img_zero"],
                        rate_ms=round(ms, 1), elapsed_s=round(el, 1),
                        eta_min=round((len(files) - k - 1) * ms / 1000 / 60, 1))
            print(f"  [{k + 1}/{len(files)}] {el:.0f}s "
                  f"框 {st['box']}（rescan {st['box_rescan']}）低置信 {st['low_conf']} "
                  f"空图 {st['img_zero']}，{ms:.0f} ms/图")

    jsonl.close()

    # ---- 抽检样张（先清空重建：冒烟期的旧样张不与全量混放，命名不含图名防重复拼接）----
    import shutil
    shutil.rmtree(os.path.join(OUT, "quality"), ignore_errors=True)
    os.makedirs(os.path.join(OUT, "quality"), exist_ok=True)
    def _draw(fn, tag):
        bgr = cv2.imread(os.path.join(DATA, "images", fn))
        if bgr is None:
            return
        entries = _dets_of.get(fn, [])
        for e in entries:
            color = (0, 255, 255) if e["src"] == "rescan" else \
                    (0, 165, 255) if e["low_conf"] else (0, 200, 0)
            p1, p2 = (int(e["x1"]), int(e["y1"])), (int(e["x2"]), int(e["y2"]))
            cv2.rectangle(bgr, p1, p2, color, 2)
            cv2.putText(bgr, f"{e['clf_label'][:20]} {e['score']:.2f}",
                        (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imwrite(os.path.join(OUT, "quality", f"{tag}_{stem0(fn)}.jpg"), bgr)

    # 重读 jsonl 拿每图 dets（画框用；内存里不存全量）
    _dets_of = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j["dets"]:
                _dets_of[j["file"]] = j["dets"]

    def stem0(fn):
        return os.path.splitext(fn)[0]

    n_viz = 0
    for lab, lst in species_samples.items():
        for s, fn in sorted(lst, reverse=True)[:2]:
            _draw(fn, f"sp_{lab.replace(' ', '_')[:30]}_{n_viz}")
            n_viz += 1
    for s, fn in sorted(low_samples)[:12]:
        _draw(fn, f"low_{s:.2f}")
    for s, fn in sorted(hard_samples, reverse=True)[:12]:
        _draw(fn, f"hard_{s:.2f}")
    for fn in zero_files:
        _draw(fn, "zero")

    with open(os.path.join(OUT, "hard_examples.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(hard_paths)))
    prog_update(img_done=st["img"] + st["skip"], box=st["box"],
                low_conf=st["low_conf"], rescan_box=st["box_rescan"],
                img_zero=st["img_zero"], eta_min=0, pl_done=True)
    top_species = sorted(by_species.items(), key=lambda kv: -kv[1])[:20]
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"elapsed_s": round(time.time() - t0, 1), "score_hist": score_hist,
                   "by_species_top20": top_species, **st}, f, ensure_ascii=False, indent=2)

    print(f"\n=== 伪标签完成 ===\n图 {st['img']}（跳过 {st['skip']} / 坏图 {st['bad']}）"
          f"\n框 {st['box']}（rescan 难例框 {st['box_rescan']}，低置信 {st['low_conf']}）"
          f"\n有框图 {st['img_det']}，全帧+rescan 均空 {st['img_zero']}"
          f"\n难例图 {len(hard_paths)} → hard_examples.txt"
          f"\n样张 → {os.path.join(OUT, 'quality')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

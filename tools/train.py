"""Train a bird classifier on the regional dataset.

Two models are supported so the experiment can separate "regional data" from
"bigger backbone":
  * mobilenetv4_small - ~4M params, the realistic on-device target
  * birdnet11m         - the released ConvNeXt-style net (11.2M), for comparison

The split is grouped by GBIF occurrence key, so every photo of one individual
bird lands on the same side. Splitting per-photo (as the released dataset does)
leaks train identities into val and inflates accuracy by tens of points.

Runs on CUDA when available and falls back to CPU; on OOM it halves the batch
and uses gradient accumulation to keep the effective batch size.
"""
import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# 脱离启动时 stdout 管道是 GBK，日志里的 ✅/中文会炸（HANDOFF "训练必设 PYTHONIOENCODING=utf-8" 的脚本内等价物）
import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


LOG_PATH = None  # main() 里指向 <out>/train.log，供仪表盘实时监控 epoch


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    if LOG_PATH:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {m}\n")
        except OSError:
            pass


# --------------------------------------------------------------------------- data
class BirdSet(Dataset):
    def __init__(self, rows, labels, train):
        self.rows, self.labels = rows, labels
        self.train = train
        aug = [T.RandomResizedCrop(224, scale=(0.55, 1.0)),
               T.RandomHorizontalFlip(),
               T.RandomRotation(20, expand=False),
               T.ColorJitter(0.3, 0.3, 0.3, 0.05)]
        # RandomErasing needs a CHW tensor, so it goes after ToTensor/Normalize
        self.tf_train = T.Compose(aug + [T.ToTensor(),
                                         T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                                         T.RandomErasing(p=0.15)])
        self.tf_val = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                                  T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.rows)

    def _path(self, file):
        return file if os.path.isabs(file) else os.path.join(self.img_dir, file)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = Image.open(self._path(r["file"])).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        tf = self.tf_train if self.train else self.tf_val
        if self.train:
            for _ in range(2):                      # retry a decode on a bad jpeg
                try:
                    img = Image.open(self._path(r["file"])).convert("RGB")
                    break
                except Exception:
                    pass
        return tf(img), self.labels[r["scientific_name"]]


def _img_path(img_dir, file):
    """合并数据集（全国兜底）的 file 是绝对路径——图片分散在五区+补抓目录。"""
    return file if os.path.isabs(file) else os.path.join(img_dir, file)


def load_rows(dataset_csv):
    img_dir = os.path.join(os.path.dirname(dataset_csv), "images")
    rows = []
    with open(dataset_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if os.path.exists(_img_path(img_dir, r["file"])):
                rows.append(r)
    return img_dir, rows


def grouped_split(rows, val_frac=0.2, seed=0):
    """Split by occurrence (individual), stratified per species."""
    by_species = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r.get("occurrence_key") or r.get("gbif_key") or r["file"]
        by_species[r["scientific_name"]][key].append(r)
    train, val = [], []
    rng = random.Random(seed)
    for sci, groups in by_species.items():
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_frac))
        for i, k in enumerate(keys):
            (val if i < n_val else train).extend(groups[k])
    return train, val


def split_by_list(rows, val_csv):
    """按冻结的 val 文件清单切分（frozen_val.csv，见 tools/freeze_val.py）。

    与 grouped_split 同口径防泄漏：同一 occurrence_key 的图不许一半在 val 一半在 train
    —— 数据集扩量后可能给已在 val 的观察补新图，这些新行连 train 也不能进。
    """
    with open(val_csv, encoding="utf-8") as f:
        val_files = {r["file"] for r in csv.DictReader(f)}
    have = {r["file"] for r in rows}
    val = [r for r in rows if r["file"] in val_files]
    blocked = {r.get("occurrence_key") for r in val if r.get("occurrence_key")}
    train = [r for r in rows
             if r["file"] not in val_files
             and (r.get("occurrence_key") or "") not in blocked]
    stats = {
        "frozen": len(val_files),
        "present": len(val),
        "gone_from_dataset": len(val_files - have),
        "sibling_rows_dropped": len(rows) - len(val) - len(train),
    }
    return train, val, stats


# --------------------------------------------------------------------------- models
def build_model(name, num_classes):
    if name == "birdnet11m":
        import sys
        repo = r"E:\code\code tools\bird-cn-373\code\Bird-Classification-main"
        sys.path.insert(0, repo)
        from _BirdNet import BirdNet
        return BirdNet(dims=(64, 128, 256, 512), layers=(3, 4, 6, 3),
                       class_num=num_classes)
    try:
        import timm
        return timm.create_model(name, pretrained=True, num_classes=num_classes)
    except ImportError:
        from torchvision import models
        m = getattr(models, name)(weights="IMAGENET1K_V1")
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m


# --------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to dataset.csv")
    ap.add_argument("--model", default="mobilenetv4_small_300")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-list", default=None,
                    help="frozen_val.csv 路径：按冻结清单切 val，别按当前全量重切"
                         "（新旧模型才可比，见 tools/freeze_val.py）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-per-species", type=int, default=0)
    ap.add_argument("--balance", choices=["none", "loss", "sampler", "both"],
                    default="none",
                    help="贫数据类平衡：loss=类加权损失, sampler=加权采样, both=两者")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    global LOG_PATH
    LOG_PATH = os.path.join(a.out, "train.log")
    # 轮次分隔行：train.log 是**追加**写的（见下面 log() 的 "a"），同一目录重训时
    # 上一轮的 epoch 行还在文件里。监控台按最后一条这样的分隔行切出"本轮"，
    # 否则会把上一轮的成绩当成当前曲线（旧日志没有这行，它会退化为用
    # "图像 N 张 / M 种" 那一行当边界）。
    log(f"=== run {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev}" + (f" {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    img_dir, rows = load_rows(a.dataset)
    species = sorted({r["scientific_name"] for r in rows})
    labels = {s: i for i, s in enumerate(species)}
    log(f"图像 {len(rows)} 张 / {len(species)} 种")

    if a.max_per_species:
        keep = defaultdict(int)
        out = []
        for r in rows:
            s = r["scientific_name"]
            if keep[s] < a.max_per_species:
                keep[s] += 1
                out.append(r)
        rows = out
        log(f"限流后 {len(rows)} 张")

    if a.val_list:
        tr, va, st = split_by_list(rows, a.val_list)
        log(f"按冻结验证集划分: train {len(tr)} / val {len(va)} "
            f"(冻结清单 {st['frozen']} 张，其中 {st['gone_from_dataset']} 张已不在当前 "
            f"dataset.csv；{st['sibling_rows_dropped']} 行因与 val 同 observation 被排除出 train)")
    else:
        tr, va = grouped_split(rows, a.val_frac)
        log(f"按 observation 分组划分: train {len(tr)} / val {len(va)} "
            f"(val 占 {len(va)/(len(tr)+len(va))*100:.1f}%)")
    # leakage sanity check
    tr_obs = {r.get("occurrence_key") for r in tr}
    va_obs = {r.get("occurrence_key") for r in va}
    overlap = len(tr_obs & va_obs)
    log(f"泄漏检查: train/val 共享 observation = {overlap} "
        f"{'✅' if overlap == 0 else '❌'}")

    ds_tr = BirdSet(tr, labels, True)
    ds_tr.img_dir = img_dir
    ds_va = BirdSet(va, labels, False)
    ds_va.img_dir = img_dir

    # 贫数据类平衡（--balance）：类加权 loss + 按类反频率的加权采样
    from collections import Counter
    cls_count = Counter(r["scientific_name"] for r in tr)
    use_loss_w = a.balance in ("loss", "both")
    use_sampler = a.balance in ("sampler", "both")
    crit_w = None
    sampler = None
    if use_loss_w or use_sampler:
        counts = [max(1, cls_count.get(s, 1)) for s in species]
        log(f"类别平衡 mode={a.balance}: min {min(counts)} / max {max(counts)}")
    if use_loss_w:
        # 有效样本数平方根平衡（class-balanced loss 的平滑版）
        n_tot, n_cls = sum(counts), len(counts)
        w = [(n_tot / (n_cls * c)) ** 0.5 for c in counts]
        w = torch.tensor(w, dtype=torch.float32)
        w = w / w.mean()  # 均值归一，保持整体 loss 量级
        # AMP 下 cross_entropy 的输入是 half，weight 必须同 dtype
        crit_w = w.to(dev).half() if dev.type == "cuda" else w
        crit_w = crit_w.to(dev)
    if use_sampler:
        s_w = [1.0 / cls_count.get(r["scientific_name"], 1) for r in tr]
        sampler = torch.utils.data.WeightedRandomSampler(
            s_w, num_samples=len(tr), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=a.batch,
                       shuffle=(sampler is None), sampler=sampler,
                       num_workers=a.workers, pin_memory=(dev.type == "cuda"),
                       drop_last=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False,
                       num_workers=a.workers, pin_memory=(dev.type == "cuda"),
                       persistent_workers=a.workers > 0)

    net = build_model(a.model, len(species)).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    log(f"模型 {a.model}  {n_par/1e6:.2f}M 参数  {len(species)} 类")

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.02)
    steps = max(1, len(dl_tr) // a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05, weight=crit_w)

    best, hist = 0.0, []
    for ep in range(a.epochs):
        # stage 1: head only for a couple of epochs
        freeze = ep < a.head_epochs
        if freeze and not hasattr(net, "_froze"):
            head = next((m for n in ("head", "classifier", "fc")
                         if (m := getattr(net, n, None)) is not None), None)
            if head is not None:
                for p in net.parameters():
                    p.requires_grad = False
                for p in head.parameters():
                    p.requires_grad = True
                log("阶段1: 只训分类头")
            else:
                log("阶段1 跳过: 未找到分类头，直接全网微调")
            net._froze = True
        if ep == a.head_epochs:
            for p in net.parameters():
                p.requires_grad = True
            log("阶段2: 全网微调")

        net.train()
        t0, tot, seen, correct = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                out = net(x)
                loss = crit(out, y)
            scaler.scale(loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
            bs = y.size(0)
            tot += loss.item() * bs
            seen += bs
            correct += (out.argmax(1) == y).sum().item()

        # val
        net.eval()
        vt, vseen, top1, top3, top5 = 0.0, 0, 0, 0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                    o = net(x)
                vt += crit(o, y).item() * y.size(0)
                vseen += y.size(0)
                rank = o.argsort(1, descending=True)
                top1 += (rank[:, :1] == y[:, None]).any(1).sum().item()
                top3 += (rank[:, :3] == y[:, None]).any(1).sum().item()
                top5 += (rank[:, :5] == y[:, None]).any(1).sum().item()
        acc = [top1 / vseen, top3 / vseen, top5 / vseen]
        hist.append({"epoch": ep + 1, "train_loss": tot / max(1, seen),
                     "train_acc": correct / max(1, seen), "val_loss": vt / vseen,
                     "top1": acc[0], "top3": acc[1], "top5": acc[2]})
        log(f"ep{ep+1:>3}/{a.epochs}  loss {tot/max(1,seen):.3f}  "
            f"train_acc {correct/max(1,seen)*100:5.2f}%  |  val "
            f"top1 {acc[0]*100:5.2f}%  top3 {acc[1]*100:5.2f}%  top5 {acc[2]*100:5.2f}%  "
            f"({time.time()-t0:.0f}s)")

        if acc[0] > best:
            best = acc[0]
            torch.save({"model": net.state_dict(), "model_name": a.model,
                        "species": species, "labels": labels,
                        "epoch": ep + 1, "top1": acc[0], "top3": acc[1],
                        "top5": acc[2], "params": n_par,
                        "train_cfg": vars(a)}, os.path.join(a.out, "best.pt"))

    with open(os.path.join(a.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "species": species, "history": hist,
                   "best_top1": best, "params": n_par,
                   "n_images": len(rows), "n_species": len(species)}, f,
                  ensure_ascii=False, indent=1)
    log(f"完成 best top1={best*100:.2f}% -> {a.out}/best.pt")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""终局对账：用 **train.py 自己那套代码**（同一模型构造、同一验证集切分、同一预处理）
去判华东 best.pt，看得到 56.12%（训练日志）还是 68.13%（eval_report）。

如果得到 56.12%：训练记录是对的，eval_report 侧另有蹊跷（继续查它）。
如果得到 68.13%：**训练日志记错了**，华东的真实水平是 68%，它不是最弱的区。

用法：python tools/diag_huadong_recheck.py
"""
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import BirdSet, build_model, grouped_split, load_rows  # noqa: E402

DATA = r"E:\code\code tools\data"
MODELS = r"E:\code\code tools\models"
REGIONS = {
    "huazhong": "huazhong_mnv4_25k",
    "huadong": "huadong_mnv4_300_rec",
    "dongbei": "dongbei_mnv4_rec",
}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
    print(f"{'区':<10}{'best.pt 记录':>12}{'train.py 判分':>13}{'差':>8}  {'训练日志 best':>14}")
    logs = {"huazhong": 57.18, "huadong": 56.12, "dongbei": 58.82}
    for region, run in REGIONS.items():
        ck = torch.load(os.path.join(MODELS, run, "best.pt"), map_location="cpu",
                        weights_only=False)
        species = ck["species"]
        labels = {s: i for i, s in enumerate(species)}

        img_dir, rows = load_rows(os.path.join(DATA, region, "dataset.csv"))
        species_in_data = sorted({r["scientific_name"] for r in rows})
        # 与 train.py 完全一致：labels 由「数据里的科学名排序」生成
        labels = {s: i for i, s in enumerate(species_in_data)}
        _, va = grouped_split(rows, 0.2)

        ds = BirdSet(va, labels, train=False)
        ds.img_dir = img_dir
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

        net = build_model(ck["model_name"], len(species)).to(dev)
        net.load_state_dict(ck["model"])
        net.eval()

        seen = t1 = t3 = t5 = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                    o = net(x)
                rank = o.argsort(1, descending=True)
                seen += y.size(0)
                t1 += (rank[:, :1] == y[:, None]).any(1).sum().item()
                t3 += (rank[:, :3] == y[:, None]).any(1).sum().item()
                t5 += (rank[:, :5] == y[:, None]).any(1).sum().item()
        got = t1 / seen * 100
        print(f"{region:<10}{ck['top1']*100:>11.2f}%{got:>12.2f}%{got-ck['top1']*100:>+7.2f}"
              f"  {logs[region]:>13.2f}%   n_val={seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
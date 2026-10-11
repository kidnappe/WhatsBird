# -*- coding: utf-8 -*-
r"""全国兜底模型无人值守链：GBIF 授权抓取 → 自动接训练。

阶段0  探针（缓存；各物种商用授权可用量评估，快速）
阶段1  全量抓取（--commercial-only 上架授权红线；断点续传，崩溃/中断重跑本驱动即接续）
阶段2  训练（tools/train.py，GPU；监控台 REGIONS["national"] 观测 train.log）

数据口径（LICENSES.md 红线）：仅 CC0 / CC_BY* / CC_BY_SA*；全国 bbox 18,54,73,135；
物种清单 = cn373 占位模型的 373 种（labels_birdnet373.txt，替换后 App 零改动）。

Usage: python -u national_pipeline.py
"""
import os
import subprocess
import sys
import time

ROOT = r"E:\code\observebird"
PY = sys.executable
SCRAPER = os.path.join(ROOT, "tools", "national_clf", "build_dataset_gbif.py")
SPECIES = os.path.join(ROOT, "tools", "national_clf", "species_national_373.csv")
NAT_DATA = r"E:\code\code tools\data\national"
NAT_MODEL = r"E:\code\code tools\models\national_mnv4_v1"
LOG = os.path.join(NAT_DATA, "pipeline.log")


def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run(cmd):
    log("RUN " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd)
    log(f"exit={r.returncode}  ({(time.time()-t0)/3600:.1f}h)")
    return r.returncode


def main():
    os.makedirs(NAT_DATA, exist_ok=True)
    log("=== national pipeline start ===")

    # 阶段0：探针（结果进 gbif_availability.csv，缓存）
    run([PY, "-u", SCRAPER, "--species", SPECIES, "--out", NAT_DATA,
         "--probe-only", "--bbox", "18", "54", "73", "135", "--commercial-only"])

    # 阶段1：全量抓取（断点续传）
    rc = run([PY, "-u", SCRAPER, "--species", SPECIES, "--out", NAT_DATA,
              "--per-species", "300", "--bbox", "18", "54", "73", "135",
              "--commercial-only", "--nice"])
    if rc != 0:
        log("抓取未正常结束，链停止——重跑本驱动即从断点接续")
        return 1

    # 阶段2：训练（373 类强不均衡 → loss 平衡；GPU）
    rc = run([PY, "-u", os.path.join(ROOT, "tools", "train.py"),
              "--dataset", os.path.join(NAT_DATA, "dataset.csv"),
              "--model", "mobilenetv4_conv_small.e2400_r224_in1k",
              "--epochs", "30", "--batch", "12", "--balance", "loss",
              "--workers", "4", "--out", NAT_MODEL])
    log("=== pipeline 完成 ===" if rc == 0 else f"=== 训练失败 exit={rc} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())

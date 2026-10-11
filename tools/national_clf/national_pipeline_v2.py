# -*- coding: utf-8 -*-
r"""全国兜底模型无人值守链 v2（2026-10-07 用户拍板：短期忽略授权，合成五区为主路）。

阶段1  缺口种补抓：141 个五区没有的种，全国 bbox 无授权过滤（_license 列照记，
        将来商用化按行过滤即可）
阶段2  合并：五区 148k + 补抓 → 全国 dataset.csv（绝对路径；<20 张的类剔除）
阶段3  训练：tools/train.py（373 类强不均衡 → loss 平衡；GPU，约 13-15 小时）
监控台 REGIONS["national"] 观测 train.log。中断重跑本驱动即断点接续。

Usage: python -u national_pipeline_v2.py
"""
import os
import shutil
import subprocess
import sys
import threading
import time

# stdout 管道默认 GBK，日志里的「䴓」等扩展区字符会 UnicodeEncodeError 炸掉无人值守链
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = r"E:\code\observebird"
PY = sys.executable
SCRAPER = os.path.join(ROOT, "tools", "national_clf", "build_dataset_gbif.py")
GAP_SPECIES = os.path.join(ROOT, "tools", "national_clf", "species_gap.csv")
MERGE = os.path.join(ROOT, "tools", "national_clf", "build_national_merged.py")
NAT_DATA = r"E:\code\code tools\data\national"
GAP_DATA = r"E:\code\code tools\data\national_gap"
NAT_MODEL = r"E:\code\code tools\models\national_mnv4_v1"
LOG = os.path.join(NAT_DATA, "pipeline.log")

_stop_beat = threading.Event()


def _heartbeat():
    """监控台数据_dir 口径镜像：把补抓阶段的心跳(progress.json)/实时 dataset.csv
    每 30s 镜像到 data/national——否则监控台 national 条目在抓取阶段只能显示
    「等待数据」（train_dashboard 只读 REGIONS[].data_dir 下的这两个文件）。"""
    src_prog = os.path.join(GAP_DATA, "progress.json")
    src_ds = os.path.join(GAP_DATA, "dataset.csv")
    while not _stop_beat.wait(30):
        try:
            if os.path.exists(src_prog):
                shutil.copyfile(src_prog, os.path.join(NAT_DATA, "progress.json"))
            if os.path.exists(src_ds):
                shutil.copyfile(src_ds, os.path.join(NAT_DATA, "dataset.csv"))
        except OSError:
            pass


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
    log("=== national pipeline v2 start（合成五区主路，短期忽略授权）===")
    threading.Thread(target=_heartbeat, daemon=True).start()

    # 阶段1：缺口种补抓（无授权过滤；断点续传）
    rc = run([PY, "-u", SCRAPER, "--species", GAP_SPECIES, "--out", GAP_DATA,
              "--per-species", "300", "--bbox", "18", "54", "73", "135", "--nice"])
    if rc != 0:
        log("补抓未正常结束，链停止——重跑本驱动即断点接续")
        return 1

    # 阶段2：合并（五区 148k + 缺口补抓；license 列保留）
    rc = run([PY, "-u", MERGE, "--out", NAT_DATA])
    if rc != 0:
        log("合并失败，链停止")
        return 1

    # 阶段3：训练
    rc = run([PY, "-u", os.path.join(ROOT, "tools", "train.py"),
              "--dataset", os.path.join(NAT_DATA, "dataset.csv"),
              "--model", "mobilenetv4_conv_small.e2400_r224_in1k",
              "--epochs", "30", "--batch", "12", "--balance", "loss",
              "--max-per-species", "500", "--workers", "4", "--out", NAT_MODEL])
    log("=== pipeline v2 完成 ===" if rc == 0 else f"=== 训练失败 exit={rc} ===")
    return rc


if __name__ == "__main__":
    sys.exit(main())

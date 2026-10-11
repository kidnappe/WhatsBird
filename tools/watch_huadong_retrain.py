# -*- coding: utf-8 -*-
r"""华东守候：iNat 补量收尾 → 自动拉起华东重训 → 验收写 PIPELINE_DONE。

**触发**：data/huadong/progress.json 的 stage=done（fetch_inat_region finalize）。
**收尾三件事**（docs/handoff/root/RERANK-DEVICE.md 清单 #1）：
  1. 验收 dataset.csv 行数/类数（预期 ~34,730 行 / 150 类）
  2. 删 fetch_target_inat.json（已完成轮次的残留会让监控台轮次条/口径异常）
  3. 训练（标准配方 batch12/accum6/epochs30/workers0，--out models\huadong_mnv4_300_rec，
     与旧模型 55.52% 并存作对比锚点）
**验收**：train.log 出现「完成 best top1=」才写 PIPELINE_DONE（项目约定，不训半成品）。

用法（run_detached.ps1 脱离启动）:
  python -u tools/watch_huadong_retrain.py
"""
import csv
import json
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
CODE_TOOLS = r"E:\code\code tools"
PY = sys.executable
POLL = 60
MODEL_DIR = os.path.join(CODE_TOOLS, "models", "huadong_mnv4_300_rec")
DATA = os.path.join(CODE_TOOLS, "data", "huadong")
PJ = os.path.join(DATA, "progress.json")
BUSY_PATS = ("build_dataset_gbif", "train.py", "auto_region_pipeline",
             "pipeline_region", "fetch_inat_region")


def log(m):
    print(f"{time.strftime('%H:%M:%S')} [hd-watch] {m}", flush=True)


def env():
    return {**os.environ, "PYTHONIOENCODING": "utf-8",
            "HF_ENDPOINT": "https://hf-mirror.com"}


def cmdlines():
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=90).stdout or ""
    except Exception:
        return ""


def busy():
    hits = []
    for line in cmdlines().splitlines():
        if "watch_huadong_retrain" in line:
            continue
        for pat in BUSY_PATS:
            if pat in line:
                hits.append(line.strip()[:110])
                break
    return hits


def dataset_stats():
    ds = os.path.join(DATA, "dataset.csv")
    species, n = set(), 0
    with open(ds, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("scientific_name"):
                species.add(r["scientific_name"])
                n += 1
    return n, len(species)



def main():
    os.makedirs(MODEL_DIR, exist_ok=True)   # 监控台 REGIONS 指向本目录，先建好免得自检报缺
    log("守候华东 iNat 补量：等 progress.json stage=done")
    idle = 0
    while True:
        st = ts = None
        try:
            j = json.load(open(PJ, encoding="utf-8"))
            st, ts = j.get("stage"), j.get("ts")
        except Exception:
            pass
        if st == "done":
            log(f"补量完成（{ts}）→ 进入收尾")
            break
        if st in ("interrupted", "failed"):
            log(f"⚠️ 补量终态 {st}（{ts}）—— 不在半成品上训练，退出")
            return 3
        dl_alive = "fetch_inat_region" in cmdlines()
        if not dl_alive:
            idle += 1
            if idle >= 3:
                log(f"⚠️ 下载进程没了且 stage={st} 无终态 —— 判定孤儿结束，退出")
                return 4
        else:
            idle = 0
        time.sleep(POLL)

    time.sleep(10)                                    # 等 finalize 落盘
    os.makedirs(MODEL_DIR, exist_ok=True)             # 监控台 REGIONS 已指向这里
    rows, ncls = dataset_stats()
    log(f"验收：dataset.csv {rows:,} 行 / {ncls} 类（预期 ~34,730 行 / 150 类）")
    ft = os.path.join(DATA, "fetch_target_inat.json")
    if os.path.exists(ft):
        os.remove(ft)
        log("已删 fetch_target_inat.json（已完成轮次的残留，防监控台口径异常）")

    # ---- 训练（有占用就等 5 分钟重试，12 次）----
    for attempt in range(1, 13):
        b = busy()
        if not b:
            break
        log(f"⚠️ 有别的任务在跑，5 分钟后重试（{attempt}/12）")
        for x in b:
            log(f"    [busy] {x}")
        time.sleep(300)
    else:
        log("⚠️ 重试耗尽，放弃训练")
        return 2

    log("=== 拉起华东重训（mobilenetv4_conv_small / 30ep / batch12 / accum6）===")
    rc = subprocess.run(
        [PY, "-u", os.path.join(TOOLS, "train.py"),
         "--dataset", os.path.join(DATA, "dataset.csv"),
         "--model", "mobilenetv4_conv_small.e2400_r224_in1k",
         "--epochs", "30", "--batch", "12", "--accum", "6", "--workers", "0",
         "--out", MODEL_DIR], env=env()).returncode

    ok = False
    try:
        ok = "完成 best top1=" in open(
            os.path.join(MODEL_DIR, "train.log"),
            encoding="utf-8", errors="replace").read()
    except Exception:
        pass
    if ok:
        open(os.path.join(MODEL_DIR, "PIPELINE_DONE"), "w",
             encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S")
                                     + " watch_huadong_retrain")
        log(f"✅ 华东重训完成 -> {MODEL_DIR}（REGIONS 已指向本目录，卡片自动转绿）")
        log("遗留人工项：资产链 copy_models_to_android.py + regen_guide.py --check"
            "（TFLite 导出需人工）")
        return 0
    log(f"⚠️ 训练退出码 {rc} 但无「完成 best top1=」—— 判未完成，没写 PIPELINE_DONE")
    return 5


if __name__ == "__main__":
    sys.exit(main())

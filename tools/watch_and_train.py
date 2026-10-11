# -*- coding: utf-8 -*-
"""守候下载结束，然后接着跑 审计 → 清洗 → 训练。

**为什么需要它**

`auto_region_pipeline.py` 是"下载子进程 + 父进程 wait()"的结构。如果**父进程意外死了**
（2026-10-05 华东那轮就是：stderr 空、没有"结束"行、下载子进程还好好的），
下载会照常跑完，但**没有人接着做 audit/clean/train** —— 链条静默断在半路。
本脚本顶上这一段。

**它做什么**

  轮询 `<data>/progress.json` 的 `stage`：
    · `done`       → 依次跑 audit_taxon_keys → clean_dataset → train.py
    · `interrupted`/`failed` → 记日志并退出（不训练半成品）
    · 长时间没动静且没有下载进程 → 判定孤儿结束，报出来退出
  训练完成判据：`<model>/train.log` 出现 "完成 best top1="（与项目既有约定一致）

用法（经 run_detached.ps1 脱离启动）:
  python -u tools/watch_and_train.py --region huadong
  python -u tools/watch_and_train.py --region huadong --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
CODE_TOOLS = r"E:\code\code tools"
PY = sys.executable
POLL_SEC = 60


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [watch] {msg}", flush=True)


def download_alive(region):
    """命令行里带 build_dataset_gbif 且带该区路径的 python 进程还有吗。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=90).stdout or ""
    except Exception:
        return True                      # 查不到就当活着，别误判
    for line in out.splitlines():
        if "build_dataset_gbif" in line and region in line:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--accum", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data_dir = os.path.join(CODE_TOOLS, "data", a.region)
    ds = os.path.join(data_dir, "dataset.csv")
    audit = os.path.join(data_dir, "gbif_taxon_audit.csv")
    cache = os.path.join(data_dir, "gbif_prefetch_cache.json")
    model_dir = os.path.join(CODE_TOOLS, "models", f"{a.region}_mnv4_300")
    pj = os.path.join(data_dir, "progress.json")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "HF_ENDPOINT": "https://hf-mirror.com"}
    species_csv = os.path.join(data_dir, f"species_{a.region}_150.csv")

    log(f"守候 {a.region}：等 {pj} 的 stage 变成 done")
    log(f"  数据 {ds}")
    log(f"  模型 {model_dir}")

    idle = 0
    while True:
        st, ts = None, None
        try:
            j = json.load(open(pj, encoding="utf-8"))
            st, ts = j.get("stage"), j.get("ts")
        except Exception:
            pass
        if st == "done":
            log(f"下载完成（{ts}）→ 开始 审计 → 清洗 → 训练")
            break
        if st in ("interrupted", "failed"):
            log(f"⚠️ 下载终态是 {st}（{ts}）—— **不训练半成品**，退出。"
                f"人工确认后再决定")
            return 3
        alive = download_alive(a.region)
        if not alive:
            idle += 1
            if idle >= 3:
                log(f"⚠️ 没有 {a.region} 的下载进程、progress.json 也没写终态"
                    f"（stage={st}）—— 判定为孤儿结束。**不训练半成品**")
                return 4
        else:
            idle = 0
        time.sleep(POLL_SEC)

    # ---- 1) 审计 ----
    log("1/3 audit_taxon_keys（抓取失败会写 FETCH_FAILED，不再写 NONE）")
    if not a.dry_run:
        rc = subprocess.run([PY, os.path.join(TOOLS, "audit_taxon_keys.py"),
                             "--cache", cache, "--out", audit], env=env).returncode
        if rc != 0:
            log(f"审计失败 exit={rc} —— 停在这里，不往下走")
            return 1
    # ---- 2) 清洗（fail-safe：网络失败不删行）----
    log("2/3 clean_dataset（FETCH_FAILED / 查询失败一律保留）")
    if not a.dry_run:
        rc = subprocess.run([PY, os.path.join(TOOLS, "clean_dataset.py"),
                             "--dataset", ds, "--audit", audit,
                             "--species-list", species_csv], env=env).returncode
        if rc != 0:
            log(f"清洗失败 exit={rc} —— 停在这里")
            return 1
    # ---- 3) 训练 ----
    log("3/3 train.py（mobilenetv4_conv_small / 30 epoch / workers 0）")
    if a.dry_run:
        log("dry-run：到此为止")
        return 0
    rc = subprocess.run([PY, "-u", os.path.join(TOOLS, "train.py"),
                         "--dataset", ds,
                         "--model", "mobilenetv4_conv_small.e2400_r224_in1k",
                         "--epochs", str(a.epochs), "--batch", str(a.batch),
                         "--accum", str(a.accum), "--workers", "0",
                         "--out", model_dir], env=env).returncode
    if rc != 0:
        log(f"训练失败 exit={rc}")
        return 1
    tl = os.path.join(model_dir, "train.log")
    ok = False
    try:
        ok = "完成 best top1=" in open(tl, encoding="utf-8", errors="replace").read()
    except Exception:
        pass
    if ok:
        open(os.path.join(model_dir, "PIPELINE_DONE"), "w",
             encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"✅ 全部完成 -> {model_dir}")
    else:
        log(f"⚠️ 训练进程退出码 {rc}，但 train.log 里没有「完成 best top1=」"
            f"—— 判为未完成，**没写 PIPELINE_DONE**")
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())

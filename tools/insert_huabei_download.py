# -*- coding: utf-8 -*-
"""把华北补量下载插进华南之后的空档：等华南下载终态 → 顺序跑华北下载 → 退出。

只管下载；训练仍由 overnight_chain 串行排队（华东→华南→华北），下载与训练并行
是 2026-10-06 用户拍板的（下载 0 GPU、内存实测 4GB+ 余量）。

参数与 recover_region.py 第一步逐字一致（同账本同缓存，链接手时恢复的下载步骤
会看到全部达标而秒过），所以这里跑完或跑一半都无损 —— 链会兜底重试。

用法（经 run_detached.ps1 脱离启动）:
  python -u tools/insert_huabei_download.py
"""
import json
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
CODE_TOOLS = r"E:\code\code tools"
PY = sys.executable
POLL = 60


def log(m):
    print(f"{time.strftime('%H:%M:%S')} [insert] {m}", flush=True)


def stage_of(region):
    try:
        return json.load(open(
            os.path.join(CODE_TOOLS, "data", region, "progress.json"),
            encoding="utf-8")).get("stage")
    except Exception:
        return None


def main():
    log("等华南下载终态（stage=done 放行；interrupted/failed 也放行，两区互不依赖）")
    while True:
        st = stage_of("huanan_sw")
        if st in ("done", "interrupted", "failed"):
            log(f"华南终态 stage={st} → 启动华北补量下载")
            break
        time.sleep(POLL)

    cmd = [PY, "-u", os.path.join(TOOLS, "build_dataset_gbif.py"),
           "--species", os.path.join(CODE_TOOLS, "data", "huabei", "gbif_availability.csv"),
           "--out", os.path.join(CODE_TOOLS, "data", "huabei"),
           "--per-species", "300", "--max-pages", "6", "--workers", "12",
           "--api-timeout", "60", "--bbox", "36", "43", "110", "120", "--nice"]
    log("$ " + " ".join(cmd))
    rc = subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    log(f"华北下载 exit={rc} → 插入器退出（训练由 overnight_chain 排队，无需本进程）")
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""Unattended regional pipeline: species list -> download -> clean -> train.

Run detached via run_detached.ps1 (this process must survive window close).
Progress artifacts in <data_dir>:
  gbif_build.log.<ts>       download log (same pattern the dashboard parses)
  progress.json             下载状态唯一来源：stage=done/interrupted/failed
                            + exit_code + species_done/total + finished_at + pid
                            （由 build_dataset_gbif.py 的 finalize_progress 写）
  PIPELINE_DONE             written when training finished successfully
  PIPELINE_STAGE            current stage marker (download/clean/train/done/failed)

注意：历史上这里还会写 gbif_build.log.<ts>.exit 当"下载完成标志"。那个标记
不区分"跑完"和"被杀"，且被监控台当成完成判据，是"华东只下 57/150 却显示
100% 已完成"事故的直接原因，2026-10-05 已废弃 —— 完成状态只看 progress.json。
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


def log(stage: str, msg: str):
    line = f"{time.strftime('%H:%M:%S')} [{stage}] {msg}"
    print(line, flush=True)


def stage_marker(data_dir: str, stage: str):
    with open(os.path.join(data_dir, "PIPELINE_STAGE"), "w", encoding="utf-8") as f:
        f.write(stage)


def sh(cmd, env):
    log("run", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True,
                    choices=["huabei", "dongbei", "huazhong", "huadong", "huanan_sw"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--per-species", type=int, default=300)
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    env = {**os.environ,
           "PYTHONIOENCODING": "utf-8",
           "HF_ENDPOINT": "https://hf-mirror.com"}
    regions = json.load(open(os.path.join(TOOLS, "regions.json"), encoding="utf-8"))
    bbox = regions[a.region]["bbox"]
    data_dir = os.path.join(CODE_TOOLS, "data", a.region)
    model_dir = os.path.join(CODE_TOOLS, "models", f"{a.region}_mnv4_300")
    os.makedirs(data_dir, exist_ok=True)
    species_csv = os.path.join(data_dir, f"species_{a.region}_{a.n}.csv")
    ds_csv = os.path.join(data_dir, "dataset.csv")

    # ---- 1) 物种清单 ----
    stage_marker(data_dir, "species")
    if not os.path.exists(species_csv):
        sh([sys.executable, os.path.join(TOOLS, "pick_region_species.py"),
            "--region", a.region, "--n", str(a.n), "--out", species_csv], env)
    if not os.path.exists(species_csv):
        sys.exit("物种清单生成失败")

    # ---- 2) 下载（子进程，日志与监控台同格式）----
    stage_marker(data_dir, "download")
    ts = time.strftime("%Y%m%d-%H%M%S")
    dl_log = os.path.join(data_dir, f"gbif_build.log.{ts}")
    cmd = [sys.executable, os.path.join(TOOLS, "build_dataset_gbif.py"),
           "--species", species_csv, "--out", data_dir,
           "--per-species", str(a.per_species), "--max-pages", str(a.max_pages),
           "--workers", "12", "--api-timeout", "60",
           "--bbox", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
           "--nice"]
    log("download", "启动: " + " ".join(cmd))
    with open(dl_log, "wb") as fo, open(dl_log + ".err", "wb") as fe:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env)
        code = proc.wait()
    # 不再写 gbif_build.log.<ts>.exit：那个标记不区分"跑完"和"被杀"，而且
    # 被下游当成完成判据（华东 57/150 被当成下完的直接原因）。完成状态由
    # build_dataset_gbif.py 自己写进 <data_dir>/progress.json 的 stage 字段。
    log("download", f"结束 exit={code}（完成状态见 {data_dir}\\progress.json 的 stage）")
    if code != 0:
        stage_marker(data_dir, "download_failed")
        sys.exit(f"下载失败 exit={code}")

    # ---- 3) 审计 + 清洗 ----
    stage_marker(data_dir, "clean")
    audit = os.path.join(data_dir, "gbif_taxon_audit.csv")
    sh([sys.executable, os.path.join(TOOLS, "audit_taxon_keys.py"),
        "--cache", os.path.join(data_dir, "gbif_prefetch_cache.json")], env)
    sh([sys.executable, os.path.join(TOOLS, "clean_dataset.py"),
        "--dataset", ds_csv, "--audit", audit, "--species-list", species_csv], env)

    # ---- 4) 训练（独占资源：本脚本运行期间不再有下载任务）----
    stage_marker(data_dir, "train")
    sh([sys.executable, "-u", os.path.join(TOOLS, "train.py"),
        "--dataset", ds_csv,
        "--model", "mobilenetv4_conv_small.e2400_r224_in1k",
        "--epochs", str(a.epochs), "--batch", "12", "--accum", "6",
        "--workers", "0", "--out", model_dir], env)

    stage_marker(data_dir, "done")
    with open(os.path.join(model_dir, "PIPELINE_DONE"), "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("done", f"华中管线全部完成 -> {model_dir}")


if __name__ == "__main__":
    main()

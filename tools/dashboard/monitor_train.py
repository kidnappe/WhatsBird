"""Live progress monitor for a train.py run.

The trainer's stdout goes through a buffered pipe, but best.pt is rewritten
whenever val top1 improves and carries epoch/top1/top3/top5 — so real progress
can be read off the checkpoint. Completion is detected via history.json.

Usage:
    python tools/monitor_train.py --out "E:/code/code tools/models/huabei_mnv4" [--interval 30]
"""
import argparse
import datetime
import json
import os
import subprocess
import time


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def gpu_status() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip().replace("\n", " | ") or "GPU n/a"
    except Exception:
        return "GPU n/a"


def gpu_busy() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return bool(r.stdout.strip())
    except Exception:
        return True  # 查不到就假设在跑，别误报


def read_ckpt(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None  # 多半是正在写入，下轮再读


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="train.py 的 --out 目录")
    ap.add_argument("--interval", type=float, default=30.0)
    a = ap.parse_args()

    ckpt_path = os.path.join(a.out, "best.pt")
    hist_path = os.path.join(a.out, "history.json")
    start = time.time()
    last_mtime, last_report = 0.0, ""

    print(f"[{now()}] 监控启动: {a.out}", flush=True)

    while True:
        best_txt = "-"
        mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0.0
        if mtime:
            if mtime != last_mtime:
                d = read_ckpt(ckpt_path)
                if d:
                    last_report = (
                        f"epoch {d['epoch']}/30 best | "
                        f"top1 {d['top1']*100:.2f}% top3 {d['top3']*100:.2f}% "
                        f"top5 {d['top5']*100:.2f}%"
                    )
                    last_mtime = mtime
            best_txt = last_report

        elapsed = datetime.timedelta(seconds=int(time.time() - start))
        print(f"[{now()}] 已监控 {elapsed} | {best_txt} | {gpu_status()}", flush=True)

        if not gpu_busy():
            time.sleep(5)
            if os.path.exists(hist_path):
                with open(hist_path, encoding="utf-8") as f:
                    h = json.load(f)
                print(f"[{now()}] 训练完成! best_top1={h['best_top1']*100:.2f}% "
                      f"({h['n_images']} 张 / {h['n_species']} 类)", flush=True)
                print("最近 5 个 epoch:", flush=True)
                for e in h["history"][-5:]:
                    print(f"  ep{e['epoch']:>3} val top1 {e['top1']*100:5.2f}% "
                          f"top3 {e['top3']*100:5.2f}% top5 {e['top5']*100:5.2f}%",
                          flush=True)
                return
            print(f"[{now()}] GPU 已空闲但没有 history.json——训练进程可能异常退出,"
                  f"请检查", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()

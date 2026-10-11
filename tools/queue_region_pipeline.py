"""Queue the regional pipeline behind a running job (training/download).

训练与 GBIF 下载不能并行（15GB 内存顶不住，见 HANDOFF 教训 #4）。本脚本先等
指定 PID 退出（训练结束或被停），空窗 grace 秒后接力启动 auto_region_pipeline。

必须通过 run_detached.ps1 脱离启动（整条链都挂在 WmiPrvSE 下，关窗口不受影响）：
  .\tools\run_detached.ps1 -Exe D:\software\Python312\python.exe `
      -ScriptArgs @('-u','E:\code\observebird\tools\queue_region_pipeline.py',
                    '--wait-pid','<pid>','--region','huadong') `
      -LogPath "E:\code\code tools\data\huadong\pipeline_queue.log"
"""
import argparse
import os
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))


def pid_alive(pid: int) -> bool:
    """tasklist 精确查 PID；同时要求映像名是 python.exe，防 PID 复用误判。"""
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[1] == str(pid):
            return parts[0].lower() == "python.exe"
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True,
                    help="要等待结束的进程（如正在跑的训练）")
    ap.add_argument("--region", required=True)
    ap.add_argument("--grace", type=int, default=30,
                    help="目标进程退出后的空窗秒数（等内存/GPU 释放）")
    ap.add_argument("--poll", type=int, default=15)
    a = ap.parse_args()

    print(f"[queue] 等待 PID {a.wait_pid} 退出（每 {a.poll}s 轮询）", flush=True)
    while pid_alive(a.wait_pid):
        time.sleep(a.poll)
    print(f"[queue] PID {a.wait_pid} 已退出，{a.grace}s 后启动 {a.region} 管线", flush=True)
    time.sleep(a.grace)

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    code = subprocess.run(
        [sys.executable, "-u", os.path.join(TOOLS, "auto_region_pipeline.py"),
         "--region", a.region],
        env=env).returncode
    print(f"[queue] auto_region_pipeline --region {a.region} 结束 exit={code}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

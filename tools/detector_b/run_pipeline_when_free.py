# -*- coding: utf-8 -*-
"""B 数据链后台 driver：等华东重训收尾 → 伪标签全量 → 校准清单 → COCO bird 子集下载。

启动（本会话已挂）：
  python -u run_pipeline_when_free.py > E:\\code\\code tools\\data\\detector_b\\pipeline.log 2>&1

步骤间独立：伪标签失败会跳过校准清单但继续 COCO 下载；每步都可手动单独重跑。
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _progress import update as prog_update  # noqa: E402  progress.json（监控台读）

TOOLS = r"E:\code\code tools"
HERE = r"E:\code\observebird\tools\detector_b"
PY = sys.executable
DONE = os.path.join(TOOLS, "models", "huadong_mnv4_300_rec", "PIPELINE_DONE")
PJ = os.path.join(TOOLS, "data", "huadong", "progress.json")


def log(m):
    print(f"{time.strftime('%H:%M:%S')} [detB] {m}", flush=True)


def run(script, *args):
    log(f"=== {os.path.basename(script)} {' '.join(args)} ===")
    t0 = time.time()
    rc = subprocess.run([PY, "-u", os.path.join(HERE, script), *args]).returncode
    log(f"=== {os.path.basename(script)} rc={rc}（{time.time() - t0:.0f}s）===")
    return rc


def main():
    prog_update(stage="pseudo_label")
    if os.path.exists(DONE):
        log("华东重训已完成（PIPELINE_DONE 在），直接进数据链")
    else:
        log("等华东重训 PIPELINE_DONE（最长 8h）...")
        for i in range(8 * 60):
            if os.path.exists(DONE):
                log(f"PIPELINE_DONE 出现（轮询 {i} 分钟）")
                break
            time.sleep(60)
        else:
            log("⚠️ 等 8h 没等到 PIPELINE_DONE，放弃（华东可能 interrupt/failed，去查 watch 日志）")
            return 3
    time.sleep(120)  # 等 train 进程完全退出，避开 pseudo_label 的 train.py 占用检查

    for attempt in range(1, 7):  # 若有别的训练链已顶上，等它
        rc = run("pseudo_label.py")
        if rc != 2:
            break
        log(f"训练链占用，5 分钟后重试（{attempt}/6）")
        time.sleep(300)

    if rc == 0:
        prog_update(stage="calib")
        run("make_calib_b.py")
    else:
        log(f"⚠️ 伪标签 rc={rc}，跳过校准清单")

    prog_update(stage="coco")
    rc2 = run("make_coco_bird_subset.py")
    prog_update(stage="done" if rc == 0 and rc2 == 0 else "failed")
    log(f"=== 数据链结束：伪标签 rc={rc} / 校准 rc={0 if rc == 0 else '-'} / coco rc={rc2} ===")
    return 0 if rc == 0 and rc2 == 0 else 1


if __name__ == "__main__":
    os.makedirs(os.path.join(TOOLS, "data", "detector_b"), exist_ok=True)
    sys.exit(main())

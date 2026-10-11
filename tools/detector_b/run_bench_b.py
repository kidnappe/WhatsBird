# -*- coding: utf-8 -*-
"""带实时进度的 B 检测器 bench 运行器（监控台读它写的 bench_progress.json）。

为什么不改 bench_detector_a/b：那两个是**评审尺子**，A/B 两路线的数字必须同源可比。
这里只在**外层**起子进程、解析它本来就打印的进度行，评分代码一行不碰。

产出：
  <data>/detector_b/bench_progress.json   实时进度（监控台轮询）
  <data>/detector_b/bench_run.log         子进程完整输出（事后复盘）
  <data>/detector_b/results/*.json        各模型评分（bench 自己写）

用法：
  python run_bench_b.py <dec.onnx> <tag> [--also-s512] [--limit N]
  python run_bench_b.py --status           # 只打印当前进度
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_detector_b.py")
DETB = r"E:\code\code tools\data\detector_b"
PROGRESS = os.path.join(DETB, "bench_progress.json")
RUNLOG = os.path.join(DETB, "bench_run.log")
RESULTS = os.path.join(DETB, "results")

# 验收线（docs/handoff/detector_b/HANDOFF.md 阶段 3 / docs/handoff/detector_b/README-B-LOCAL.md §4，PC 口径）
THRESH = {
    "tiny": {"goal": 0.15, "label": "tiny 桶召回 ≥15%"},
    "small": {"goal": 0.70, "label": "small 桶召回 ≥70%"},
    "recall": {"goal": 0.45, "label": "总召回 ≥45%"},
    "fp_per_img": {"goal": 0.30, "label": "FP/图 ≤0.3", "lower_is_better": True},
    "huabei": {"goal": 0.70, "label": "华北检出率不大幅回退（现役 77.5%）"},
}

# stdout 进度行（bench_detector_a 里 print 的原文）
_RE_COCO = re.compile(
    r"\[(?P<model>[^\]]+)\]\s+(?P<done>\d+)/(?P<total>\d+)\s+"
    r"recall@0\.35=(?P<hit>\d+)/(?P<gt>\d+)=(?P<recall>[\d.]+)\s+fp=(?P<fp>\d+)")
# ⚠️ 不要在这条里写中文（"检出率"）：子进程若以非 UTF-8 编码输出，中文会先变成替换符，
# 含中文的正则就永远匹配不上。用 \S* 吃掉那段标签即可（位置/数字已足够定位）。
_RE_HUABEI = re.compile(
    r"\[(?P<model>[^\]]+)\]\s+huabei\s+(?P<done>\d+)/(?P<total>\d+)\s+"
    r"\S*@0\.35=(?P<rate>[\d.]+)")
_RE_PHASE = re.compile(r"===\s*\[(?P<model>[^\]]+)\]\s*(?P<phase>.+?)\s*===")


def _blank(model=None, phase=None, done=0, total=0):
    return {"running": True, "finished": False, "exit_code": None,
            "tag": None, "started": time.time(), "elapsed": 0.0,
            "model": model, "phase": phase, "done": done, "total": total,
            "pct": 0.0, "live": {}, "results": {}, "models": [],
            "error": None, "log_tail": []}


def read_progress():
    try:
        with open(PROGRESS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_progress(st):
    tmp = PROGRESS + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
        os.replace(tmp, PROGRESS)     # 原子替换：监控台永远读不到半个文件
    except OSError:
        pass


def load_results():
    """读 results/*.json 里的验收关键项（bench 自己写的最终分）。"""
    out = {}
    for p in glob.glob(os.path.join(RESULTS, "*.json")):
        tag = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        c = (d.get("coco") or {}).get("th0.35") or {}
        hu = (d.get("huabei") or {}).get("th0.35") or {}
        if not c:
            continue
        bs = c.get("by_size") or {}
        row = {
            "recall": c.get("recall"),
            "tiny": (bs.get("tiny<0.25%") or {}).get("recall"),
            "small": (bs.get("small<1%") or {}).get("recall"),
            "large": (bs.get("large>=1%") or {}).get("recall"),
            "fp_per_img": c.get("fp_per_img"),
            "huabei": hu.get("det_rate"),
            "ms_per_img": (d.get("coco") or {}).get("ms_per_img"),
            "mtime": os.path.getmtime(p),
        }
        out[tag] = row
    return out


def verdict(row):
    """逐项判定是否过线；返回 [(key, 值, 门槛, 过?)...]。"""
    out = []
    for k, spec in THRESH.items():
        v = row.get(k)
        if v is None:
            continue
        goal = spec["goal"]
        ok = (v <= goal) if spec.get("lower_is_better") else (v >= goal)
        out.append({"key": k, "label": spec["label"], "value": v,
                    "goal": goal, "ok": ok})
    return out


def run(args):
    cmd = [sys.executable, "-u", BENCH, args.onnx]
    if args.tag:
        cmd.append(args.tag)
    if args.also_s512:
        cmd.append("--also-s512")
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print("bench 命令:", " ".join(cmd), flush=True)

    st = _blank()
    st["tag"] = args.tag
    st["models"] = ([args.tag] if args.tag else []) + (["s512"] if args.also_s512 else [])
    write_progress(st)

    creationflags = 0
    if os.name == "nt":
        # 训练正在同机跑（6 核笔记本）：bench 降优先级，别抢 GPU 侧的 CPU 供数
        creationflags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    # 子进程强制 UTF-8 输出：Windows 控制台默认 GBK，中文进度行的"检出率"三个字会
    # 变成替换符 → 解析器抓不到。PYTHONIOENCODING 让子进程按 UTF-8 写管道。
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.time()
    with open(RUNLOG, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                bufsize=1, creationflags=creationflags, env=env)
        for ln in proc.stdout:
            logf.write(ln)
            logf.flush()
            ln = ln.rstrip("\n")
            st["log_tail"] = (st["log_tail"] + [ln])[-40:]
            m = _RE_PHASE.search(ln)
            if m:
                st["model"] = m.group("model")
                st["phase"] = m.group("phase")
                # 换阶段要把进度一起归零：只清 done/total 会留下上一阶段的 pct，
                # 页面显示成"0/0 · 100%"（实测踩到）。
                st["done"] = st["total"] = 0
                st["pct"] = 0.0
                st["live"] = {}
            m = _RE_COCO.search(ln)
            if m:
                d, t = int(m.group("done")), int(m.group("total"))
                st.update(model=m.group("model"), phase="COCO bird 子集",
                          done=d, total=t, pct=round(d / max(t, 1) * 100, 1))
                st["live"] = {"recall": float(m.group("recall")),
                              "hit": int(m.group("hit")), "gt": int(m.group("gt")),
                              "fp": int(m.group("fp"))}
            m = _RE_HUABEI.search(ln)
            if m:
                d, t = int(m.group("done")), int(m.group("total"))
                st.update(model=m.group("model"), phase="华北 200",
                          done=d, total=t, pct=round(d / max(t, 1) * 100, 1))
                st["live"] = {"huabei": float(m.group("rate"))}
            st["elapsed"] = round(time.time() - t0, 1)
            st["results"] = load_results()
            write_progress(st)
        code = proc.wait()
    st["exit_code"] = code
    st["running"] = False
    st["finished"] = True
    st["pct"] = 100.0
    st["elapsed"] = round(time.time() - t0, 1)
    st["results"] = load_results()
    if code != 0:
        st["error"] = f"bench 退出码 {code}（见 bench_run.log 尾部）"
    write_progress(st)
    print(f"bench 结束 rc={code}，耗时 {st['elapsed']}s", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?", default=None)
    ap.add_argument("tag", nargs="?", default=None)
    ap.add_argument("--also-s512", action="store_true", help="同表带现役 s512 对照")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="只打印当前进度")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(read_progress(), ensure_ascii=False, indent=1))
        return 0
    if not a.onnx:
        print("用法: python run_bench_b.py <dec.onnx> [tag] [--also-s512] [--limit N]")
        return 2
    return run(a)


if __name__ == "__main__":
    sys.exit(main())

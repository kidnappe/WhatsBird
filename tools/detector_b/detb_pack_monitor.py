# -*- coding: utf-8 -*-
r"""B 检测器**外包包专用**只读训练监控台（自包含、零依赖、可离线）。

为什么单独写一个，而不是搬主仓 tools/dashboard/train_dashboard.py：
  主仓那台 2752 行、多地区（下载/训练/伪标签）大杂烩，且它的 B 训练采集
  update_detb_train() 读的是 run_detached 产出的 **train_b512.log.<ts> +
  .err.<ts> 双文件**。外包机是双击 start_train.bat 直跑，根本没有那对文件——
  它只有 loguru FileSink 写的**单个** train_log.txt。所以照搬会永远采不到数据。

数据源（全部在包内，相对路径，无任何主机绝对路径）：
  <root>/yolox_repo/YOLOX_outputs/<exp>/train_log.txt   ← 迭代/评测/完成/崩溃
  <root>/yolox_repo/YOLOX_outputs/<exp>/latest_ckpt.pth ← 断点存在性 + mtime
  <root>/yolox_repo/YOLOX_outputs/<exp>/best_ckpt.pth   ← AP 破纪录时才更新
  nvidia-smi                                            ← GPU 利用率/显存/温度

只读：只提供 GET /api/status，不提供任何 POST（外包机不装杀手锏，避免误杀训练）。

Usage:
    python detb_pack_monitor.py                 # root = 本脚本所在目录（包根）
    python detb_pack_monitor.py --root E:\...\observebird_detb_train_pack
    python detb_pack_monitor.py --port 8787 --exp yolox_bird_640
"""
import argparse
import collections
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 启动横幅是中文。Windows 控制台默认 GBK，而这些中文 GBK 都能编出来，所以
# **不要**强行改成 UTF-8：那会把本来正常的 GBK 控制台打成乱码（实测踩过）。
# 只补一个容错（万一控制台是 cp437 之类编不出中文的编码，替换而非崩溃），
# 编码本身沿用调用方/控制台的设定。
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 日志行正则（loguru 前缀 + yolox trainer 消息体）----
# 时间戳不再用正则取：loguru 恒为行首 19 字符 "YYYY-MM-DD HH:MM:SS"，切片即可
# （见 _parse_into 里的 ln[:19]），快且不必维护这个正则。
_IT_RE = re.compile(
    r"epoch:\s*(\d+)/(\d+),\s*iter:\s*(\d+)/(\d+),\s*gpu mem:\s*(\d+)Mb,\s*"
    r"mem:\s*([\d.]+)Gb,\s*iter_time:\s*([\d.]+)s,\s*data_time:\s*([\d.]+)s,\s*"
    r"total_loss:\s*([\d.]+),\s*iou_loss:\s*([\d.]+),\s*l1_loss:\s*([\d.]+),\s*"
    r"conf_loss:\s*([\d.]+),\s*cls_loss:\s*([\d.]+),\s*lr:\s*([\d.eE+-]+),\s*"
    r"size:\s*(\d+),\s*ETA:\s*(.+)$")
_EPOCH_RE = re.compile(r"start train epoch\s*(\d+)")
_NOAUG_RE = re.compile(r"No mosaic aug now")
_DONE_RE = re.compile(r"Training of experiment is done and the best AP is\s*([\d.]+)")
_RESUME_RE = re.compile(r"loaded checkpoint .* \(epoch (\d+)\)")
# pycocotools summarize 输出 6 行 AP + 4 行 AR：必须精确锁 "IoU=0.50:0.95 + area=all"
# 那行当总 AP，否则 best_ap 会被 area=large（值最高）污染；AP50 同理要 area=all。
_AP_RE = re.compile(
    r"Average Precision.*?@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*"
    r"maxDets=100\s*\]\s*=\s*([\d.]+)")
_AP50_RE = re.compile(
    r"Average Precision.*?@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*"
    r"maxDets=100\s*\]\s*=\s*([\d.]+)")

STALE_SEC = 900          # 与主监控台同一哲学：15 分钟无日志且未完成 → 判停滞
LOSS_CAP = 1200          # 曲线点上限
SPEED_WIN = 40           # 用最近 N 个迭代行估速

# 评测节奏默认值（与包内 yolox_bird_exp.py 一致）；若能读到 exp 文件则以文件为准
EVAL_INTERVAL = 5        # 常规阶段的评测间隔（轮）
NO_AUG_EPOCHS = 10       # 末尾 no_aug 轮数；进入该阶段后 trainer 把 interval 改成 1


def eval_epochs(total_epochs, no_aug_epochs=NO_AUG_EPOCHS, base_interval=EVAL_INTERVAL):
    """复刻 trainer.py 的评测调度，返回会触发评测的 epoch 列表（1-based）。

    trainer.py 里两处联动：
      before_epoch: if epoch+1 == max_epoch - no_aug_epochs → exp.eval_interval = 1
      after_epoch : if (epoch+1) % exp.eval_interval == 0   → evaluate_and_save_model()

    所以 ep50/no_aug10/interval5 = [5,10,...,35] + [40,41,...,50] 共 18 次，
    **不是**每 5 轮一次（那样只有 10 次）。AP 块与 epoch 的配对全靠这个顺序。
    """
    out, interval = [], base_interval
    for e in range(total_epochs):
        n = e + 1
        if n == total_epochs - no_aug_epochs:
            interval = 1
        if n % interval == 0:
            out.append(n)
    return out


def read_exp_params(root):
    """尽力从包内 yolox_bird_exp.py 读 eval_interval / no_aug_epochs / max_epoch。

    读不到就用默认值——这是**纯展示/配对用途**，不该因解析失败而让页面报错。
    """
    vals = {"eval_interval": EVAL_INTERVAL, "no_aug_epochs": NO_AUG_EPOCHS,
            "max_epoch": 50}
    try:
        with open(os.path.join(root, "yolox_bird_exp.py"), encoding="utf-8") as f:
            txt = f.read()
    except OSError:
        return vals
    for key, cast in (("eval_interval", int), ("no_aug_epochs", int),
                      ("max_epoch", int)):
        m = re.search(r"self\.%s\s*=\s*(\d+)" % key, txt)
        if m:
            try:
                vals[key] = cast(m.group(1))
            except ValueError:
                pass
    return vals


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def find_log(root, exp=None):
    """定位 train_log.txt。exp 给了就优先它，否则取 mtime 最新的一个。"""
    pat = os.path.join(root, "yolox_repo", "YOLOX_outputs", "*", "train_log.txt")
    cands = glob.glob(pat)
    if not cands:
        return None
    if exp:
        for p in cands:
            if os.path.basename(os.path.dirname(p)) == exp:
                return p
    return max(cands, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# 增量解析
# ---------------------------------------------------------------------------
# 为什么必须增量：满程日志是 50 ep × ~4000 iter ≈ 54MB / 20 万行。全量重解析一次
# 要 1.2~1.6s，而页面每 3 秒刷一次 —— 等于常驻大半个核在跟训练抢 CPU、还每 3 秒
# 把 54MB 从磁盘重读一遍。训练要跑 8~11 小时，这个浪费不能忍，尤其外包机是
# 16G/8G 的游戏本。改成"只解析自上次以来新增的字节"后，稳态开销降到微秒级。
#
# 状态按日志路径缓存；日志被截断/换文件（size 变小）或路径变了就整份重解析。
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _new_parse_state():
    return {
        "size": 0, "carry": "", "n": 0, "max_ep": 0,
        "last": {}, "pts": [], "recent": collections.deque(maxlen=SPEED_WIN),
        "aps": [], "ap50s": [], "eval_marks": 0,
        "cur_epoch": None, "ap_epochs": [],
        "last_err_at": None, "last_iter_at": None, "last_done_at": None,
        "final_ap": None, "err_tail": [], "err_left": 0,
        "tail": collections.deque(maxlen=40),
        "no_aug": False, "resumed_from": None,
        "started_at": None,
    }


def _parse_into(s, lines, base_idx):
    """把一批新行并进解析状态。base_idx = 这批首行的全局行号（用于位置比较）。"""
    for k, ln in enumerate(lines):
        i = base_idx + k
        s["tail"].append(ln)
        if s["err_left"] > 0:            # 错误现场：继续收后续行当证据
            s["err_tail"].append(ln)
            s["err_left"] -= 1
        if not s["started_at"] and len(ln) >= 19 and ln[4:5] == "-" and ln[10:11] == " ":
            s["started_at"] = ln[:19]
        if "after_iter" in ln:
            m = _IT_RE.search(ln)
            if m:
                (ep, tot, it, itot, gmem, ram, itime, _dt, tot_loss, iou, _l1,
                 conf, cls, lr, size, eta) = m.groups()
                ep, it, itot = int(ep), int(it), int(itot)
                s["last"] = {"epoch": ep, "total_epochs": int(tot), "iter": it,
                             "iters_per_epoch": itot, "gpu_mem_mb": int(gmem),
                             "ram_gb": float(ram), "iter_time": float(itime),
                             "loss": float(tot_loss), "iou_loss": float(iou),
                             "conf_loss": float(conf), "cls_loss": float(cls),
                             "lr": lr, "size": int(size), "eta_txt": eta.strip()}
                s["max_ep"] = max(s["max_ep"], ep)
                ts = ln[:19]
                gi = (ep - 1) * itot + it
                s["pts"].append((gi, float(tot_loss), ep))
                s["recent"].append((ts, gi))
                s["last_iter_at"] = i
            continue
        if "start train epoch" in ln:
            m = _EPOCH_RE.search(ln)
            if m:
                n_ep = int(m.group(1))
                s["max_ep"] = max(s["max_ep"], n_ep)
                s["cur_epoch"] = n_ep      # 当前在训哪一轮（评测归属靠它）
            continue
        if "Average Precision" in ln:
            m = _AP_RE.search(ln)
            if m:
                s["aps"].append(float(m.group(1)) * 100)
                # 评测发生在 after_epoch（紧随 "start train epochN" 之后），
                # 所以此刻的 cur_epoch 就是这次 AP 的归属轮次——直接读日志，
                # 不去推算节奏：续训/重复评测/改参数都天然正确。
                s["ap_epochs"].append(s["cur_epoch"])
                continue
            m = _AP50_RE.search(ln)
            if m:
                s["ap50s"].append(float(m.group(1)) * 100)
            continue
        if "is done and the best AP" in ln:
            m = _DONE_RE.search(ln)
            if m:
                s["final_ap"] = float(m.group(1))
                s["last_done_at"] = i
            continue
        if "Evaluate in main" in ln:
            s["eval_marks"] += 1
            continue
        if "No mosaic" in ln:
            s["no_aug"] = True
            continue
        if "loaded checkpoint" in ln:
            m = _RESUME_RE.search(ln)
            if m:
                s["resumed_from"] = int(m.group(1))
            continue
        if "Traceback (most recent call last)" in ln or "An error has been caught" in ln:
            s["last_err_at"] = i
            s["err_tail"] = [ln]
            s["err_left"] = 24


def _advance(path, s):
    """只读自 s['size'] 之后的新增字节并解析。返回是否有新内容。"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if size < s["size"]:          # 被截断/重建 → 整份重来
        s.clear()
        s.update(_new_parse_state())
    if size == s["size"]:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(s["size"])
            chunk = f.read()
    except OSError:
        return False
    s["size"] = size
    txt = s["carry"] + chunk.decode("utf-8", errors="replace")
    parts = txt.split("\n")
    s["carry"] = parts.pop() if parts else ""     # 末尾不完整行留到下轮
    if parts:
        _parse_into(s, parts, s["n"])
        s["n"] += len(parts)
    return True


def ckpt_info(path):
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return {"exists": True, "mb": round(st.st_size / 1048576, 1),
            "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "age_sec": round(time.time() - st.st_mtime)}


def gpu_info():
    """nvidia-smi 一次问全。没卡/没驱动就 ok=False，页面显示 n/a 而不是报错。"""
    q = ("name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
         "power.draw,power.limit")
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + q,
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8)
        if r.returncode != 0 or not r.stdout.strip():
            return {"ok": False, "err": "nvidia-smi 无输出"}
        first = r.stdout.strip().splitlines()[0]
        p = [x.strip() for x in first.split(",")]
        out = {"ok": True, "name": p[0], "util": p[1], "mem_used": p[2],
               "mem_total": p[3], "temp": p[4],
               "power": p[5] if len(p) > 5 else None,
               "power_limit": p[6] if len(p) > 6 else None}
        try:
            out["mem_pct"] = round(100.0 * float(p[2]) / float(p[3]), 1)
        except Exception:
            out["mem_pct"] = None
        return out
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


def parse_seconds(eta_txt):
    """把 trainer 的 ETA 文本（'3 days, 5:25:23' / '8:42:49'）折成秒。"""
    if not eta_txt:
        return None
    d = re.search(r"(\d+)\s*day", eta_txt)
    days = int(d.group(1)) if d else 0
    hms = re.search(r"(\d+):(\d+):(\d+)", eta_txt)
    if not hms:
        return None
    h, m, s = (int(x) for x in hms.groups())
    return days * 86400 + h * 3600 + m * 60 + s


def collect(root, exp=None):
    st = {
        "root": root, "ts": time.time(),
        "log": {"path": None, "exists": False, "age_sec": None},
        "run": {"experiment": None, "epoch": None, "total_epochs": 50,
                "iter": None, "iters_per_epoch": None, "global_iter": None,
                "total_iters": None, "progress_pct": None, "loss": None,
                "iou_loss": None, "conf_loss": None, "cls_loss": None,
                "lr": None, "gpu_mem_mb": None, "ram_gb": None,
                "iter_time": None, "size": None, "eta_txt": None,
                "eta_sec": None, "speed": None, "no_aug": False,
                "started_at": None, "elapsed_sec": None, "last_line_at": None},
        "eval": {"count": 0, "ap_hist": [], "best_ap": None, "final_ap": None,
                 "best_ckpt": None, "latest_ckpt": None},
        "loss_series": [], "tail": [], "error": {"is_error": False, "tail": []},
        "status": {"level": "idle", "text": "未启动"},
        "return_list": [],
    }

    # 评测节奏参数：以包内 exp 文件为准（读不到则用内置默认，绝不因此报错）
    exp_params = read_exp_params(root)
    st["no_aug_epochs"] = exp_params["no_aug_epochs"]
    st["eval_interval"] = exp_params["eval_interval"]
    st["run"]["total_epochs"] = exp_params["max_epoch"]

    logp = find_log(root, exp)
    if not logp:
        st["status"]["text"] = "未找到 train_log.txt（还没开训？）"
        st["gpu"] = gpu_info()
        return st

    st["log"]["path"] = os.path.relpath(logp, root)
    st["log"]["exists"] = True
    st["log"]["mtime"] = time.strftime("%m-%d %H:%M:%S",
                                       time.localtime(os.path.getmtime(logp)))
    st["log"]["age_sec"] = round(time.time() - os.path.getmtime(logp))
    exp_dir = os.path.dirname(logp)
    st["run"]["experiment"] = os.path.basename(exp_dir)

    # 增量解析：只有新增字节进正则，稳态开销微秒级（见 _advance 注释）
    with _CACHE_LOCK:
        s = _CACHE.get(logp)
        if s is None:
            s = _CACHE[logp] = _new_parse_state()
            if os.path.getsize(logp) > 256 * 1024 * 1024:
                # 超大日志（理论上不该有）放弃增量，避免一次卡死页面
                s["size"] = os.path.getsize(logp)
        _advance(logp, s)

    st["tail"] = list(s["tail"])
    st["eval"]["latest_ckpt"] = ckpt_info(os.path.join(exp_dir, "latest_ckpt.pth"))
    st["eval"]["best_ckpt"] = ckpt_info(os.path.join(exp_dir, "best_ckpt.pth"))
    st["run"].update(s["last"])
    st["run"]["no_aug"] = s["no_aug"]
    if s["resumed_from"]:
        st["run"]["resumed_from"] = s["resumed_from"]
    st["run"]["started_at"] = s["started_at"]
    if s["started_at"]:
        try:
            t0 = datetime.datetime.strptime(s["started_at"], "%Y-%m-%d %H:%M:%S")
            st["run"]["elapsed_sec"] = int(time.time() - t0.timestamp())
        except Exception:
            pass

    max_ep = s["max_ep"]
    aps = s["aps"]
    ap50s = s["ap50s"]
    eval_marks = s["eval_marks"]
    last_err_at = s["last_err_at"]
    last_iter_at = s["last_iter_at"]
    last_done_at = s["last_done_at"]
    iter_pts = s["pts"]

    # 全局进度
    if st["run"]["epoch"] and st["run"]["iters_per_epoch"] and st["run"]["total_epochs"]:
        gi = (st["run"]["epoch"] - 1) * st["run"]["iters_per_epoch"] + st["run"]["iter"]
        tot = st["run"]["total_epochs"] * st["run"]["iters_per_epoch"]
        st["run"]["global_iter"] = gi
        st["run"]["total_iters"] = tot
        st["run"]["progress_pct"] = round(100.0 * gi / tot, 2)
        if iter_pts:
            st["run"]["eta_sec"] = parse_seconds(st["run"]["eta_txt"])

    # 速率：最近 N 个迭代行的时间戳差（跨续训也准，因为只看最近窗口）
    win = list(s["recent"])
    if len(win) >= 3:
        try:
            t_a = datetime.datetime.strptime(win[0][0], "%Y-%m-%d %H:%M:%S")
            t_b = datetime.datetime.strptime(win[-1][0], "%Y-%m-%d %H:%M:%S")
            dt = (t_b - t_a).total_seconds()
            dgi = win[-1][1] - win[0][1]
            if dt > 0 and dgi > 0:
                st["run"]["speed"] = round(dgi / dt, 2)          # it/s
                if st["run"]["total_iters"] and st["run"]["global_iter"]:
                    left = st["run"]["total_iters"] - st["run"]["global_iter"]
                    st["run"]["eta_sec"] = int(left / st["run"]["speed"])
        except Exception:
            pass

    # AP 历史：第 k 个 AP 块对应第 k 次评测的 epoch。
    # 首选**从日志直接观测**到的归属（ap_epochs：解析到 AP 块时记下的 cur_epoch）；
    # 只有在拿不到（例如解析中途才接上、历史块没看到 start train epoch）时才回退到
    # 按 trainer 调度推算。观测值天然免疫续训、重复评测和参数改动。
    # ⚠️ 推算那条路有过真 bug：评测**不是**固定每 5 轮——trainer.py 的 before_epoch
    # 里有 `self.exp.eval_interval = 1`（进入 no_aug 阶段时触发），所以 ep50/no_aug10
    # 的实际节奏是 [5,10,...,35] + [40,41,...,50] 共 18 次，而非 range(5,51,5) 的 10 次。
    computed = eval_epochs(st["run"]["total_epochs"] or 50,
                           st.get("no_aug_epochs", 10))
    for k, ap in enumerate(aps):
        ep = s["ap_epochs"][k] if k < len(s["ap_epochs"]) else None
        if ep is None and k < len(computed):
            ep = computed[k]
        st["eval"]["ap_hist"].append({
            "epoch": ep, "ap": round(ap, 2),
            "ap50": round(ap50s[k], 2) if k < len(ap50s) else None})
    st["eval"]["count"] = max(eval_marks, len(aps))
    # best_ap 只认真实 AP 块。不掺 final_ap：崩溃时 after_train 打的 0.00 会把
    # "还没评过"污染成"评过且得 0 分"，页面上看就是一张假的 0.00 best AP。
    if aps:
        st["eval"]["best_ap"] = round(max(aps), 2)

    # 崩溃检测：判据是**位置**而非时间。yolox 死时 traceback 落在日志末尾，其后
    # 不再有迭代行；反之若用户在崩溃后已 resume_train，traceback 之后仍有迭代行，
    # 那就是"崩过但已恢复"，不能一直红着报警。用时间判（如"2 分钟内有报错"）会在
    # 日志刚写完的瞬间漏判，用位置判则两种情形都对。
    if last_err_at is not None:
        st["error"]["is_error"] = True
        st["error"]["tail"] = list(s["err_tail"])
        st["error"]["fatal"] = (last_iter_at is None or last_err_at > last_iter_at)
    # "is done" 行同理要按位置判：崩溃时 after_train 留下的那句 0.00 会**永久留档**，
    # 续训后仍在文件里。只有当它后面再没有迭代行时，才代表训练真的结束了。
    st["eval"]["finished"] = (last_done_at is not None
                              and (last_iter_at is None or last_done_at > last_iter_at))
    if st["eval"].get("final_ap") is None and s.get("final_ap") is not None:
        st["eval"]["final_ap"] = s["final_ap"]

    # 曲线降采样（pts 是 (gi, loss, epoch)）
    pts = iter_pts
    if len(pts) > LOSS_CAP:
        step = len(pts) / LOSS_CAP
        pts = [pts[min(len(pts) - 1, int(i * step))] for i in range(LOSS_CAP)]
    st["loss_series"] = [{"gi": p[0], "loss": round(p[1], 3), "epoch": p[2]}
                         for p in pts]

    # ---- 状态判定（顺序 = 优先级）----
    # ⚠️ 坑（实测）：yolox 的 after_train 是**无条件**打印 "Training of experiment
    # is done and the best AP is X" 的——训练中途崩溃时它照样打，且值通常是 0.00
    # （本包样本日志里 ep5 评测被 Ninja/JIT 坑崩掉就是这个形态）。所以"完成"必须
    # 让位给"崩溃"，否则外包同学会看到绿色的"训练完成"而实际早就死了。
    age = st["log"]["age_sec"]
    fatal = st["error"].get("fatal", False)
    done = st["eval"].get("finished", False) and not fatal
    if fatal:
        st["status"] = {"level": "bad",
                        "text": f"训练已崩溃（{age // 60} 分钟前）——报错见下方"}
    elif done:
        st["status"] = {"level": "ok",
                        "text": f"训练完成 · best AP {st['eval']['final_ap']:.2f}"}
    elif st["error"]["is_error"] and not fatal:
        # 崩过但之后有迭代行 = 已续训恢复，别一直报红
        st["status"] = {"level": "warn",
                        "text": f"训练中 · ep {st['run']['epoch']}/{st['run']['total_epochs']}"
                                f"（此前崩溃过，已续训）"}
    elif max_ep and age is not None and age > STALE_SEC:
        st["status"] = {"level": "bad",
                        "text": f"停滞：{age // 60} 分钟无新日志（训练可能已退出）"}
    elif max_ep:
        st["status"] = {"level": "warn",
                        "text": f"训练中 · ep {st['run']['epoch']}/{st['run']['total_epochs']}"
                                f" · {st['run']['progress_pct']}%"}
    else:
        st["status"] = {"level": "warn", "text": "日志已就位，等待首个迭代行…"}

    st["gpu"] = gpu_info()

    # ---- 回传清单（外包同学照着抄）----
    for f in ("latest_ckpt.pth", "best_ckpt.pth", "train_log.txt"):
        p = os.path.join(exp_dir, f)
        info = ckpt_info(p)
        st["return_list"].append({
            "name": f, "exists": bool(info),
            "mb": info["mb"] if info else None,
            "mtime": info["mtime"] if info else None})
    tb = os.path.join(exp_dir, "tensorboard")
    st["return_list"].append({"name": "tensorboard/", "exists": os.path.isdir(tb),
                              "mb": None, "mtime": None})
    return st


STATE = {"data": None, "ts": 0.0}
LOCK = threading.Lock()


def snapshot(root, exp):
    with LOCK:
        if STATE["data"] is None or time.time() - STATE["ts"] > 2.0:
            STATE["data"] = collect(root, exp)
            STATE["ts"] = time.time()
        return STATE["data"]


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B 检测器训练监控（外包包）</title>
<style>
  body{background:#111;color:#ddd;font-family:system-ui,"Microsoft YaHei",sans-serif;
       margin:20px;max-width:1000px;margin-inline:auto}
  h1{font-size:19px;color:#fff;margin:0 0 4px} h1 small{color:#888;font-weight:normal;font-size:13px}
  #state{border-radius:12px;padding:12px 18px;margin:12px 0;font-size:16px;font-weight:700}
  .st-idle{background:#222;color:#888}
  .st-warn{background:#1e3a5f;color:#7dd3fc}
  .st-ok{background:#14321f;color:#4ade80}
  .st-bad{background:#3a1212;color:#f87171}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));gap:10px;margin:14px 0}
  .card{background:#1d1d1d;border-radius:12px;padding:12px 14px;min-width:0}
  .card .v{font-size:22px;font-weight:700;color:#4ade80;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .card .v.b{color:#7dd3fc} .card .v.y{color:#fbbf24} .card .v.r{color:#f87171}
  .card .l{font-size:12px;color:#888;margin-top:3px}
  .barwrap{background:#333;border-radius:8px;height:22px;overflow:hidden;position:relative;margin:8px 0}
  .barwrap>div{height:100%;width:0%;background:linear-gradient(90deg,#4ade80,#22d3ee);transition:width .5s}
  .barlabel{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
            font-size:12px;color:#fff;text-shadow:0 1px 2px #000}
  #epbar{display:flex;gap:2px;margin:8px 0 2px}
  #epbar i{flex:1;height:16px;background:#2b2b2b;border-radius:3px}
  #epbar i.f{background:#4ade80} #epbar i.cur{background:#fbbf24;box-shadow:0 0 6px #fbbf24}
  #epbar i.na{background:#7c2d12}
  canvas{background:#1d1d1d;border-radius:12px;width:100%;height:230px;display:block;margin:12px 0}
  .box{background:#14181d;border-radius:12px;padding:10px 14px;height:210px;overflow-y:auto;
       font:12px/1.7 Consolas,monospace;color:#86efac;white-space:pre-wrap;word-break:break-all}
  .box.plain{background:#1a1a1a;color:#9ca3af}
  .alert{display:none;border-radius:10px;padding:10px 14px;margin:10px 0;font-size:13px}
  .alert.on{display:block;background:#3a1212;color:#fda4a4;font:12px/1.6 Consolas,monospace;
            max-height:220px;overflow-y:auto;white-space:pre-wrap}
  table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
  td{padding:7px 10px;background:#1d1d1d;border-bottom:2px solid #111}
  td.ok{color:#4ade80} td.no{color:#666}
  h2{font-size:14px;color:#aaa;margin:20px 0 6px;font-weight:600}
  .muted{color:#666;font-size:12px;margin-top:6px}
  .rftab{background:none;border:1px solid #444;border-radius:8px;color:#aaa;
         padding:4px 12px;font-size:12px;cursor:pointer;font-family:inherit}
</style></head><body>
<h1>B 检测器训练监控 <small id="sub">外包包 · 只读</small></h1>
<div id="state" class="st-idle">连接中…</div>
<div id="err" class="alert"></div>

<div class="cards" id="cards"></div>

<div id="epbar"></div>
<div class="barwrap"><div id="gbar"></div><div class="barlabel" id="glabel">-</div></div>

<h2>损失 / AP 曲线</h2>
<canvas id="loss" width="1000" height="230"></canvas>
<canvas id="ap" width="1000" height="230"></canvas>

<h2>训练日志尾部（train_log.txt）</h2>
<div class="box" id="log">-</div>

<h2>回传清单（训练完把这 4 项所在的整个目录发回）</h2>
<table id="rl"></table>
<div class="muted" id="foot"></div>

<script>
function el(i){return document.getElementById(i)}
function fmt(s){if(s==null)return"-";s=Math.max(0,Math.round(s));
  var d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),x=s%60;
  return d?d+"d "+h+"h "+m+"m":(h?h+"h "+m+"m":m+"m "+x+"s")}
function card(v,l,cls){return '<div class="card"><div class="v '+(cls||'')+'">'+v+'</div><div class="l">'+l+'</div></div>'}
function line(cv,pts,color,ylab){
  var c=el(cv),x=c.getContext('2d'),W=c.width,H=c.height;
  x.clearRect(0,0,W,H);
  x.strokeStyle='#2b2b2b';x.lineWidth=1;
  for(var i=0;i<=4;i++){var y=10+i*(H-30)/4;x.beginPath();x.moveTo(34,y);x.lineTo(W-8,y);x.stroke()}
  if(!pts||pts.length<2){x.fillStyle='#555';x.font='13px sans-serif';x.fillText('暂无数据',40,H/2);return}
  var xs=pts.map(function(p){return p[0]}),ys=pts.map(function(p){return p[1]});
  var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);
  var y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);
  if(x1===x0)x1=x0+1; if(y1===y0){y1=y0+1;y0=Math.max(0,y0-1)}
  var pad=(y1-y0)*0.08; y0-=pad; y1+=pad;
  x.strokeStyle=color;x.lineWidth=2;x.beginPath();
  pts.forEach(function(p,i){
    var px=34+(p[0]-x0)/(x1-x0)*(W-44),py=H-20-(p[1]-y0)/(y1-y0)*(H-34);
    i?x.lineTo(px,py):x.moveTo(px,py)});
  x.stroke();
  x.fillStyle='#777';x.font='11px monospace';
  x.fillText(y1.toFixed(2),2,16);x.fillText(y0.toFixed(2),2,H-22);
  x.fillText(ylab,W-70,16);
}
function draw(d){
  var r=d.run,e=d.eval,g=d.gpu||{};
  var st=el('state');st.className='st-'+d.status.level;st.textContent=d.status.text;
  var pct=r.progress_pct==null?0:r.progress_pct;
  el('gbar').style.width=pct+'%';
  el('glabel').textContent=r.global_iter?('总进度 '+r.global_iter.toLocaleString()+' / '+
      r.total_iters.toLocaleString()+' iter  ('+pct+'%)'):'等待首个迭代行';
  var n=r.total_epochs||50,cur=r.epoch||0,h='';
  for(var i=1;i<=n;i++)h+='<i class="'+(i<cur?'f':(i===cur?'cur':''))+'"></i>';
  el('epbar').innerHTML=h;
  el('cards').innerHTML=
    card(r.epoch!=null?(r.epoch+'/'+r.total_epochs):'-','epoch','')+
    card(r.iter!=null?(r.iter+'/'+r.iters_per_epoch):'-','iter','')+
    card(r.loss!=null?r.loss.toFixed(3):'-','total_loss','y')+
    card(r.lr||'-','lr','')+
    card(r.iter_time!=null?r.iter_time.toFixed(3)+'s':'-','iter_time','')+
    card(r.speed?r.speed.toFixed(2)+' it/s':'-','速度','')+
    card(r.gpu_mem_mb?r.gpu_mem_mb+'M':'-','显存(log)','')+
    card(r.size||'-','输入尺寸','')+
    card(e.best_ap!=null?e.best_ap.toFixed(2):'-','best AP','b')+
    card(e.count,'评测次数','')+
    card(g.ok?g.util+'%':'-','GPU 利用率','b')+
    card(g.ok?g.mem_used+'/'+g.mem_total+'M':'-','GPU 显存','b')+
    card(g.ok?g.temp+'°C':'-','GPU 温度','')+
    card(fmt(r.eta_sec),'剩余预估','')+
    card(fmt(r.elapsed_sec),'已运行','');
  el('log').textContent=(d.tail||[]).join('\n');
  el('log').scrollTop=el('log').scrollHeight;
  var eo=el('err');
  if(d.error.is_error){eo.className='alert on';eo.textContent=(d.error.tail||[]).join('\n')}
  else{eo.className='alert'}
  var rl='';
  (d.return_list||[]).forEach(function(f){
    rl+='<tr><td>'+f.name+'</td><td class="'+(f.exists?'ok':'no')+'">'+
        (f.exists?'✓ 已生成':'— 未生成')+'</td><td>'+(f.mb!=null?f.mb+' MB':'')+
        '</td><td>'+(f.mtime||'')+'</td></tr>'});
  el('rl').innerHTML=rl;
  el('sub').textContent=(r.experiment||'-')+' · 日志 '+((d.log.age_sec==null)?'-':d.log.age_sec+'s 前更新');
  line('loss',(d.loss_series||[]).map(function(p){return [p.gi,p.loss]}),'#4ade80','loss');
  line('ap',(e.ap_hist||[]).map(function(p){return [p.epoch,p.ap]}),'#7dd3fc','AP');
  el('foot').textContent='只读监控：不提供任何停止/启动按钮，训练进程不受本页面影响。'+
    ' 日志 15 分钟无更新会判为停滞。';
}
function tick(){
  fetch('/api/status').then(function(r){return r.json()}).then(function(d){
    if(d.error){el('state').className='st-bad';el('state').textContent='采集错误: '+d.error;return}
    draw(d)}).catch(function(e){
    el('state').className='st-bad';el('state').textContent='(连接中断，重试中) '+e});
}
tick();setInterval(tick,3000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    root = None
    exp = None

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import urlparse
        if urlparse(self.path).path == "/api/status":
            try:
                self._send(snapshot(self.root, self.exp))
            except Exception as e:
                # 采集异常别伪装成"连接中断"（主台踩过的坑），如实回 500 + 原因
                self._send({"error": f"{type(e).__name__}: {e}"}, 500)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._send({"error": "read-only monitor: no POST endpoints"}, 405)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE,
                    help="训练包根目录（含 yolox_repo/）；默认 = 本脚本所在目录")
    ap.add_argument("--exp", default=None,
                    help="指定实验目录名（如 yolox_bird_640）；默认取 mtime 最新")
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    Handler.root, Handler.exp = root, a.exp
    print(f"[{now()}] B 检测器外包包监控台（只读）", flush=True)
    print(f"[{now()}] root : {root}", flush=True)
    print(f"[{now()}] log  : {find_log(root, a.exp) or '未找到 train_log.txt'}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[{now()}] dashboard: http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())

"""Real-time training dashboard (local web page).

Serves http://127.0.0.1:8787 with live progress: best checkpoint numbers,
epoch progress bar, top-1 curve, GPU status, and an event log.
Data source: best.pt (rewritten by train.py whenever val top1 improves)
and history.json (written once at completion).

Usage:
    python -u tools/train_dashboard.py --out "E:/code/code tools/models/huabei_mnv4" [--port 8787]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {
    "started": time.time(),
    "events": [],        # [{ts, msg}] newest first
    "gpu": "n/a",
    "regions": {},       # region key -> per-region train/download 状态
    "models_root": None, # models 根目录（扫 history.json 用）
    "net_series": [],    # [{t, up, dn, ts}] 网卡上下行 KB/s（全机共享）
    "net_prev": None,
    "net_last": None,    # 最近一次有效 net 快照
}

# 多地区监控注册表（key -> 模型目录/数据目录/进度基线）。首个 key 是打开页面时的默认地区。
REGIONS = {
    "huanan_sw": {"model": r"E:\code\code tools\models\huanan_sw_mnv4",
                  "data": r"E:\code\code tools\data\huanan_sw",
                  "logdir": r"E:\code\code tools\data\huanan_sw",
                  "target": None, "baseline": 0, "label": "华南+西南"},
    "huabei":   {"model": r"E:\code\code tools\models\huabei_mnv4_300",
                 "data": r"E:\code\code tools\data\huabei",
                 "logdir": r"E:\code\code tools\data",
                 "target": 29650, "baseline": 12561, "label": "华北 300/种"},
    "huazhong": {"model": r"E:\code\code tools\models\huazhong_mnv4_25k",
                 "data": r"E:\code\code tools\data\huazhong",
                 "logdir": r"E:\code\code tools\data\huazhong",
                 "target": None, "baseline": 0, "label": "华中（河南等）",
                 "species_csv": r"E:\code\code tools\bird-cn-373\code\Bird-Classification-main\species_huazhong_150.csv"},
    "huadong":  {"model": r"E:\code\code tools\models\huadong_mnv4_300",
                 "data": r"E:\code\code tools\data\huadong",
                 "logdir": r"E:\code\code tools\data\huadong",
                 "target": None, "baseline": 0, "label": "华东（鲁苏皖浙赣闽）",
                 "species_csv": r"E:\code\code tools\data\huadong\species_huadong_150.csv"},
    "national": {"model": r"E:\code\code tools\models\national_mnv4_v1",
                  "data": r"E:\code\code tools\data\national",
                  "logdir": r"E:\code\code tools\data\national",
                  "target": None, "baseline": 0, "label": "全国兜底 373（自训，授权数据）",
                  "species_csv": r"E:\code\observebird\tools\national_clf\species_national_373.csv"},
}

# 地区中文名（--region 包模式从这里取 label；主开发机注册表自带 label 用不到它）
REGION_LABELS = {
    "huabei": "华北", "dongbei": "东北", "huazhong": "华中",
    "huadong": "华东", "huanan_sw": "华南+西南", "national": "全国",
}

def dynamic_target(key: str) -> int:
    """目标行数 = Σ min(300, 物种可用量)（GBIF 预取缓存）+ 各抓取器声明的增量
    （fetch_target_*.json，如 iNat 直连 / Commons 补充抓取）。"""
    try:
        cache = json.load(open(
            os.path.join(REGIONS[key]["data"], "gbif_prefetch_cache.json"),
            encoding="utf-8"))
        total = sum(min(300, v["n"]) for v in cache.values())
    except Exception:
        total = REGIONS[key]["target"] or 0
    import glob
    for p in glob.glob(os.path.join(REGIONS[key]["data"], "fetch_target_*.json")):
        try:
            total += int(json.load(open(p, encoding="utf-8"))["total"])
        except Exception:
            pass
    return total


def region_state(key: str) -> dict:
    """每地区的训练+下载子状态（不存在则初始化并尝试从盘上恢复）。"""
    with LOCK:
        rs = STATE["regions"].get(key)
    if rs is not None:
        return rs
    rs = {
        "model_dir": REGIONS[key]["model"], "data_dir": REGIONS[key]["data"],
        "logdir": REGIONS[key]["logdir"],
        "target": dynamic_target(key), "baseline": REGIONS[key]["baseline"],
        "train_start": time.time(),
        "samples": [], "pace": None, "last_mtime": 0.0, "last_epoch": -1,
        "dl_series": [], "rate_series": [], "rate_now": None,
        "done": False, "done_ts": 0.0, "history": None, "final": None,
        "train_log": [], "real_epoch": None, "epoch_float": None,
        "download": None,
    }
    state_path = os.path.join(rs["model_dir"], ".dashboard_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                snap = json.load(f)
            # 只恢复下载速率曲线；训练采样点不恢复——快照可能属于同目录里
            # 被中止的上一轮 run，混进来会让曲线出现"旧值点位"（实测踩过）
            for k2 in ("dl_series", "rate_series"):
                rs[k2] = snap.get(k2, [])
        except Exception:
            pass
    with LOCK:
        STATE["regions"][key] = rs
    return rs
LOCK = threading.Lock()
TOTAL_EPOCHS = 30


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def event(msg: str):
    with LOCK:
        STATE["events"].insert(0, {"ts": now(), "msg": msg})
        STATE["events"] = STATE["events"][:300]
    print(f"[{now()}] {msg}", flush=True)


def gpu_line() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip().replace("\n", " | ") or "n/a"
    except Exception:
        return "n/a"


def gpu_busy() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip())
    except Exception:
        return True


def read_ckpt(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def parse_download_log(path: str) -> dict:
    """解析 build_dataset_gbif.py 的日志：进度行 / API 行 / 失败计数。"""
    import re
    info = {"configured": True, "exists": os.path.exists(path), "stage": "运行中",
            "species_done": None, "total_species": 150, "cum": None,
            "eta_min": None, "fails": None, "tail": [], "last_line": None,
            "finished": False, "mtime": None}
    if not info["exists"]:
        return info
    try:
        raw = open(path, "rb").read()
    except Exception:
        return info
    # 日志几乎都是 UTF-8；个别手动启动没设 PYTHONIOENCODING 会写成 GBK
    # （实测：华北 20261004-183121），按 UTF-8 硬解会把中文关键词全变乱码
    text = None
    for enc in ("utf-8", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    info["mtime"] = datetime.datetime.fromtimestamp(
        os.path.getmtime(path)).strftime("%H:%M:%S")
    info["finished"] = os.path.exists(path + ".exit")
    if info["finished"]:
        info["stage"] = "已结束"
    info["tail"] = lines[-25:]
    info["fails"] = "\n".join(lines[-400:]).count("失败")
    for line in reversed(lines):
        m = re.search(r"\[\s*(\d+)/(\d+)\].*?用(\d+)\s+累计(\d+)\s+ETA\s+(\d+)分", line)
        if m:
            info["species_done"] = int(m.group(1))
            info["total_species"] = int(m.group(2))
            info["cum"] = int(m.group(4))
            info["eta_min"] = int(m.group(5))
            if not info["finished"]:
                info["stage"] = "下载中"
            break
    for line in reversed(lines):
        if "候选" in line:
            info["last_line"] = line.strip()
            break
    # ---- 预取探测阶段：进度 + 完成报告 + 数据量警告 ----
    # [预取] 行逐种一条（含缓存命中），去重即探测进度；「合计可用记录」行标志探测完成
    total = None
    probed = set()
    report = None
    for line in lines:
        m = re.search(r"物种\s*(\d+)\s*种", line)
        if m:
            total = int(m.group(1))
        m = re.search(r"\[预取\]\s+(\S+)", line)
        if m:
            probed.add(m.group(1))
        m = re.search(r"合计可用记录\s*(\d+)\s*\|\s*有数据的种\s*(\d+)/(\d+)", line)
        if m:
            report = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if probed or report:
        avail, with_data = (report[0], report[1]) if report else (None, None)
        t = report[2] if report else (total or 150)
        warn = None
        if report:
            # 阈值对照实跑可用量：华北 5.2万 / 华南 5.5万（健康），华中 0.9万（真实稀疏先例）
            if with_data < 0.8 * t or avail < 5000:
                warn = "critical"   # 疑似 bbox/物种清单/网络配置错误
            elif avail < 20000:
                warn = "low"        # 量偏少但有先例：下载继续，训练走类平衡
        info["probe"] = {"stage": "done" if report else "probing",
                         "probed": len(probed), "total": t,
                         "available": avail, "with_data": with_data,
                         "warn": warn}
        if not report and info["species_done"] is not None:
            # 下载进度行已出现 → 串行链条里探测必然已结束；本日志缺合计行
            # （开头被截断/历史日志）时也不得停在"探测中"误导
            info["probe"]["stage"] = "done"
    if info["species_done"] is None and not info["finished"] \
            and info.get("probe") and info["probe"]["stage"] == "probing":
        info["stage"] = "预取中"
    return info


def update_train(key: str, rs: dict):
    """训练状态采集：best.pt 破纪录 / history.json 完成 / train.log 真实进度。"""
    import re
    out_dir = rs["model_dir"]
    ckpt_path = os.path.join(out_dir, "best.pt")
    hist_path = os.path.join(out_dir, "history.json")
    log_path = os.path.join(out_dir, "train.log")

    mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0.0
    if mtime and mtime != rs["last_mtime"]:
        d = read_ckpt(ckpt_path)
        rs["last_mtime"] = mtime
        if d and d.get("epoch", -1) > rs["last_epoch"]:
            rs["last_epoch"] = d["epoch"]
            rs["samples"].append({
                "epoch": d["epoch"],
                "top1": round(d["top1"] * 100, 2),
                "top3": round(d["top3"] * 100, 2),
                "top5": round(d["top5"] * 100, 2),
                "ts": now(),
            })
            if rs["pace"] is None and d["epoch"] > 0:
                rs["pace"] = max(1.0, time.time() - rs["train_start"]) / d["epoch"]
            event(f"[{key}] epoch {d['epoch']}/{TOTAL_EPOCHS} 新纪录: "
                  f"top1 {d['top1']*100:.2f}% top3 {d['top3']*100:.2f}% "
                  f"top5 {d['top5']*100:.2f}%")
        try:
            with open(os.path.join(out_dir, ".dashboard_state.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"samples": rs["samples"], "pace": rs["pace"],
                           "last_mtime": rs["last_mtime"]}, f, ensure_ascii=False)
        except Exception:
            pass

    # 同目录重训检测（v2 复用 national_mnv4_v1 目录口径）：已 done 后 best.pt 又被
    # 改写且晚于完成锚点 done_ts（=上轮 history.json 的 mtime）→ 新一轮开始，
    # 重置训练态按本轮重新采集；否则卡片永远冻结在上一轮的「已完成」
    if rs["done"] and mtime > rs.get("done_ts", 0.0):
        rs.update(done=False, history=None, final=None, samples=[], pace=None,
                  last_epoch=-1, epoch_float=None, train_start=time.time())
        event(f"[{key}] 检测到同目录新一轮训练（best.pt 已更新），监控态已重置")
    if rs["done"]:
        return
    # 完成判定：history.json 出现即完成（不用 GPU 判定——桌面进程常驻会误判）。
    # done 锚点记 history.json 的 mtime：同目录重训进行中，旧 history.json 的
    # mtime 不会晚于锚点 → 不冻结，继续走下面的 train.log 实时解析
    if os.path.exists(hist_path):
        h_mtime = os.path.getmtime(hist_path)
        if h_mtime <= rs.get("done_ts", 0.0):
            pass
        else:
            try:
                with open(hist_path, encoding="utf-8") as f:
                    h = json.load(f)
            except Exception:
                return
            rs["done"] = True
            rs["done_ts"] = h_mtime
            rs["history"] = h["history"]
            rs["final"] = {"best_top1": round(h["best_top1"] * 100, 2),
                           "n_images": h["n_images"], "n_species": h["n_species"]}
            event(f"[{key}] 训练完成! best_top1={h['best_top1']*100:.2f}% "
                  f"({h['n_images']} 张 / {h['n_species']} 类)")
            return

    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                log_lines = f.read().splitlines()[-120:]
        except Exception:
            log_lines = []
        # train.log 追加写、每轮以「=== run」分隔行开头（log() 会加 HH:MM:SS 前缀，
        # 故用包含匹配）：切出最后一条分隔行之后的行（本轮）——否则同目录重训时
        # 上一轮的 epoch 行 / 「完成 best」会混进曲线与进度判定
        for _i in range(len(log_lines) - 1, -1, -1):
            if "=== run" in log_lines[_i]:
                log_lines = log_lines[_i:]
                break
        real_epoch = None
        for line in reversed(log_lines):
            m = re.search(r"ep\s*(\d+)/", line)
            if m:
                real_epoch = int(m.group(1)); break
        # 逐 epoch 曲线以 train.log 为准重建（120 行覆盖整个 run）：按行序
        # 后行覆盖前行，同 epoch 上重启后的新 run 行会替换旧 run 的残留值；
        # best.pt 快照只补日志窗口之外的点，不再与之混接
        log_pts = {}
        for line in log_lines:
            m = re.search(r"ep\s*(\d+)/\d+\s+.*val\s+top1\s+([\d.]+)%\s+top3\s+([\d.]+)%"
                          r"\s+top5\s+([\d.]+)%", line)
            if m:
                log_pts[int(m.group(1))] = {"epoch": int(m.group(1)),
                                            "top1": float(m.group(2)),
                                            "top3": float(m.group(3)),
                                            "top5": float(m.group(4)),
                                            "ts": line.split()[0]}
        if log_pts:
            snaps = {x["epoch"]: x for x in rs["samples"]}
            snaps.update(log_pts)
            rs["samples"] = sorted(snaps.values(), key=lambda x: x["epoch"])
        else:
            rs["samples"].sort(key=lambda x: x["epoch"])
        rs["train_log"] = log_lines
        rs["real_epoch"] = real_epoch
        done_training = any("完成 best" in l for l in log_lines)
        for line in reversed(log_lines):
            m = re.search(r"^(\d{2}:\d{2}:\d{2})\s+ep\s*(\d+)/(\d+)\s+.*\((\d+)s\)\s*$", line)
            if m:
                try:
                    t = datetime.datetime.strptime(m.group(1), "%H:%M:%S")
                    ts = datetime.datetime.now().replace(
                        hour=t.hour, minute=t.minute, second=t.second)
                    dur = int(m.group(4)); total = int(m.group(3))
                    frac = max(0.0, min(1.0, (time.time() - ts.timestamp()) / dur))
                    rs["epoch_float"] = float(total) if done_training \
                        else min(int(m.group(2)) + frac, float(total))
                except Exception:
                    pass
                break


def update_download(key: str, rs: dict):
    """下载状态采集：日志解析 + dataset.csv 实时行数 + 速率序列（per-region）。"""
    import glob
    cands = [p for p in glob.glob(os.path.join(rs["logdir"], "gbif_build.log.*"))
             if not p.endswith((".err", ".exit"))]
    if cands:
        info = parse_download_log(max(cands, key=os.path.getmtime))
    else:
        # 无构建日志（自训管线的 progress.json 由驱动心跳镜像到 data_dir）：
        # stage=download → 抓取中（透出 species_done/total_species 供前端渲染），
        # stage=done → 已结束
        finished = False
        stage_txt, sp_done, sp_tot = "等待数据", None, None
        try:
            prog = os.path.join(rs["data_dir"], "progress.json")
            p = json.load(open(prog, encoding="utf-8"))
            finished = p.get("stage") == "done"
            sp_done, sp_tot = p.get("species_done"), p.get("species_total")
            if finished:
                stage_txt = "已结束"
            elif p.get("stage") == "download":
                stage_txt = "抓取中"
        except Exception:
            pass
        info = {"configured": True, "exists": False,
                "stage": stage_txt,
                "species_done": sp_done, "total_species": sp_tot, "cum": None,
                "eta_min": None, "fails": None, "tail": [], "last_line": None,
                "finished": finished, "mtime": None}
    ds = os.path.join(rs["data_dir"], "dataset.csv")
    if os.path.exists(ds):
        try:
            with open(ds, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()[1:]
            info["rows"] = len(lines)
            info["species_live"] = len({l.split(",")[1] for l in lines if l and "," in l})
            info["idle_sec"] = round(time.time() - os.path.getmtime(ds))
            # 分源行数（dataset_key 列）：iNat 直连抓取的行
            info["rows_inat"] = sum(
                1 for l in lines
                if len(l.split(",")) > 11 and l.split(",")[11] == "inat_api")
        except Exception:
            pass
    # Commons 补充抓取：独立 CSV + 自报目标
    cs = os.path.join(rs["data_dir"], "dataset_commons.csv")
    if os.path.exists(cs):
        try:
            with open(cs, encoding="utf-8") as f:
                info["rows_commons"] = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    for src in ("inat", "commons"):
        p = os.path.join(rs["data_dir"], f"fetch_target_{src}.json")
        try:
            j = json.load(open(p, encoding="utf-8"))
            info[f"target_{src}"] = int(j.get("total") or 0)
            if j.get("baseline_rows") is not None:
                info["fetch_baseline"] = int(j["baseline_rows"])
        except Exception:
            pass
    # 双来源进度条（iNat 直连 + Commons 代理）只在这些产物存在的地区显示
    # （目前 = 华中），其他地区不渲染"未启动"占位条
    info["dual_source"] = bool(info.get("target_inat") or info.get("target_commons")
                               or info.get("rows_commons") is not None)
    info["target"] = rs["target"]; info["baseline"] = rs["baseline"]
    # 「本轮补量」口径修正：抓取器轮次（fetch_target_inat.json 存在）的目标 =
    # 轮起点行数 + 本轮增量，基线必须锚在轮起点——否则此前 GBIF 已落盘的行会被
    # 算进本轮，进度条一开局就跳到一半（实测：华中 iNat 轮开局即 ~46%）
    if info.get("target_inat") is not None:
        fb = info.get("fetch_baseline")
        if fb is None:  # 老格式文件无 baseline_rows：回退 = 当前行数 − iNat 行
            fb = (info.get("rows") or 0) - (info.get("rows_inat") or 0)
        info["baseline"] = fb
        info["target"] = fb + (info["target_inat"] or 0)
    pr = info.get("probe")
    # 按报告内容签名去重：同一份报告只报一次；链条重跑产生新报告（数字变了）再报
    if pr and pr["stage"] == "done" \
            and rs.get("probe_reported_sig") != (pr["available"], pr["with_data"]):
        rs["probe_reported_sig"] = (pr["available"], pr["with_data"])
        if pr["warn"]:
            detail = ("疑似 bbox/物种清单/网络配置问题，请核对下载日志 [预取] 行并联系用户"
                      if pr["warn"] == "critical" else
                      "明显低于华北/华南的 5 万+ 水平（华中 0.9 万为先例，训练可走类平衡）")
            event(f"[{key}] ⚠ 探测报告：合计可用 {pr['available']} 条 / "
                  f"有数据 {pr['with_data']}/{pr['total']} 种 —— {detail}（下载继续，不受影响）")
    tnow = time.time()
    # 下载已结束 → 冻结速率序列（不再加平线点，保留历史高峰形状）
    sampling = info.get("rows") is not None and not info.get("finished")
    if sampling:
        ser = rs["dl_series"] + [{"t": tnow, "rows": info["rows"]}]
        ser = [p for p in ser if tnow - p["t"] <= 3600][-500:]
        rate = None
        if ser and tnow - ser[0]["t"] >= 60:
            rate = max(0.0, (ser[-1]["rows"] - ser[0]["rows"]) / ((tnow - ser[0]["t"]) / 60))
        if rate is not None:
            rs["rate_now"] = round(rate, 1)
            if not rs["rate_series"] or tnow - rs["rate_series"][-1]["t"] >= 15:
                rs["rate_series"] = (rs["rate_series"] +
                    [{"t": tnow, "r": round(rate, 1), "ts": time.strftime("%H:%M:%S")}])[-240:]
        info["rate_now"] = rs["rate_now"]
        info["rate_series"] = [{"ts": p["ts"], "r": p["r"], "t": p["t"]}
                               for p in rs["rate_series"]]
        rs["dl_series"] = ser
    info["net"] = STATE.get("net_last")
    rs["download"] = info


def collect_net() -> dict | None:
    """网卡上下行速率（KB/s）：psutil 字节计数差值，5s 窗口，全机共享。"""
    try:
        import psutil
        cnt = psutil.net_io_counters()
        tnow = time.time()
        prev = STATE.get("net_prev")
        if prev and tnow - prev[0] >= 5:
            dt = tnow - prev[0]
            up = (cnt.bytes_sent - prev[1]) / dt / 1024
            dn = (cnt.bytes_recv - prev[2]) / dt / 1024
            with LOCK:
                STATE["net_prev"] = (tnow, cnt.bytes_sent, cnt.bytes_recv)
                ns = STATE["net_series"]
                if not ns or tnow - ns[-1]["t"] >= 10:
                    ns = (ns + [{"t": tnow, "up": round(max(0, up), 1),
                                 "dn": round(max(0, dn), 1),
                                 "ts": time.strftime("%H:%M:%S")}])[-360:]
                    STATE["net_series"] = ns
                net = {"up": round(max(0, up), 1), "dn": round(max(0, dn), 1),
                       "series": [{"ts": p["ts"], "up": p["up"], "dn": p["dn"],
                                   "t": p["t"]} for p in STATE["net_series"]]}
                STATE["net_last"] = net
                return net
            return STATE.get("net_last")
        if not prev:
            with LOCK:
                STATE["net_prev"] = (tnow, cnt.bytes_sent, cnt.bytes_recv)
    except Exception:
        pass
    return STATE.get("net_last")


def poller(interval: float):
    for key in REGIONS:
        region_state(key)
    STATE["models_root"] = None
    vals = list(REGIONS.values())
    if vals:
        STATE["models_root"] = os.path.dirname(os.path.normpath(vals[0]["model"]))
    event(f"监控启动：地区 {', '.join(REGIONS)}")
    while True:
        gpu = gpu_line()
        net = collect_net()
        for key, rs in list(STATE["regions"].items()):
            try:
                update_download(key, rs)
                update_train(key, rs)
            except Exception as e:
                event(f"[{key}] 采集异常: {e}")
        with LOCK:
            STATE["gpu"] = gpu
            if net:
                STATE["net_last"] = net
        time.sleep(interval)


def collect_runs(models_root: str, region_states: list, region: str = "") -> list:
    """收集全部训练 run：已完成读 history.json，进行中的从各地区 train.log 解析。
    region 非空时只保留该地区前缀的 run（历史对比页按地区隔离）。"""
    import re
    runs = []
    if models_root and os.path.isdir(models_root):
        for d in sorted(os.listdir(models_root)):
            if d.startswith("_smoke"):  # 冒烟实验不入对比
                continue
            if region and not d.startswith(region):  # 地区隔离
                continue
            hp = os.path.join(models_root, d, "history.json")
            if not os.path.exists(hp):
                continue
            try:
                with open(hp, encoding="utf-8") as f:
                    h = json.load(f)
                runs.append({
                    "name": d, "done": True,
                    "best": round(h["best_top1"] * 100, 2),
                    "n_images": h.get("n_images"), "n_species": h.get("n_species"),
                    "history": [{"epoch": e["epoch"],
                                 "top1": round(e["top1"] * 100, 2),
                                 "top3": round(e["top3"] * 100, 2),
                                 "top5": round(e["top5"] * 100, 2)}
                                for e in h["history"]],
                })
            except Exception:
                continue
    done_names = {r["name"] for r in runs}
    for name, log_lines in region_states:
        if region and not name.startswith(region):
            continue  # 地区隔离：别区训练中的曲线不混进本区对比页
        hist, best = [], 0.0
        for line in log_lines:
            m = re.search(
                r"ep\s*(\d+)/\d+\s+.*val\s+top1\s+([\d.]+)%\s+top3\s+([\d.]+)%"
                r"\s+top5\s+([\d.]+)%", line)
            if m:
                t1 = float(m.group(2))
                best = max(best, t1)
                hist.append({"epoch": int(m.group(1)), "top1": t1,
                             "top3": float(m.group(3)), "top5": float(m.group(4))})
        if hist:
            runs = [r for r in runs if r["name"] != name]
            runs.append({"name": name + "（训练中）", "done": False,
                         "best": round(best, 2), "n_images": None,
                         "n_species": None, "history": hist})
    return runs


def status_json(region: str) -> str:
    if region not in REGIONS:
        region = next(iter(REGIONS), None)
        if region is None:
            return json.dumps({"error": "no regions configured", "regions": []},
                              ensure_ascii=False)
    # region_state 按需初始化：/api/status 在采集线程首轮前到来时也不至于串区
    rs = STATE["regions"].get(region) or region_state(region)
    if rs is None:
        return json.dumps({"error": "no regions"}, ensure_ascii=False)
    with LOCK:
        samples = list(rs["samples"])
        best_epoch = samples[-1]["epoch"] if samples else 0
        real_epoch = rs.get("real_epoch")
        epoch_float = rs.get("epoch_float")
        elapsed_tr = time.time() - rs["train_start"]
        if epoch_float:
            est = int(epoch_float)
        elif real_epoch:
            est = max(best_epoch, real_epoch)
        else:
            est = best_epoch
            if rs["pace"]:
                est = max(best_epoch, min(TOTAL_EPOCHS, int(elapsed_tr / rs["pace"])))
        if rs["done"]:
            est = TOTAL_EPOCHS
        total_pct = (round(epoch_float / TOTAL_EPOCHS * 100, 1)
                     if epoch_float else round(est / TOTAL_EPOCHS * 100, 1))
        payload = {
            "now": now(),
            "now_ts": time.time(),
            "region": region,
            "regions": list(REGIONS.keys()),
            "region_label": REGIONS[region]["label"],
            "elapsed_min": round((time.time() - STATE["started"]) / 60, 1),
            "total_epochs": TOTAL_EPOCHS,
            "est_epoch": est,
            "real_epoch": real_epoch,
            "epoch_float": round(epoch_float, 2) if epoch_float else None,
            "epoch_pct": total_pct,
            "pace_sec": rs["pace"],
            "progress_pct": total_pct,
            "overrun": est >= TOTAL_EPOCHS and not rs["done"] and not epoch_float,
            "best_epoch": best_epoch,
            "samples": samples,
            "events": STATE["events"][:80],
            "gpu": STATE["gpu"],
            "done": rs["done"],
            "final": rs["final"],
            "history": rs["history"],
            "train_log": rs.get("train_log", []),
            "download": rs.get("download"),
            "commons_running": commons_running(),
            "runs": collect_runs(
                STATE.get("models_root"),
                [(k, r.get("train_log", []))
                 for k, r in STATE["regions"].items()
                 if not r["done"]], region),  # 含本区训练中的 run：重训可与历史对比
        }
    return json.dumps(payload, ensure_ascii=False)


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>ObserveBird 监控中心</title>
<style>
  body{background:#111;color:#ddd;font-family:system-ui,"Microsoft YaHei",sans-serif;
       margin:24px;max-width:960px;margin-inline:auto}
  h1{font-size:20px;color:#fff} h1 small{color:#888;font-weight:normal}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
         gap:10px;margin:16px 0}
  .card{background:#1d1d1d;border-radius:12px;padding:14px 16px;min-width:0}
  .card .v{font-size:26px;font-weight:700;color:#4ade80;white-space:nowrap;
           overflow:hidden;text-overflow:ellipsis}
  .card .l{font-size:12px;color:#888;margin-top:4px}
  #bar{background:#333;border-radius:8px;height:22px;overflow:hidden;margin:10px 0;
      position:relative}
  #bar>div{background:linear-gradient(90deg,#4ade80,#22d3ee);height:100%;width:0%}
  #barlabel{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
      font-size:12px;color:#fff;text-shadow:0 1px 2px #000}
  #epbar{display:flex;gap:2px;margin:8px 0 2px}
  #epbar i{flex:1;height:18px;background:#2b2b2b;border-radius:3px}
  #epbar i.f{background:#4ade80}
  #epbar i.cur{background:#fbbf24;box-shadow:0 0 6px #fbbf24}
  svg{background:#1d1d1d;border-radius:12px;width:100%;height:260px}
  #log{background:#1a1a1a;border-radius:12px;padding:10px 14px;height:220px;
       overflow-y:auto;font:12px/1.7 Consolas,monospace;color:#9ca3af}
  #tlog{background:#14181d;border-radius:12px;padding:10px 14px;height:200px;
       overflow-y:auto;font:12px/1.7 Consolas,monospace;color:#86efac}
  #log b{color:#e5e7eb}
  .done{color:#4ade80;font-size:18px;font-weight:700}
  .muted{color:#666;font-size:12px}
  #state{border-radius:12px;padding:12px 18px;margin:12px 0;font-size:16px;font-weight:700}
  .st-training{background:#1e3a5f;color:#7dd3fc}
  .st-overrun{background:#4a3208;color:#fbbf24}
  .st-done{background:#14321f;color:#4ade80}
  #tabs{display:flex;gap:8px;border-bottom:1px solid #333;margin:12px 0 0}
  .tab{flex:1;padding:10px 0;background:none;border:none;border-bottom:2px solid transparent;
       cursor:pointer;font-size:14px;color:#888;font-family:inherit}
  .tab:hover{color:#ccc}
  .tab.active{color:#4ade80;border-bottom-color:#4ade80;font-weight:600}
  .panel{display:none}
  .panel.active{display:block}
  .bar2wrap{background:#333;border-radius:8px;height:20px;overflow:hidden;position:relative}
  #bar2{height:100%;width:0%;background:linear-gradient(90deg,#22d3ee,#4ade80)}
  #probebar{height:100%;width:0%;background:linear-gradient(90deg,#a78bfa,#22d3ee)}
  #bar2label,#probelabel{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
      font-size:12px;color:#fff;text-shadow:0 1px 2px #000;pointer-events:none}
  .srcbar{margin:8px 0 0}
  .srcbar .lbl{display:flex;justify-content:space-between;font-size:12px;color:#aaa;
      margin-bottom:3px}
  .srcbar .track{background:#333;border-radius:6px;height:14px;overflow:hidden}
  .srcbar .fill{height:100%;width:0%;border-radius:6px;transition:width .6s}
  .srcbtn{background:#4ade80;color:#052e16;border:none;border-radius:14px;
      padding:4px 14px;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit}
  .srcbtn:hover{background:#86efac}
  .srcbtn:disabled{opacity:.45;cursor:default}
  .dllogbox{background:#1a1a1a;border-radius:12px;padding:10px 14px;height:260px;
       overflow-y:auto;font:12px/1.7 Consolas,monospace;color:#9ca3af;
       white-space:pre-wrap;word-break:break-all}
  /* 下载面板：学习 _served.html 的 kpi + canvas 曲线 */
  .card .v.r{color:#f87171}
  .card .v.g{color:#4ade80}
  .kpi-lbl{font-size:12px;color:#888;margin-top:4px}
  .alert{padding:10px 14px;border-radius:10px;margin:10px 0;font-size:13px;display:none}
  .alert.on{display:block}
  .alert.wa{background:#3a2d08;color:#fbbf24}
  .alert.bad{background:#3a1212;color:#f87171}
  #state{display:flex;align-items:center;justify-content:space-between;gap:12px}
  .stopbtn{background:#3a1212;color:#f87171;border:1px solid #7f1d1d;border-radius:10px;
       padding:8px 18px;font-size:13px;cursor:pointer;font-family:inherit;margin:8px 0}
  .stopbtn.inline{margin:0;padding:4px 14px;font-size:12.5px;
       background:rgba(255,255,255,.12);border-color:currentColor;color:inherit}
  .stopbtn.inline.armed{background:#f87171;color:#1a0505;border-color:#f87171}
  .stopbtn:hover{background:#4a1616}
  .stopbtn.armed{background:#f87171;color:#1a0505;font-weight:700;border-color:#f87171}
  .stopbtn:disabled{opacity:.35;cursor:default}
  #pan-dl .chartcard{background:#1d1d1d;border-radius:12px;padding:12px;margin:10px 0}
  #pan-dl .chartcard h3{margin:0 0 8px;font-size:14px}
  #ratecv{width:100%;height:110px;display:block}
  .rates-line{font-size:12px;color:#8a8a8a;margin-top:6px}
  #pan-cmp .chartcard h3{margin:0 0 8px;font-size:14px}
  #cmptable table{width:100%;border-collapse:collapse;font-size:12.5px}
  #cmptable th{color:#888;font-weight:500;font-size:11.5px;text-align:left;
               padding:4px 6px;border-bottom:1px solid #333}
  #cmptable td{padding:3px 6px;border-bottom:1px solid #262626;color:#ddd}
  #cmptable td.r,#cmptable th.r{text-align:right;font-variant-numeric:tabular-nums}
  #cmptable tr:nth-child(3n) td{border-bottom-color:#3a3a3a}
</style></head><body>
<h1>ObserveBird 监控中心
  <select id="regionSel" style="background:#262626;color:#ddd;border:1px solid #444;
    border-radius:8px;padding:4px 10px;font-size:13px;font-family:inherit;vertical-align:middle">
  </select>
  <small id="sub"></small></h1>
<div id="tabs">
  <button class="tab active" data-t="train">训练监控</button>
  <button class="tab" data-t="dl">数据下载</button>
  <button class="tab" data-t="cmp">历史对比</button>
</div>
<div class="panel active" id="pan-train">
<div id="state" class="st-training">
  <span id="stateTxt">训练中，请稍候…</span>
  <button class="stopbtn inline" id="stopTrain" data-label="⏹ 停止"
    onclick="armStop(this,'train')">⏹ 停止</button>
</div>
<div class="cards">
  <div class="card"><div class="v" id="top1">--</div><div class="l">best val top-1 · 首选即正确</div></div>
  <div class="card"><div class="v" id="top3">--</div><div class="l">top-3 · 前三含正确</div></div>
  <div class="card"><div class="v" id="top5">--</div><div class="l">top-5 · 前五含正确</div></div>
  <div class="card"><div class="v" id="curep">--</div><div class="l">当前 epoch（估计）</div></div>
  <div class="card"><div class="v" id="epoch">--</div><div class="l">破纪录 epoch（checkpoint）</div></div>
  <div class="card"><div class="v" id="gpu" style="font-size:16px">--</div><div class="l">GPU</div></div>
</div>
<div id="epbar"></div>
<div class="muted" id="eplabel"></div>
<div id="bar"><div></div><span id="barlabel">0/30 · 0%</span></div>
<p class="muted">指标含义：模型对每张验证图给出 146 类的排序。top-1 = 排第一的就是正确答案的比例；
top-3 / top-5 = 正确答案出现在前三 / 前五的比例（App 交互就是给用户 Top-5 候选再确认，所以 top-5 同样关键）。
进度条为按训练节奏的<b>估计值</b>，「破纪录 epoch」卡片是 checkpoint 里的实际值。</p>
<svg id="chart" viewBox="0 0 900 260" preserveAspectRatio="none"></svg>
<p class="muted">基线参考：全国 373 种模型 top-1 = 32.39% ｜ 判定参考：&gt;50% 走区域化全量，&lt;40% 换方向（数据量混淆见 SESSION §7）</p>
<h3>事件日志</h3><div id="log"></div>
<h3>训练日志（train.log 尾部，逐 epoch 实时）</h3><div id="tlog"></div>
</div>
<div class="panel" id="pan-dl">
  <div class="alert wa" id="dl-al"></div>
  <div class="alert wa" id="probe-warn"></div>
  <button class="stopbtn" id="stopDl" data-label="⏹ 停止补量下载（断点可续）"
    onclick="armStop(this,'download')">⏹ 停止补量下载（断点可续）</button>
  <div class="cards">
    <div class="card"><div class="v" id="dl_rows">--</div><div class="kpi-lbl">本轮补量 / 补量上限</div></div>
    <div class="card"><div class="v" id="dl_total">--</div><div class="kpi-lbl">数据集累计（张）</div></div>
    <div class="card"><div class="v" id="dl_species">--</div><div class="kpi-lbl">已落图物种 / 150</div></div>
    <div class="card"><div class="v" id="dl_rate">--</div><div class="kpi-lbl" id="dl_rate_lbl">张/分</div></div>
    <div class="card"><div class="v" id="dl_eta">--</div><div class="kpi-lbl">预计还需（按当前速率）</div></div>
  </div>
  <div class="bar2wrap"><div id="bar2"><div></div></div><span id="bar2label"></span></div>
  <div class="bar2wrap" id="probewrap" style="display:none"><div id="probebar"><div></div></div><span id="probelabel"></span></div>
  <div id="srcbars"></div>
  <div class="muted" id="dlcaption" style="margin:6px 0 4px"></div>
  <div class="chartcard">
    <h3>吞吐曲线（最近 60 分钟，右侧为当前时刻）</h3>
    <canvas id="ratecv" width="1040" height="220"></canvas>
    <div class="rates-line" id="rates"></div>
  </div>
  <h3>下载日志（尾部 25 行）</h3><div id="dllog" class="dllogbox"></div>
</div>
<div class="panel" id="pan-cmp">
  <div class="chartcard">
    <h3 style="display:flex;align-items:center;justify-content:space-between;gap:10px">
      <span>各轮训练曲线叠加（x = epoch，悬停看数值）</span>
      <select id="cmpMetric" onchange="renderCmp(window.__runs||[])"
        style="background:#262626;color:#ddd;border:1px solid #444;border-radius:8px;
               padding:4px 8px;font-size:13px;font-family:inherit">
        <option value="top1" selected>val top-1</option>
        <option value="top3">val top-3</option>
        <option value="top5">val top-5</option>
      </select>
    </h3>
    <svg id="cmpchart" viewBox="0 0 900 280" preserveAspectRatio="none"></svg>
  </div>
  <div id="cmpcards" class="cards"></div>
  <div class="chartcard">
    <h3>逐 epoch 对比：本轮 vs 上一轮完成 run（Δ = 本轮 − 上一轮，正值=提升）</h3>
    <div id="cmptable" style="max-height:340px;overflow-y:auto"></div>
  </div>
</div>
<div id="probeModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.66);
  z-index:99;align-items:center;justify-content:center">
  <div style="background:#1c1c1c;border:1px solid #444;border-radius:14px;max-width:540px;
    margin:16px;padding:22px 26px;box-shadow:0 8px 40px rgba(0,0,0,.6)">
    <h3 id="pmTitle" style="margin:0 0 10px;font-size:17px"></h3>
    <p id="pmBody" style="margin:0 0 10px;font-size:13.5px;line-height:1.7"></p>
    <p class="muted" style="font-size:12px;margin:0 0 16px">下载继续、不受影响；本次打开页面只提醒这一次。</p>
    <button onclick="closeProbeModal()" style="background:#4ade80;color:#06220e;border:0;
      border-radius:10px;padding:9px 22px;font-size:13.5px;font-weight:600;cursor:pointer;
      font-family:inherit">我知道了，继续监控</button>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
/* 标签切换（参考 workshop.html：TABS 数组 + activate 统一切换 + hash 定位） */
const TABS=[['train','pan-train'],['dl','pan-dl'],['cmp','pan-cmp']];
function activate(name){
  TABS.forEach(function(t){
    document.querySelector('.tab[data-t="'+t[0]+'"]').classList.toggle('active', t[0]===name);
    document.getElementById(t[1]).classList.toggle('active', t[0]===name);
  });
}
document.querySelectorAll('.tab').forEach(function(b){
  b.addEventListener('click', function(){
    activate(b.getAttribute('data-t'));
    history.replaceState(null, '', '#'+b.getAttribute('data-t'));
  });
});
/* 停止按钮：两段式确认（3 秒内再点执行），防误触 */
/* 手动启动 Commons 抓取：POST /api/start，后端拉起脱离进程 */
async function startCommons(btn){
  btn.disabled=true; btn.textContent='启动中…';
  try{
    const j=await (await fetch('/api/start',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({region:window.__curRegion||''})})).json();
    btn.textContent=j.started?'已启动 ✓':(j.msg||'未启动');
  }catch(e){ btn.textContent='失败：'+e; }
  setTimeout(()=>{btn.disabled=false;btn.textContent='▶ 启动';},5000);
}
async function armStop(btn, target){
  if(btn.dataset.armed==='1'){
    btn.dataset.armed='0';
    btn.textContent='停止中…';
    try{
      const j=await (await fetch('/api/stop',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({target})})).json();
      btn.textContent=j.killed&&j.killed.length
        ? `已停止（${j.killed.length} 个进程）`:'没有在运行的进程';
      btn.disabled=true;
      setTimeout(()=>{btn.disabled=false;btn.textContent=btn.dataset.label;},6000);
    }catch(e){ btn.textContent='停止失败：'+e; }
  }else{
    btn.dataset.armed='1';
    btn.textContent='⚠ 确认停止？再点一次执行';
    btn.classList.add('armed');
    setTimeout(()=>{ if(btn.dataset.armed==='1'){
      btn.dataset.armed='0';btn.classList.remove('armed');btn.textContent=btn.dataset.label;}},3000);
  }
}
function showProbeModal(pb){
  const crit=pb.warn==='critical';
  $('pmTitle').textContent=crit?'⚠ 探测结果异常：数据量明显过少':'⚠ 探测完成：数据量偏少';
  $('pmTitle').style.color=crit?'#f87171':'#fbbf24';
  $('pmBody').innerHTML=crit
    ?`GBIF 预检合计可用 <b>${(pb.available??0).toLocaleString()}</b> 条 / 有数据 <b>${pb.with_data??'--'}/${pb.total}</b> 种——这个量级更像 <b>bbox、物种清单或网络配置出了问题</b>（健康参考：华北 5.2 万 / 华南 5.5 万可用）。下载会继续、不受影响；请翻看下载日志的 [预取] 行核对可用量分布，并尽快联系用户确认。`
    :`GBIF 预检合计可用 <b>${(pb.available??0).toLocaleString()}</b> 条 / 有数据 <b>${pb.with_data??'--'}/${pb.total}</b> 种，低于华北/华南的 5 万+ 水平（先例：华中约 0.9 万，真实稀疏，已实跑通过）。下载继续、不受影响；量少训练也相应变快，回传后主开发机会按类平衡处理。`;
  $('probeModal').style.display='flex';
}
function closeProbeModal(){$('probeModal').style.display='none';}
function renderDl(dl){
  if(!dl||dl.configured===false){$('dl_rows').textContent='未配置';return;}
  const rows=dl.rows, target=dl.target, base=dl.baseline;
  const cap = target&&base!=null ? target-base : null;
  const inc = (rows!=null&&base!=null) ? rows-base : null;
  let pct=null;
  if(inc!=null&&cap){
    pct=Math.max(0,Math.min(100,Math.round(inc/cap*1000)/10));
    $('bar2').style.width=pct+'%';
    $('bar2label').textContent='本轮补量 '+pct.toFixed(1)+'%';
  }
  $('dl_rows').textContent=(inc!=null&&cap)?('+'+inc.toLocaleString()+' / '+cap.toLocaleString()):'--';
  $('dl_total').textContent=rows!=null?rows.toLocaleString()+(target?' / '+target.toLocaleString():''):'--';
  $('dlcaption').textContent=(rows!=null&&target&&base!=null)
    ? `数据集 ${rows.toLocaleString()}/${target.toLocaleString()} = ${(100*rows/target).toFixed(1)}% ｜ 完成物种 ${dl.species_done??'--'}/${dl.total_species??'--'} ｜ 近期失败 ${dl.fails??'-'} 次`
    : '';
  const sp = dl.species_live!=null ? dl.species_live : dl.species_done;
  $('dl_species').textContent = sp!=null ? sp : '--';
  // 速率卡：数值着色 + 动态标签
  const rv=dl.rate_now;
  const rateEl=$('dl_rate');
  rateEl.textContent=rv==null?'–':(rv>=100?Math.round(rv):rv.toFixed(1));
  rateEl.className='v '+(rv==null?'':(rv>100?'g':(rv<20?'r':'')));
  $('dl_rate_lbl').textContent=rv==null?'张/分（攒历史中）':'近 5 分钟 张/分';
  // ETA 卡：预计还需时间
  let etaTxt='–';
  if(!dl.finished&&rv>0&&rows!=null&&target&&target>rows){
    const etaMin=(target-rows)/rv;
    etaTxt=etaMin>=60?(etaMin/60).toFixed(1)+' 小时':Math.max(1,Math.round(etaMin))+' 分钟';
  }else if(dl.finished){etaTxt='已完成';}
  $('dl_eta').textContent=etaTxt;
  const dlBtn=$('stopDl');
  if(dl.finished){dlBtn.disabled=true;dlBtn.textContent='✓ 下载已结束';}
  else if(dlBtn.textContent.startsWith('✓')&&dlBtn.disabled){dlBtn.disabled=false;dlBtn.textContent=dlBtn.dataset.label;}
  // 双源进度条：iNat 直连 + Commons 补充（各自 rows/target，独立计数）
  const srcs=[
    {name:'iNat 直连', rows:dl.rows_inat, target:dl.target_inat,
     color:'linear-gradient(90deg,#4ade80,#22d3ee)'},
    {name:'Commons（代理）', rows:dl.rows_commons, target:dl.target_commons,
     color:'linear-gradient(90deg,#fbbf24,#fb7185)'},
  ];
  if(dl.dual_source){
    $('srcbars').style.display='';
    $('srcbars').innerHTML=srcs.map(s=>{
    const running=s.commons&&window.__commonsRunning;
    const btn=(s.commons&&!running)
      ?`<button class="srcbtn" onclick="startCommons(this)">▶ 启动</button>`:'';
    const runTag=(s.commons&&running)?`<span class="muted">运行中</span>`:'';
    if(s.rows==null&&s.target==null)
      return `<div class="srcbar"><div class="lbl"><span>${s.name}</span>`
        +`<span style="display:flex;gap:8px;align-items:center"><span class="muted">未启动</span>${btn}</span></div>`
        +`<div class="track"><div class="fill" style="width:0%;${s.commons?'background:'+s.color:''}"></div></div></div>`;
    const pct=(s.rows!=null&&s.target)?Math.min(100,Math.round(s.rows/s.target*1000)/10):0;
    return `<div class="srcbar"><div class="lbl"><span>${s.name}</span>`
      +`<span style="display:flex;gap:8px;align-items:center"><span>${(s.rows??0).toLocaleString()} / ${(s.target??0).toLocaleString()}（${pct}%）</span>${btn}${runTag}</span></div>`
      +`<div class="track"><div class="fill" style="width:${pct}%;background:${s.color}"></div></div></div>`;
  }).join('');
  }else $('srcbars').style.display='none';
  $('dllog').innerHTML=(dl.tail||[]).map(l=>`<div>${l}</div>`).join('')||'<div class="muted">（无日志）</div>';
  // 速率明细行（学 _served.html 的 .rates-line）
  const net=dl.net||{};
  $('rates').innerHTML=`近5分钟 <b style="color:#4ade80">${rv==null?'–':rv.toFixed(1)}</b> 张/分`
    +` · 下行 <b style="color:#22d3ee">${net.dn==null?'–':net.dn.toFixed(0)}</b> KB/s`
    +` · 上行 <b style="color:#fbbf24">${net.up==null?'–':net.up.toFixed(0)}</b> KB/s`
    +(dl.idle_sec!=null?` ｜ 最后一张落盘 <b>${dl.idle_sec}</b> 秒前（ETA 依据近 5 分钟平滑速率）`:'');
  // 停滞警报（>180 秒无落盘且未结束）
  const al=$('dl-al');
  if(!dl.finished&&dl.idle_sec!=null&&dl.idle_sec>180){
    al.textContent='⚠ 已 '+Math.round(dl.idle_sec/60)+' 分钟没有新图片落盘——下载进程可能停了、卡在 GBIF API 阶段，或触发了限流。查看日志尾部确认。';
    al.classList.add('on');
  }else al.classList.remove('on');
  // 探测阶段：进度条 + 完成报告 + 数据量警告（横幅常驻，弹窗每次打开页面最多一次）
  const pb=dl.probe, pw=$('probe-warn');
  if(pb){
    $('probewrap').style.display='';
    const pp=pb.stage==='done'?100:(pb.total?Math.min(100,Math.round(pb.probed/pb.total*1000)/10):0);
    $('probebar').style.width=pp+'%';
    $('probelabel').textContent=pb.stage==='done'
      ?(pb.available!=null
        ?`探测完成：合计可用 ${(pb.available??0).toLocaleString()} 条 · 有数据 ${pb.with_data??'--'}/${pb.total} 种`
        :'探测已完成（本日志无合计行，详情见下载日志）')
      :`物种探测中（GBIF 可用量预检，此阶段无图片产出属正常） ${pb.probed}/${pb.total}（${pp}%）`;
    if(pb.warn){
      const crit=pb.warn==='critical';
      pw.textContent=crit
        ?'⚠ 探测可用量异常偏低——疑似 bbox/物种清单/网络配置问题，请核对下载日志 [预取] 行并联系用户（下载继续）'
        :'⚠ 探测可用量偏少（健康区 5 万+；华中 0.9 万为先例）——下载继续，训练按类平衡处理';
      pw.className='alert on '+(crit?'bad':'wa');
      const pk='ob_probe_pop_'+(window.__obRegion||'');
      if(sessionStorage.getItem(pk)!=='1'){
        sessionStorage.setItem(pk,'1');
        showProbeModal(pb);
      }
    }else pw.className='alert wa';
  }else{$('probewrap').style.display='none';pw.className='alert wa';}
  drawRates(dl);
}
function drawRates(dl){
  const c=$('ratecv'),x=c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  const now=window.__serverNow||Date.now()/1000;
  const W=c.width,H=c.height-18,pad=6;
  const series=[
    {pts:(dl.rate_series||[]).map(p=>[p.t,p.r]), color:'#4ade80', grid:true},
    {pts:(dl.net&&dl.net.series||[]).map(p=>[p.t,p.dn]), color:'#22d3ee'},
    {pts:(dl.net&&dl.net.series||[]).map(p=>[p.t,p.up]), color:'#fbbf24'},
  ];
  const any=series.find(s=>s.pts.length>=2);
  if(!any){
    x.fillStyle='#555';x.font='14px sans-serif';x.textAlign='center';
    x.fillText('速率采样中…（约 1 分钟后出曲线）',c.width/2,c.height/2);x.textAlign='left';return;
  }
  // 网格：按补量速率系列自动缩放（4 条）
  const g0=series[0];
  if(g0.pts.length>=2){
    let mn=Infinity,mx=-Infinity;
    for(const [,v] of g0.pts){mn=Math.min(mn,v);mx=Math.max(mx,v);}
    if(mx-mn<1)mx=mn+1;
    x.strokeStyle='#333';x.lineWidth=1;
    for(let g=0;g<=4;g++){const y=pad+(H*g/4);
      x.beginPath();x.moveTo(pad,y);x.lineTo(W-pad,y);x.stroke();
      x.fillStyle='#666';x.font='11px sans-serif';
      x.fillText((mx-(mx-mn)*g/4).toFixed(mx-mn<20?1:0),2,y-2);}
  }
  // 时间窗自适应：历史不满 60 分钟按实际跨度铺满全宽，随时间延展到 60min 封顶
  let span=3600;
  const allT=(dl.rate_series||[]).map(p=>p.t)
    .concat(((dl.net&&dl.net.series)||[]).map(p=>p.t));
  if(allT.length) span=Math.max(300,now-Math.min(...allT)+60);
  const X=t=>pad+(W-2*pad)*(1-(now-t)/span);  // 右侧=现在
  for(const s of series){
    if(s.pts.length<2)continue;
    let mn=Infinity,mx=-Infinity;
    for(const [,v] of s.pts){mn=Math.min(mn,v);mx=Math.max(mx,v);}
    if(mx-mn<1)mx=mn+1;
    x.strokeStyle=s.color;x.lineWidth=2;x.beginPath();
    let started=false;
    s.pts.forEach(p=>{
      if(p[0]<now-span)return;
      const px=X(p[0]),py=pad+H*(1-(p[1]-mn)/(mx-mn));
      started?x.lineTo(px,py):(x.moveTo(px,py),started=true);});
    x.stroke();
  }
  // 时间轴两端
  x.fillStyle='#666';x.font='11px sans-serif';
  x.fillText('-'+Math.round(span/60)+'min',pad,H+14);
  x.textAlign='right';x.fillText('现在',W-pad,H+14);x.textAlign='left';
}
function renderCmp(runs, samples){
  if(!runs.length)return;
  const metric=$('cmpMetric')?$('cmpMetric').value:'top1';
  const mlabel={top1:'top-1',top3:'top-3',top5:'top-5'}[metric]||metric;
  // 线的形式与主监控图表一致：top-1 绿实线，top-3 蓝虚线，top-5 青虚线
  const style={top1:{color:'#4ade80',dash:false},
               top3:{color:'#93c5fd',dash:true},
               top5:{color:'#22d3ee',dash:true}}[metric];
  // 曲线叠加：x=epoch(1..30)；最新 run 全色，旧 run 半透明
  const svg=$('cmpchart'),L=44,R=880,T=14,B=250,maxE=30;
  const maxY=Math.max(10,...runs.flatMap(r=>r.history.map(h=>h[metric])))*1.15;
  const X=e=>L+(e/maxE)*(R-L), Y=v=>B-(v/maxY)*(B-T-24);
  let g='';
  for(let i=0;i<=4;i++){const v=maxY/4*i,y=Y(v);
    g+=`<line x1="${L}" y1="${y}" x2="${R}" y2="${y}" stroke="#333"/>`
      +`<text x="6" y="${y+4}" fill="#666" font-size="11">${v.toFixed(0)}%</text>`;}
  for(let e=5;e<=maxE;e+=5)
    g+=`<line x1="${X(e)}" y1="${T}" x2="${X(e)}" y2="${B}" stroke="#2a2a2a"/>`
      +`<text x="${X(e)}" y="${B+16}" fill="#666" font-size="11" text-anchor="middle">${e}</text>`;
  const legend=runs.map((r,i)=>{
    const last=i===runs.length-1;
    const op=last?1:0.45, w=last?2.5:1.8;
    const dash=style.dash?` stroke-dasharray="6,4"`:'';
    const pts=r.history.map(h=>`${X(h.epoch)},${Y(h[metric])}`).join(' ');
    // 逐点悬停热区（透明大圆 + title 三项全量数值）
    const dots=r.history.map(h=>
      `<circle cx="${X(h.epoch)}" cy="${Y(h[metric])}" r="7" fill="transparent">`
      +`<title>${r.name} · ep${h.epoch} · ${mlabel} ${h[metric].toFixed(2)}%`
      +` ｜ top-1 ${h.top1.toFixed(2)}% · top-3 ${h.top3.toFixed(2)}% · top-5 ${h.top5.toFixed(2)}%</title></circle>`).join('');
    g+=`<polyline points="${pts}" fill="none" stroke="${style.color}" stroke-width="${w}"`
      +` opacity="${op}"${dash}/>`+dots;
    // best 端点（该 metric 最高点）：金点，与主图表 best 标记同形式，支持悬停
    const bp=r.history.reduce((a,b)=>b[metric]>a[metric]?b:a,r.history[0]);
    g+=`<circle cx="${X(bp.epoch)}" cy="${Y(bp[metric])}" r="5" fill="#fbbf24" opacity="${op}">`
      +`<title>${r.name} · best ${mlabel} ${bp[metric].toFixed(2)}%（ep${bp.epoch}）</title></circle>`;
    return `<text x="${L+8}" y="${T+14+i*16}" fill="${style.color}" font-size="11" opacity="${op}">`
      +`— ${r.name}（best ${r.best}%${last?'':', 对比参照'}）</text>`;
  }).join('');
  // 之前保存的破纪录快照（best.pt 采样点）：作为端点标在最新 run 曲线上
  const cur=runs[runs.length-1];
  const snapPts=(samples||[]).filter(p=>p.epoch<=maxE);
  const snapDots=snapPts.map(p=>
    `<circle cx="${X(p.epoch)}" cy="${Y(p[metric])}" r="3.5" fill="#fbbf24">`
    +`<title>${cur.name} · 破纪录快照 ep${p.epoch} · ${mlabel} ${p[metric].toFixed(2)}%</title></circle>`).join('');
  svg.innerHTML=g+snapDots+legend;
  // run 概要卡：数字随所选指标
  $('cmpcards').innerHTML=runs.map((r,i)=>{
    const bmax=Math.max(...r.history.map(h=>h[metric]));
    const last=i===runs.length-1;
    return `<div class="card"><div class="v" style="color:${style.color};opacity:${last?1:0.5};font-size:24px">${bmax.toFixed(2)}%</div>
      <div class="kpi-lbl">${r.name} · best ${mlabel}<br>${r.n_images!=null?r.n_images.toLocaleString()+' 张 / ':''}${r.n_species??'?'} 类
      · ${r.done?'已完成':'训练中 '+r.history.length+'/30'}</div></div>`;
  }).join('');
  // 逐 epoch 对比表：本轮（最后一个 run）vs 上一轮完成 run，Δ 列跟随所选指标
  if(runs.length<2){$('cmptable').innerHTML='<div class="muted">只有一轮训练数据，暂无对比对象</div>';return;}
  const prev=[...runs].reverse().find(r=>r.done&&r!==cur);
  const pm=new Map(prev.history.map(h=>[h.epoch,h]));
  let rows='',maxD=-999,maxDpE=null;
  const others=['top1','top3','top5'].filter(m=>m!==metric);
  const olabel={top1:'top-1',top3:'top-3',top5:'top-5'};
  // 表头颜色标识 = 图表曲线颜色（图表形式 = 记录形式）
  const dot=c=>`<span style="color:${c}">■</span> `;
  cur.history.forEach(h=>{
    const p=pm.get(h.epoch);
    const d=p?(h[metric]-p[metric]):null;
    if(d!=null&&d>maxD){maxD=d;maxDpE=h.epoch;}
    rows+=`<tr><td>${h.epoch}</td><td class="r">${h[metric].toFixed(2)}%</td>`
      +(p?`<td class="r">${p[metric].toFixed(2)}%</td><td class="r" style="color:${d>=0?'#4ade80':'#f87171'}">${d>=0?'+':''}${d.toFixed(2)}</td>`
         :'<td class="r">–</td><td class="r">–</td>')
      +`<td class="r">${h[others[0]].toFixed(2)}%</td><td class="r">${h[others[1]].toFixed(2)}%</td></tr>`;
  });
  $('cmptable').innerHTML=`<table><thead><tr><th>epoch</th>
    <th class="r">${dot(style.color)}${cur.name}（${mlabel}）</th>
    <th class="r">${dot(style.color)}${prev.name}（${mlabel}）</th><th class="r">Δ ${mlabel}</th>
    <th class="r">${olabel[others[0]]}</th><th class="r">${olabel[others[1]]}</th></tr></thead>
    <tbody>${rows}</tbody></table>`
    +(maxDpE?`<div class="muted" style="margin-top:6px">「${mlabel}」提升最大的点：epoch ${maxDpE}（+${maxD.toFixed(2)}）</div>`:'');
}
async function tick(){
  try{
    const region=localStorage.getItem('ob_region')||'';   // 空 = 服务端默认（REGIONS 首个）
    const s=await (await fetch('/api/status?region='+region)).json();
    if(s.regions){
      const rs=document.getElementById('regionSel');
      if(rs.options.length!==s.regions.length){
        rs.innerHTML=s.regions.map(x=>`<option value="${x}">${x}</option>`).join('');
      }
      if(rs.value!==s.region) rs.value=s.region;
      rs.onchange=()=>{
        localStorage.setItem('ob_region',rs.value);
        // 立即清空三页动态内容，避免显示上一地区残留
        ['chart','ratecv'].forEach(id=>{const el=document.getElementById(id);
          if(el&&el.tagName==='CANVAS')el.getContext('2d').clearRect(0,0,el.width,el.height);
          else if(el)el.innerHTML='';});
        ['cmpchart'].forEach(id=>{const el=document.getElementById(id);if(el)el.innerHTML='';});
        document.getElementById('cmptable').innerHTML='<div class=\"muted\">加载中…</div>';
        document.getElementById('cmpcards').innerHTML='';
        tick();
      };
      if(s.region_label) rs.options[rs.selectedIndex].text=s.region_label;
    }
    const last=s.samples.at(-1);
    if(last){ $('top1').textContent=last.top1.toFixed(2)+'%';
      $('top3').textContent=last.top3.toFixed(2)+'%';
      $('top5').textContent=last.top5.toFixed(2)+'%';
      $('epoch').textContent=last.epoch; }
    $('gpu').textContent=s.gpu;
    // 进度：优先 train.log 的真实 epoch（带小数，精确到 0.1%），无日志退回节奏估计
    const real=s.real_epoch, ef=s.epoch_float, est=s.est_epoch||0;
    const shown = ef!=null ? ef : (real||est);
    const pct = s.epoch_pct!=null ? s.epoch_pct : Math.round(shown/30*1000)/10;
    $('bar').firstElementChild.style.width=pct+'%';
    let eta='';
    if(!s.done && !s.overrun && s.pace_sec && ef==null){
      eta=` · 剩余约 ${Math.max(0,Math.round((s.total_epochs-est)*s.pace_sec/60))} 分钟`;
    }
    $('barlabel').textContent=(ef!=null?`真实进度 ${ef.toFixed(2)}/${s.total_epochs}`
      :(real?`真实进度 ${shown}/${s.total_epochs}`:`估计进度 ${est}/${s.total_epochs}`))
      +` · ${pct.toFixed(1)}%`+eta
      +(s.done?' · 已完成':(ef!=null?'':(real?'':(s.overrun?' · 收尾中（后半程更慢，估计已封顶）':' · 估计值（本次训练无 train.log）'))));
    $('curep').textContent= ef!=null ? ef.toFixed(1)
      : (real?real:(est+(s.overrun?'+':'')));
    // Epoch 分段进度条：30 格，1 epoch = 1 格；结束时自动切换为真实 history
    const shownCells=Math.min(shown,s.total_epochs);
    const eb=$('epbar');
    if(eb.childElementCount!==s.total_epochs){
      eb.innerHTML=Array.from({length:s.total_epochs},()=>'<i></i>').join('');
    }
    const doneCells=Math.floor(shownCells);
    [...eb.children].forEach((c,i)=>{
      c.className = i < doneCells ? 'f' : (i === doneCells && !s.done ? 'cur' : '');
    });
    $('eplabel').textContent=`Epoch 进度：${shownCells}${ef!=null?'（含轮内小数）':''}/${s.total_epochs}`
      +(ef!=null?'（真实）':(real?'（真实）':'（估计）'))
      +(s.done?' · 完整曲线已生成':'');
    // 顶部状态横幅：三种状态一眼可辨
    const st=$('state');
    const stTxt=$('stateTxt');
    const stBtn=$('stopTrain');
    if(s.done){
      st.className='st-done';
      stTxt.textContent='已完成 ✓ best top-1 '+s.final.best_top1+'%（共 '+s.final.n_images+' 张 / '+s.final.n_species+' 类）';
      stBtn.disabled=true; stBtn.textContent='✓ 已结束';
    }else if(s.overrun){
      st.className='st-overrun';
      stTxt.textContent='⏳ 训练仍在进行——进度估计已到 100% 只是估计封顶（后半程 epoch 更慢），不是已完成';
    }else{
      st.className='st-training';
      stTxt.textContent='训练中，请稍候…';
    }
    $('sub').textContent=`已运行 ${s.elapsed_min} 分钟${s.done?' · 已完成':''}`;
    if(s.done) $('sub').innerHTML+=` <span class="done">→ best top-1 ${s.final.best_top1}%</span>`;
    // 折线图：完成后用完整 30 轮 history，训练中用破纪录采样点
    const svg=$('chart');
    const pts=(s.history&&s.history.length)
      ? s.history.map(h=>({epoch:h.epoch,top1:h.top1*100,top3:h.top3*100,top5:h.top5*100}))
      : s.samples;
    const L=40,R=890,T=10,B=240,maxX=30;
    const maxY=Math.max(10,...pts.map(p=>p.top1),...pts.map(p=>p.top5||0))*1.15;
    const X=e=>L+(e/maxX)*(R-L), Y=v=>B-(v/maxY)*(B-T-24);
    let g='';
    for(let i=0;i<=4;i++){const v=maxY/4*i,y=Y(v);
      g+=`<line x1="${L}" y1="${y}" x2="${R}" y2="${y}" stroke="#333"/>`
        +`<text x="6" y="${y+4}" fill="#666" font-size="11">${v.toFixed(0)}%</text>`;}
    for(let e=5;e<=maxX;e+=5){
      g+=`<line x1="${X(e)}" y1="${T}" x2="${X(e)}" y2="${B}" stroke="#2a2a2a"/>`
        +`<text x="${X(e)}" y="${B+16}" fill="#666" font-size="11" text-anchor="middle">${e}</text>`;}
    if(!pts.length){
      svg.innerHTML=g+`<text x="465" y="130" fill="#555" font-size="14" text-anchor="middle">等待第一个数据点…</text>`;
    }else{
      const line=(key,color,w,dash)=>pts.filter(x=>x[key]!=null).length<2?'':
        `<polyline points="${pts.map(x=>X(x.epoch)+','+Y(x[key])).join(' ')}" fill="none" stroke="${color}" stroke-width="${w}"${dash?' stroke-dasharray="5,4"':''} opacity="0.85"/>`;
      const best=pts.reduce((a,b)=>b.top1>a.top1?b:a,pts[0]);
      const last=pts[pts.length-1];
      const dots=pts.map(p=>`<circle cx="${X(p.epoch)}" cy="${Y(p.top1)}" r="${p===best?5:3.5}" fill="${p===best?'#fbbf24':'#4ade80'}"/>`).join('');
      svg.innerHTML=g
        +line('top5','#22d3ee',1.5,true)
        +line('top3','#93c5fd',1.5,true)
        +line('top1','#4ade80',2.5,false)
        +dots
        +`<text x="${X(best.epoch)}" y="${Y(best.top1)-10}" fill="#fbbf24" font-size="12" text-anchor="middle">best ${best.top1.toFixed(2)}%</text>`
        +(pts.length>1?`<text x="${X(last.epoch)}" y="${Y(last.top1)+18}" fill="#9ca3af" font-size="11" text-anchor="middle">ep${last.epoch}</text>`:'')
        +`<text x="${L+4}" y="${T+12}" fill="#4ade80" font-size="11">top-1</text>`
        +`<text x="${L+52}" y="${T+12}" fill="#22d3ee" font-size="11">top-5（虚线）</text>`
        +`<text x="${L+130}" y="${T+12}" fill="#93c5fd" font-size="11">top-3（虚线）</text>`;
    }
    $('log').innerHTML=s.events.map(e=>`<div><b>[${e.ts}]</b> ${e.msg}</div>`).join('');
    $('tlog').innerHTML=(s.train_log&&s.train_log.length)
      ? s.train_log.map(l=>`<div>${l}</div>`).join('')
      : '<div class="muted">本次训练启动于 train.log 修复之前，无实时日志；结束后 history.json 会补全完整曲线。</div>';
    window.__serverNow=s.now_ts||Date.now()/1000;
    window.__obRegion=s.region||'';
    window.__runs=s.runs||[];
    window.__commonsRunning=!!s.commons_running;
    window.__curRegion=s.region;
    renderDl(s.download);
    renderCmp(window.__runs, s.samples);
  }catch(e){ $('sub').textContent='(连接中断，重试中)'; }
}
tick(); setInterval(tick,3000);
activate(location.hash.slice(1)==='dl'?'dl':'train');
</script></body></html>"""


def proc_matches(target: str, cmdline: list) -> bool:
    """目标进程判定：按命令行 token 的**文件名**精确相等，绝不用子串。

    "train.py" 作为子串会误伤 monitor_train.py 等文件名（实测在监控台上点
    「停止」把无关进程也杀了）；basename 精确相等才只命中真正的进程。
    target=download 同时覆盖 GBIF 构建与 Commons 抓取。
    """
    for t in cmdline or []:
        base = os.path.basename(str(t).replace("\\", "/")).lower()
        if target == "train" and base == "train.py":
            return True
        if target == "download" and base in ("build_dataset_gbif.py",
                                             "fetch_commons_region.py"):
            return True
    return False


def commons_running() -> bool:
    """是否有 fetch_commons_region.py 进程在跑（供启动按钮禁用态）。"""
    try:
        import psutil
    except Exception:
        return False
    for p in psutil.process_iter(["name", "cmdline"]):
        if (p.info.get("name") or "").lower() != "python.exe":
            continue
        toks = [os.path.basename(str(t).replace("\\", "/")).lower()
                for t in (p.info.get("cmdline") or [])]
        if "fetch_commons_region.py" in toks:
            return True
    return False


def start_commons(region: str) -> dict:
    """手动启动 Commons 补充抓取（脱离进程，日志落在该区数据目录）。"""
    cfg = REGIONS.get(region) or {}
    sp = cfg.get("species_csv")
    if not sp or not os.path.exists(sp):
        return {"started": False, "msg": "该区还没有物种清单（species_csv 未配置）"}
    if commons_running():
        return {"started": False, "msg": "Commons 抓取已在运行"}
    data_dir = cfg.get("data") or os.path.dirname(sp)
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir,
                            f"commons_fetch.log.{time.strftime('%Y%m%d-%H%M%S')}")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fetch_commons_region.py")
    out = open(log_path, "a", encoding="utf-8")
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, "-u", script,
                      "--species", sp, "--out", data_dir, "--nice"],
                     stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
    event(f"[{region}] Commons 抓取已手动启动 -> {os.path.basename(log_path)}")
    return {"started": True, "log": log_path}


def stop_processes(target: str) -> list:
    """按命令行精确匹配并杀掉训练/下载进程树（含 dataloader 子进程）。"""
    import psutil
    killed = []
    me = os.getpid()
    region_paths = []
    for cfg in REGIONS.values():
        region_paths.append(cfg["model"].replace("\\", "/").lower())
        region_paths.append(cfg["data"].replace("\\", "/").lower())
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            if not proc_matches(target, p.info["cmdline"]):
                continue
            cl = " ".join(p.info["cmdline"] or []).replace("\\", "/").lower()
            if target == "train" and not any(pp in cl for pp in region_paths):
                continue  # 只杀注册表内地区的训练，别项目/别目录不碰
            proc = psutil.Process(p.info["pid"])
            tree = [proc] + proc.children(recursive=True)
            for k in tree:
                try:
                    k.kill()
                except Exception:
                    pass
            psutil.wait_procs(tree, timeout=5)
            killed.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    msg = (f"已停止 {target}: PID {killed}" if killed
           else f"停止 {target}: 没有在运行的进程")
    event(msg)
    return killed


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        if path == "/api/status":
            default = next(iter(REGIONS), None)
            region = (qs.get("region") or [None])[0] or default
            region = region if region in REGIONS else default
            self._send(json.loads(status_json(region)))
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        from urllib.parse import parse_qs, urlparse
        path = urlparse(self.path).path
        if path == "/api/start":
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                region = body.get("region") or next(iter(REGIONS), "")
                if region not in REGIONS:
                    self._send({"error": "unknown region"}, 400)
                    return
                self._send(start_commons(region))
            except Exception as e:
                self._send({"error": str(e)}, 500)
            return
        if path != "/api/stop":
            self._send({"error": "unknown endpoint"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            target = body.get("target")
            if target not in ("train", "download"):
                self._send({"error": "target must be train|download"}, 400)
                return
            self._send({"target": target, "killed": stop_processes(target)})
        except Exception as e:
            self._send({"error": str(e)}, 500)

    def log_message(self, *a):
        pass  # 静默访问日志，事件日志才是主角


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="模型目录。与 REGIONS 注册区匹配时覆盖该区配置，否则注册为 local 区"
                         "（移动包场景：--out models/<region>_mnv4 --download-dir data "
                         "--download-dataset data/<region>/dataset.csv）")
    ap.add_argument("--region", default=None,
                    help="单地区模式（移动包 start_dashboard.bat 传入）：整表替换 REGIONS，"
                         "只监控该地区——注册表里的主开发机绝对路径在包机上不存在")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--train-start", dest="train_start", default=None)
    ap.add_argument("--download-dir", dest="download_dir", default=None)
    ap.add_argument("--download-dataset", dest="download_dataset", default=None)
    ap.add_argument("--download-target", dest="download_target", type=int, default=None)
    ap.add_argument("--download-baseline", dest="download_baseline", type=int, default=None)
    a = ap.parse_args()

    if a.region:
        # 包模式：REGIONS 默认表全是主开发机的绝对路径，在包机上没有意义，整表换掉
        cfg = {"label": REGION_LABELS.get(a.region, a.region),
               "target": None, "baseline": 0,
               "model": a.out or os.path.join("models", f"{a.region}_mnv4")}
        if a.download_dataset:
            cfg["data"] = os.path.dirname(a.download_dataset)
        cfg["logdir"] = (a.download_dir or cfg.get("data")
                         or os.path.dirname(cfg["model"]))
        if a.download_target is not None:
            cfg["target"] = a.download_target
        if a.download_baseline is not None:
            cfg["baseline"] = a.download_baseline
        with LOCK:
            REGIONS.clear()
            REGIONS[a.region] = cfg
    elif a.out:
        out_norm = os.path.normpath(a.out)
        with LOCK:
            hit = next((k for k, v in REGIONS.items()
                        if os.path.normpath(v["model"]) == out_norm), None)
            key = hit or "local"
            cfg = dict(REGIONS.get(key) or {"label": "本地", "target": None,
                                            "baseline": 0})
            cfg["model"] = a.out
            if a.download_dataset:
                cfg["data"] = os.path.dirname(a.download_dataset)
            cfg["logdir"] = (a.download_dir or cfg.get("logdir")
                             or cfg.get("data") or os.path.dirname(a.out))
            if a.download_target is not None:
                cfg["target"] = a.download_target
            if a.download_baseline is not None:
                cfg["baseline"] = a.download_baseline
            if hit:
                REGIONS[key] = cfg  # 就地覆盖：保持原插入顺序，默认区不变
            else:
                rest = dict(REGIONS)  # 新键排到最前（dict 无 move_to_end，重建顺序）
                REGIONS.clear()
                REGIONS[key] = cfg
                REGIONS.update(rest)

    if a.train_start:
        t = datetime.datetime.strptime(a.train_start, "%H:%M")
        today = datetime.datetime.now().replace(
            hour=t.hour, minute=t.minute, second=0, microsecond=0)
        with LOCK:
            for rs in STATE["regions"].values():
                rs["train_start"] = today.timestamp()
    threading.Thread(target=poller, args=(3.0,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[{now()}] dashboard: http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

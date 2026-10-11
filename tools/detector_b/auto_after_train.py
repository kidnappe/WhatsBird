# -*- coding: utf-8 -*-
r"""训练结束后自动跑：把累积的所有 epoch_*_ckpt.pth 逐个导出并 bench，挑 COCO 上最优。

为什么做这个（而不是只 bench best_ckpt）：
  best_ckpt 是 trainer 按 **det_val（伪标签子集）AP** 挑的；
  而验收门槛是 **COCO val2017 人工标注**口径。两个尺子可能不一致——
  ep30 可能在伪标签上破纪录被存成 best，但在 COCO 上不如 ep25。
  只 bench best 一个会漏掉这种情形。末段 ep40~50 每轮都评，那 10 个是最可能出
  最优解的区间，更不能只看 best。

流程（每个 epoch 模型）：
  ckpt -> export_b/类似的 onnx 导出 -> bench_detector_b -> 记一行汇总
  统一汇总成一张表（含验收线判定），并写 best_by_coco 指针。

跑法：
  python auto_after_train.py --wait      # 等训练结束再跑
  python auto_after_train.py             # 直接跑（不等）

跑完会写：data/detector_b/ckpt_sweep/{summary.json, summary.csv}
"""
import argparse
import json
import os
import subprocess
import sys
import time

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

REPO = r"E:\code\code tools\yolox_repo"
DETB = r"E:\code\code tools\data\detector_b"
CKPT_DIR = os.path.join(REPO, "YOLOX_outputs", "yolox_bird_512")
OUT_DIR = os.path.join(DETB, "ckpt_sweep")
ONNX_DIR = os.path.join(DETB, "ckpt_sweep", "onnx")
RESULTS = os.path.join(DETB, "results")
LOG = os.path.join(DETB, "ckpt_sweep.log")
EXPORT = r"E:\code\observebird\tools\detector_b\export_b_onnx.py"
BENCH = r"E:\code\observebird\tools\detector_b\bench_detector_b.py"

# 验收线（docs/handoff/detector_b/HANDOFF.md 阶段3 / docs/handoff/detector_b/README-B-LOCAL.md §4，PC 口径，COCO 人工标注）
THRESH = {"tiny": 0.15, "small": 0.70, "recall": 0.45, "fp_per_img": 0.30}
BASELINE = {"recall": 0.368, "tiny": 0.072, "small": 0.551, "fp_per_img": 0.33}

# ---- 四区分类器 TFLite 导出（docs/handoff/root/HANDOFF-APP-MODELS-20261006.md §4-A）----
# ⚠️ --species-list **必须**指各区自己的 dataset.csv：脚本默认值是旧华北对照表，
#   忘指就产出"中文名为空"的 labels（labels_huanan_sw.txt 的历史欠账就是这么来的）。
EXPORT_TFLITE = r"E:\code\observebird\tools\export_tflite.py"
MODELS = r"E:\code\code tools\models"
DATA = r"E:\code\code tools\data"
# (区名, 模型目录, dataset.csv)
REGION_EXPORTS = [
    ("huabei",    "huabei_mnv4_rec",     "huabei"),
    ("huanan_sw", "huanan_sw_mnv4_rec",  "huanan_sw"),
    ("huadong",   "huadong_mnv4_300_rec", "huadong"),
    ("dongbei",   "dongbei_mnv4_rec",    "dongbei"),
    # huazhong_mnv4_25k 已在 App（assets 里 tflite 与 export 产物同尺寸），跳过
]


def log(m):
    line = "%s %s" % (time.strftime("%H:%M:%S"), m)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# 训练日志（stderr）。判"结束"靠它最稳：训练进程退出后再不会有新行，
# 且末行会出现 trainer 的 "Training of experiment is done" —— 不依赖任何进程枚举
# （Win11 已移除 wmic，嵌套 PowerShell 的转义又会弄坏 WMI 的 -Filter）。
TRAIN_ERR = os.path.join(DETB, "train_b512.log.err.20261006-211949")
DONE_RE = "Training of experiment is done"


def training_alive():
    """训练是否还在跑：日志近 5 分钟有新行 = 在跑；出现 done 行 = 结束。"""
    try:
        age = time.time() - os.path.getmtime(TRAIN_ERR)
        if age < 300:                      # 5 分钟内有更新 → 还在训
            return True
        # 停了：看是不是正常收尾（done 行 = 训完了；否则 = 崩了）
        with open(TRAIN_ERR, encoding="utf-8", errors="replace") as f:
            f.seek(max(0, os.path.getsize(TRAIN_ERR) - 8192))
            tail = f.read()
        return DONE_RE not in tail
    except Exception:
        return True      # 探不到就当还活着，保守等


def wait_done(timeout_s=6 * 3600):
    log("等待训练结束（最多等 %.1f 小时）…" % (timeout_s / 3600))
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        alive = training_alive()
        if not alive:
            log("训练进程已退出")
            return True
        # 报进度
        try:
            p = os.path.join(DETB, "train_b512.log.err.20261006-211949")
            txt = open(p, encoding="utf-8", errors="replace").read()[-4000:]
            import re
            ms = re.findall(r"epoch: (\d+)/50", txt)
            cur = ms[-1] if ms else "?"
            if cur != last:
                log("训练中 ep %s/50" % cur)
                last = cur
        except Exception:
            pass
        time.sleep(60)
    log("等待超时，仍继续")
    return False


def epoch_ckpts():
    """所有 epoch_*_ckpt.pth + best_ckpt，按 epoch 号排序。"""
    out = []
    if not os.path.isdir(CKPT_DIR):
        return out
    for f in os.listdir(CKPT_DIR):
        if f.startswith("epoch_") and f.endswith("_ckpt.pth"):
            try:
                n = int(f.split("_")[1])
                out.append((n, os.path.join(CKPT_DIR, f)))
            except ValueError:
                pass
    out.sort()
    return out


def export_one(ckpt, tag):
    """导出 dec onnx（B 单类 512），返回 onnx 路径。"""
    os.makedirs(ONNX_DIR, exist_ok=True)
    prefix = os.path.join(ONNX_DIR, "yolox_bird_512_%s" % tag)
    if os.path.exists(prefix + "_dec.onnx"):
        return prefix + "_dec.onnx"
    cmd = [sys.executable, EXPORT, ckpt, "512", prefix]
    log("  导出 onnx: %s" % tag)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = prefix + "_dec.onnx"
    if r.returncode != 0 or not os.path.exists(out):
        tail = ((r.stderr or "") + (r.stdout or ""))[-800:]
        log("  导出失败 %s: %s" % (tag, tail.splitlines()[-1] if tail else "无输出"))
        return None
    return out


def bench_one(onnx, tag):
    """跑 bench_detector_b，读回 results/<tag>.json。"""
    cmd = [sys.executable, BENCH, onnx, tag]
    log("  bench: %s" % tag)
    subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    p = os.path.join(RESULTS, tag + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def row_of(tag):
    p = os.path.join(RESULTS, tag + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    c = d["coco"]["th0.35"]
    b = c["by_size"]
    return {
        "tag": tag,
        "recall": c["recall"], "tiny": b["tiny<0.25%"]["recall"],
        "small": b["small<1%"]["recall"], "large": b["large>=1%"]["recall"],
        "fp_per_img": c["fp_per_img"],
        "huabei": (d.get("huabei") or {}).get("th0.35", {}).get("det_rate"),
        "ms": d["coco"]["ms_per_img"],
    }


def export_region_tflite():
    """四区分类器 -> float32 TFLite + labels（等训练结束释放 CPU/内存后才做）。

    只写 `models/<name>/export/`（不进 assets、不改代码）——拷贝到 App 是另一道人工闸门。
    华中已导出，跳过；导出失败不阻塞后面的 sweep。
    """
    log("=" * 60)
    log("四区分类器 TFLite 导出（onnx2tf 是 CPU 多核重活，须等训练结束）")
    log("=" * 60)
    done = []
    for region, mdir, ddir in REGION_EXPORTS:
        ckpt = os.path.join(MODELS, mdir, "best.pt")
        outd = os.path.join(MODELS, mdir, "export")
        splist = os.path.join(DATA, ddir, "dataset.csv")
        tfl = os.path.join(outd, "tflite", "model_float32.tflite")
        if not os.path.exists(ckpt):
            log("  [%s] 缺 best.pt，跳过" % region)
            continue
        if os.path.exists(tfl):
            log("  [%s] 已导出，跳过" % region)
            done.append(region)
            continue
        log("  [%s] 导出中（%s）…" % (region, mdir))
        cmd = [sys.executable, EXPORT_TFLITE,
               "--ckpt", ckpt, "--out-dir", outd, "--species-list", splist]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if os.path.exists(tfl):
                sz = os.path.getsize(tfl) / 1e6
                log("  [%s] 完成 %.1f MB" % (region, sz))
                done.append(region)
            else:
                tail = ((r.stderr or "") + (r.stdout or ""))[-600:]
                log("  [%s] 未产出 tflite: %s"
                    % (region, tail.splitlines()[-1] if tail else "无输出"))
        except Exception as e:
            log("  [%s] 导出异常: %s: %s" % (region, type(e).__name__, str(e)[:120]))
    log("导出完成区: %s" % (", ".join(done) if done else "无"))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true", help="先等训练结束")
    ap.add_argument("--timeout-h", type=float, default=6.0)
    ap.add_argument("--only", default="", help="只跑指定 epoch，逗号分隔（调试用）")
    ap.add_argument("--tasks", default="sweep,export",
                    help="跑哪些：sweep(检测器B ckpt 扫 bench) / export(四区 TFLite)")
    a = ap.parse_args()

    tasks = {t.strip() for t in a.tasks.split(",") if t.strip()}

    if a.wait:
        wait_done(a.timeout_h * 3600)

    # 顺序：先 sweep（快、结论导向），后 export（慢、纯搬运）。
    # sweep 先出结果，万一磁盘/时间不够，至少拿到了"哪个 epoch 最优"这个结论。
    if "sweep" in tasks:
        run_sweep(a)
    if "export" in tasks:
        export_region_tflite()
    log("全部任务结束")


def run_sweep(a):
    ckpts = epoch_ckpts()
    if a.only:
        want = {int(x) for x in a.only.split(",") if x.strip().isdigit()}
        ckpts = [(n, p) for n, p in ckpts if n in want]
    # best_ckpt 也测（对照"训练器自己选的是不是最好的"）
    best = os.path.join(CKPT_DIR, "best_ckpt.pth")
    if os.path.exists(best) and not a.only:
        ckpts = ckpts + [("best", best)]

    log("=" * 60)
    log("ep50 后自动 sweep：共 %d 个模型" % len(ckpts))
    log("=" * 60)

    rows = []
    for n, ckpt in ckpts:
        tag = ("ep%d" % n) if isinstance(n, int) else "best"
        try:
            onnx = export_one(ckpt, tag)
            if not onnx:
                continue
            bench_one(onnx, tag)
            r = row_of(tag)
            if r:
                r["epoch"] = n if isinstance(n, int) else "best"
                rows.append(r)
                log("  %-6s recall=%.3f tiny=%.3f small=%.3f FP=%.2f"
                    % (tag, r["recall"], r["tiny"], r["small"], r["fp_per_img"]))
        except Exception as e:
            log("  %s 异常: %s: %s" % (tag, type(e).__name__, str(e)[:120]))

    # 汇总
    os.makedirs(OUT_DIR, exist_ok=True)
    best_by_tiny = max(rows, key=lambda r: r["tiny"]) if rows else None
    best_by_all = max(rows, key=lambda r: (r["tiny"] >= THRESH["tiny"]) and r["recall"]) if rows else None
    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": THRESH, "baseline_s512": BASELINE,
        "rows": rows,
        "best_by_tiny": best_by_tiny["tag"] if best_by_tiny else None,
        "best_passing": best_by_all["tag"] if best_by_all else None,
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    if rows:
        import csv
        keys = ["tag", "epoch", "recall", "tiny", "small", "large",
                "fp_per_img", "huabei", "ms"]
        with open(os.path.join(OUT_DIR, "summary.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        log("=" * 60)
        log("汇总（门槛 tiny≥0.15 small≥0.70 recall≥0.45 FP≤0.30）")
        log("  模型     recall   tiny   small     FP/图   tiny达标")
        for r in rows:
            log("  %-8s %.3f  %.3f  %.3f   %.2f     %s"
                % (r["tag"], r["recall"], r["tiny"], r["small"], r["fp_per_img"],
                   "✓" if r["tiny"] >= THRESH["tiny"] else "✗"))
        log("")
        log("tiny 最高: %s" % (best_by_tiny["tag"] if best_by_tiny else "-"))
        log("综合最优(过tiny前提下 recall 最高): %s" % (best_by_all["tag"] if best_by_all else "-"))
        log("结果: %s" % os.path.join(OUT_DIR, "summary.json"))
    log("sweep 完成")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""无人值守串行链（2026-10-06 凌晨）：华东训完 → 华南恢复+重训 → 华北恢复+重训。

**分工**

  华东（huadong）: 已有 `watch_and_train.py` 守候进程在跑（下载完自动 audit→clean→train，
                   完成写 models\\huadong_mnv4_300\\PIPELINE_DONE）。本脚本只**等它**。
  华南（huanan_sw）: `recover_region.py` 三步恢复（补登记 1,561 张 + 对照表修 5 个种
                   → 重审计 → 重清洗），然后 train.py 重训。
  华北（huabei）:   同上（金翅雀 651 张可下、黄苇鳽 3 张、白琵鹭 18 张补登记）。

**原则**

  * 一次只跑一件事（15GB 内存铁律）—— 全程串行。
  * 每区独立：一区失败**跳过该区**继续下一区，不拖垮整条链。
  * 不训练半成品：训练完成以 train.log 出现「完成 best top1=」为准（项目约定），
    确认后才写该模型目录的 PIPELINE_DONE。
  * 模型进**新目录**（huanan_sw_mnv4_rec / huabei_mnv4_rec），**不覆盖旧模型**
    （旧模型是对比锚点）。监控台 REGIONS 指向旧目录，早上要手动重指。
  * recover_region 自带 busy 检查（有别的链在跑就 exit 2）—— 本链收到 2 会等 5 分钟
    重试，最多 3 次。

用法（必须经 run_detached.ps1 脱离启动，见 HANDOFF §0.3）:
  .\\tools\\run_detached.ps1 -Exe 'D:\\software\\Python312\\python.exe' `
    -ScriptArgs @('-u','E:\\code\\observebird\\tools\\overnight_chain.py') `
    -LogPath 'E:\\code\\code tools\\data\\overnight_chain.log'
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
STALL_SEC = 5 * 3600          # progress.json 5 小时不动 + 下载进程还活着 → 判停滞
STATUS_PATH = os.path.join(CODE_TOOLS, "models", "overnight_chain_status.json")

BUSY_PATS = ("build_dataset_gbif", "train.py", "auto_region_pipeline",
             "pipeline_region", "fetch_inat_region")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [chain] {msg}", flush=True)


def env():
    return {**os.environ, "PYTHONIOENCODING": "utf-8",
            "HF_ENDPOINT": "https://hf-mirror.com"}


def ps_cmdlines():
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=90).stdout or ""
    except Exception:
        return ""                      # 查不到就当没有（recover_region 那边有自己的保守判断）


def others_busy():
    """除了本链自身之外的下载/训练进程。"""
    hits = []
    for line in ps_cmdlines().splitlines():
        if "overnight_chain" in line:
            continue
        for pat in BUSY_PATS:
            if pat in line:
                hits.append(line.strip()[:110])
                break
    return hits


def watch_alive():
    return "watch_and_train" in ps_cmdlines()


def save_status(results):
    try:
        json.dump(results, open(STATUS_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    except Exception as e:
        log(f"⚠️ 状态文件写不进去: {e}")


def wait_huadong(results):
    """等华东守候链交付。返回 'done' 或 'broken(...)' —— 断了也不挡下一区。"""
    done = os.path.join(CODE_TOOLS, "models", "huadong_mnv4_300", "PIPELINE_DONE")
    pj = os.path.join(CODE_TOOLS, "data", "huadong", "progress.json")
    log(f"等待华东：轮询 {done}")
    last_ts, last_change = None, time.time()
    while True:
        if os.path.exists(done):
            log("华东 PIPELINE_DONE ✓ —— 等守候进程退出（recover 的 busy 检查连 "
                "watch_and_train 也会算 busy，必须等它走干净）")
            while watch_alive():
                time.sleep(POLL)
            log("守候进程已退出")
            return "done"
        st = ts = None
        try:
            j = json.load(open(pj, encoding="utf-8"))
            st, ts = j.get("stage"), j.get("ts")
        except Exception:
            pass
        if ts != last_ts:
            last_ts, last_change = ts, time.time()
        lines = ps_cmdlines()
        watch = "watch_and_train" in lines
        dl = any("build_dataset_gbif" in l and "huadong" in l for l in lines.splitlines())
        if not watch and not dl:
            log(f"⚠️ 华东守候进程和下载进程都不在、又没有 PIPELINE_DONE"
                f"（progress stage={st}）—— 华东链已断，不无限等，转下一区")
            return f"broken(stage={st})"
        if time.time() - last_change > STALL_SEC and st == "download":
            log(f"⚠️ progress.json 已 {STALL_SEC//3600} 小时没动而下载进程还活着"
                f"—— 判停滞，不再等华东（注意：下载还占着，下一区的 recover 会重试后放弃）")
            return f"stalled(stage={st})"
        time.sleep(POLL)


def run_step(name, cmd, retries=12):
    """有别的任务在跑就等 5 分钟重试（12 次 = 1 小时耐心：华南下载并行插入后，
    华东若提前结束， recover 要能等到下载收尾）；别的错误码原样返回。"""
    for attempt in range(1, retries + 1):
        busy = others_busy()
        if busy:
            log(f"⚠️ {name}: 有别的任务在跑，5 分钟后重试（{attempt}/{retries}）")
            for b in busy:
                log(f"    [busy] {b}")
            time.sleep(300)
            continue
        log("$ " + " ".join(str(c) for c in cmd))
        rc = subprocess.run(cmd, env=env()).returncode
        log(f"{name} → exit={rc}")
        return rc
    log(f"⚠️ {name}: {retries} 次都被占用，放弃")
    return 2


def verify_train(model_dir):
    try:
        return "完成 best top1=" in open(
            os.path.join(model_dir, "train.log"), encoding="utf-8",
            errors="replace").read()
    except Exception:
        return False


def do_region(region, model_dir):
    out = {"recover": None, "train": None, "verified": False, "model_dir": model_dir}
    rc = run_step(f"recover {region}",
                  [PY, "-u", os.path.join(TOOLS, "recover_region.py"),
                   "--region", region])
    out["recover"] = rc
    if rc != 0:
        log(f"⚠️ {region} 恢复失败 exit={rc} —— 跳过该区训练，继续下一区")
        return out

    ds = os.path.join(CODE_TOOLS, "data", region, "dataset.csv")
    cmd = [PY, "-u", os.path.join(TOOLS, "train.py"),
           "--dataset", ds,
           "--model", "mobilenetv4_conv_small.e2400_r224_in1k",
           "--epochs", "30", "--batch", "12", "--accum", "6", "--workers", "0",
           "--out", model_dir]
    fv = os.path.join(CODE_TOOLS, "data", region, "frozen_val.csv")
    if os.path.exists(fv):
        cmd += ["--val-list", fv]
        log(f"{region}: 用冻结验证集 {fv}（与旧模型同口径可比）")
    else:
        log(f"{region}: ⚠️ 没有 frozen_val.csv，用 train.py 自己的 8:2 划分")
    out["train"] = run_step(f"train {region}", cmd)
    out["verified"] = verify_train(model_dir)
    if out["verified"]:
        open(os.path.join(model_dir, "PIPELINE_DONE"), "w",
             encoding="utf-8").write(
            time.strftime("%Y-%m-%d %H:%M:%S") + " overnight_chain")
        log(f"✅ {region} 训练完成 -> {model_dir}")
    else:
        log(f"⚠️ {region} 训练进程退出码 {out['train']}，但 train.log 没有"
            f"「完成 best top1=」—— 判为未完成，没写 PIPELINE_DONE")
    return out


def main():
    results = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "note": "华东由既有守候进程负责；本链接力华南、华北。模型进 *_rec 新目录不覆盖旧模型。",
        "morning_todo": [
            "labels/图鉴重生成：huadong / huanan_sw / huabei 三个新模型都要走"
            " copy_models_to_android.py + regen_guide.py --check（TFLite 导出需人工）",
            "监控台 tools/dashboard/train_dashboard.py 的 REGIONS 把 huanan_sw / huabei 指到 *_rec 目录并重启",
            "GBIF 补不上的种走 fetch_inat_region.py：华南绿鹭、华北牛背鹭+金眶鸻、东北 iNat 补量 19,547",
        ],
    }
    save_status(results)
    log(f"=== 无人值守链启动 pid={results['pid']} ===")

    results["huadong_wait"] = wait_huadong(results)
    save_status(results)

    results["huanan_sw"] = do_region(
        "huanan_sw", os.path.join(CODE_TOOLS, "models", "huanan_sw_mnv4_rec"))
    save_status(results)

    results["huabei"] = do_region(
        "huabei", os.path.join(CODE_TOOLS, "models", "huabei_mnv4_rec"))
    results["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_status(results)

    log("=== 链结束 ===")
    for k in ("huadong_wait", "huanan_sw", "huabei"):
        log(f"  {k}: {results[k]}")
    log(f"状态文件 {STATUS_PATH}")
    for t in results["morning_todo"]:
        log(f"  ☀ 早上要做的: {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

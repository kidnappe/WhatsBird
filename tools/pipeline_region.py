"""One-command regional pipeline, built for the migration pack.

Chain: species list -> GBIF download -> iNat top-up (auto when GBIF is
short) -> taxon audit -> dataset clean -> train -> eval. Stages run strictly
in serial order (a training run must never overlap a GBIF download on a
16 GB machine) and each stage is skipped when its artifact already exists, so
after a crash you simply re-run the same command and it resumes where it
stopped.

iNat top-up: when the GBIF prefetch cache shows species below --per-species,
fetch_inat_region.py tops those species up from iNaturalist (bbox-filtered,
research-grade, same dataset.csv schema, license recorded per row). If every
species already reached the cap the stage is skipped entirely.

Usage (on the download machine, inside the unpacked migration pack):
  python tools/pipeline_region.py --region huanan_sw
  python tools/pipeline_region.py --region huanan_sw --dry-run   # print, don't run

Monitoring while it runs (second terminal):
  python tools/dashboard/train_dashboard.py --out models/<region>_mnv4 \
      --download-dir data --download-dataset data/<region>/dataset.csv
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_MODEL = "mobilenetv4_conv_small.e2400_r224_in1k"  # HANDOFF §2.1; 旧名 mobilenetv4_small_300 在 timm 里不存在


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [chain] {msg}", flush=True)
    return msg


def child_env():
    e = os.environ.copy()
    e.setdefault("PYTHONIOENCODING", "utf-8")       # GBIF 日志含中文，Windows 控制台默认 cp936 会炸
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # timm 预训练权重，国内直连拉不到
    return e


class Runner:
    """Run a stage as a subprocess; tee stdout to console + pipeline.log."""

    def __init__(self, log_path, dry_run):
        self.log_path = log_path
        self.dry_run = dry_run

    def run(self, title, cmd, extra_log=None):
        printable = " ".join(str(c) for c in cmd)
        line = f"=== {title} ===\n$ {printable}\n"
        if self.dry_run:
            print(f"[dry-run] {printable}", flush=True)
            return 0
        log(f"开始 {title}")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} {line}")
            handles = [lf] + ([open(extra_log, "a", encoding="utf-8")]
                              if extra_log else [])
            try:
                p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=child_env())
                for raw in p.stdout:
                    text = raw.decode("utf-8", "replace").rstrip("\n")
                    print(text, flush=True)
                    for h in handles:
                        h.write(text + "\n")
                        h.flush()
                p.wait()
            finally:
                for h in handles[1:]:
                    h.close()
        if p.returncode != 0:
            log(f"!! {title} 退出码 {p.returncode}，链条中止（重跑同一命令可断点续传）")
        else:
            log(f"完成 {title}")
        return p.returncode


def read_regions():
    path = os.path.join(REPO, "tools", "regions.json")
    with open(path, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", required=True, help="regions.json 的键，如 huanan_sw")
    ap.add_argument("--n", type=int, default=150, help="区域物种数（生成清单时用）")
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--models-root", default=os.path.join(REPO, "models"))
    ap.add_argument("--species-csv", default=None,
                    help="缺省 = tools/species/species_<region>_<n>.csv；不存在时自动用 geomodel 生成")
    ap.add_argument("--geomodel-dir", default=os.path.join(REPO, "tools", "geomodel"))
    # 下载参数（华北实跑值，见 PLAN §七）
    ap.add_argument("--per-species", type=int, default=300)
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--api-timeout", type=int, default=60)
    ap.add_argument("--page", type=int, default=200)
    # 训练参数（HANDOFF §2.1 配置）
    ap.add_argument("--model", default=TRAIN_MODEL)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-workers", type=int, default=4)
    # 阶段控制
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-inat", action="store_true",
                    help="禁用 iNat 自动补量（默认：GBIF 有物种不足 --per-species 时自动补）")
    ap.add_argument("--force-download", action="store_true",
                    help="即使 progress.json 标记 done 也重跑下载（内部断点续传）")
    ap.add_argument("--force-train", action="store_true", help="即使 best.pt 已存在也重训")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    a = ap.parse_args()

    regions = read_regions()
    if a.region not in regions:
        ap.error(f"--region 必须是 regions.json 之一: {', '.join(regions)}")
    bbox = regions[a.region]["bbox"]

    data_dir = os.path.join(a.data_root, a.region)
    out_dir = os.path.join(a.models_root, f"{a.region}_mnv4")
    species_csv = a.species_csv or os.path.join(
        REPO, "tools", "species", f"species_{a.region}_{a.n}.csv")
    log_path = os.path.join(data_dir, "pipeline.log")
    runner = Runner(log_path, a.dry_run)

    log(f"区域 {a.region} bbox={bbox}\n"
        f"  数据 -> {data_dir}\n  模型 -> {out_dir}\n"
        f"  物种 -> {species_csv}\n  总日志 -> {log_path}")

    def run_or_die(title, cmd, **kw):
        if runner.run(title, cmd, **kw) != 0:
            sys.exit(1)

    # ---- 阶段 0：物种清单 ----
    if os.path.exists(species_csv):
        log(f"物种清单已存在，跳过: {species_csv}")
    else:
        onnx = os.path.join(a.geomodel_dir,
                            "BirdNET+_Geomodel_V3.0.4_Global_14K_FP16.onnx")
        if not os.path.exists(onnx):
            # 只提示不拦截：选种脚本会依次尝试 --geomodel-dir、主开发机路径、
            # 包内 tools/geomodel，都找不到才会带着明确报错退出
            log(f"提示: {onnx} 不存在，选种脚本将尝试其他 geomodel 路径")
        run_or_die("生成区域物种清单", [
            sys.executable, os.path.join(REPO, "tools", "pick_region_species.py"),
            "--region", a.region, "--n", a.n, "--out", species_csv,
            "--geomodel-dir", a.geomodel_dir])

    # ---- 阶段 1：下载 ----
    progress_path = os.path.join(data_dir, "progress.json")
    done = False
    if os.path.exists(progress_path):
        try:
            done = json.load(open(progress_path, encoding="utf-8")).get("stage") == "done"
        except (OSError, ValueError):
            done = False
    if a.skip_download:
        log("--skip-download：跳过下载阶段")
    elif done and not a.force_download:
        log("progress.json stage=done，下载已完成，跳过（--force-download 可强制重跑）")
    else:
        # gbif_build.log.<ts> 命名遵循仪表盘「数据下载」标签页的发现约定
        gbif_log = os.path.join(a.data_root,
                                f"gbif_build.log.{time.strftime('%Y%m%d-%H%M%S')}")
        log(f"下载日志 -> {gbif_log}（监控台 --download-dir 指向 {a.data_root}）")
        run_or_die("GBIF 下载", [
            sys.executable, os.path.join(REPO, "tools", "build_dataset_gbif.py"),
            "--species", species_csv, "--out", data_dir,
            "--per-species", a.per_species, "--max-pages", a.max_pages,
            "--workers", a.workers, "--api-timeout", a.api_timeout,
            "--page", a.page, "--bbox", *bbox, "--nice"], extra_log=gbif_log)

    # ---- 阶段 1.5：iNat 补量（数据不足时自动）----
    # 判定 = GBIF 预取缓存里存在可用量 < per-species 的物种；补量脚本逐种取
    # min(iNat 框内研究级量, 300-已有)，恰好补齐缺口，全部够量则整个阶段跳过。
    # 日志沿用 gbif_build.log.* 命名，监控台下载页自动解析并显示 iNat 直连条。
    if a.skip_inat:
        log("--skip-inat：跳过 iNat 补量阶段")
    elif not os.path.exists(os.path.join(REPO, "tools", "fetch_inat_region.py")):
        log("提示: 包内无 fetch_inat_region.py，跳过 iNat 补量")
    elif not os.path.exists(cache_path := os.path.join(data_dir, "gbif_prefetch_cache.json")):
        log("提示: 无 GBIF 预取缓存（下载未跑？），跳过 iNat 补量")
    else:
        cache = json.load(open(cache_path, encoding="utf-8"))
        short = {s: v.get("n", 0) for s, v in cache.items()
                 if v.get("n", 0) < a.per_species}
        if not short:
            log(f"GBIF 已覆盖全部 {len(cache)} 种（≥{a.per_species}/种），无需 iNat 补量")
        else:
            worst = sorted(short.items(), key=lambda kv: kv[1])[:5]
            log(f"GBIF 不足 {a.per_species}/种 的物种 {len(short)} 个"
                f"（最少: {', '.join(f'{s} {n}' for s, n in worst)}），启动 iNat 补量")
            inat_log = os.path.join(a.data_root,
                                    f"gbif_build.log.{time.strftime('%Y%m%d-%H%M%S')}")
            run_or_die("iNat 补量", [
                sys.executable, os.path.join(REPO, "tools", "fetch_inat_region.py"),
                "--species", species_csv, "--out", data_dir,
                "--per-species", str(a.per_species),
                "--bbox", *bbox, "--nice"], extra_log=inat_log)

    # ---- 阶段 2：taxon 复核（audit）----
    cache_path = os.path.join(data_dir, "gbif_prefetch_cache.json")
    audit_path = os.path.join(data_dir, "gbif_taxon_audit.csv")
    if os.path.exists(audit_path):
        log("audit 产物已存在，跳过")
    else:
        run_or_die("taxon audit", [
            sys.executable, os.path.join(REPO, "tools", "audit_taxon_keys.py"),
            "--cache", cache_path, "--out", audit_path])

    # ---- 阶段 3：清洗（genus 污染修复，原地重写 dataset.csv）----
    dataset_csv = os.path.join(data_dir, "dataset.csv")
    clean_stamp = os.path.join(data_dir, ".clean_done")
    if os.path.exists(clean_stamp):
        log("清洗已完成（.clean_done），跳过")
    else:
        run_or_die("清洗数据集", [
            sys.executable, os.path.join(REPO, "tools", "clean_dataset.py"),
            "--dataset", dataset_csv, "--audit", audit_path,
            "--species-list", species_csv])
        if not a.dry_run:
            open(clean_stamp, "w").close()

    # ---- 阶段 4：训练 ----
    best_pt = os.path.join(out_dir, "best.pt")
    hist_json = os.path.join(out_dir, "history.json")
    trained = os.path.exists(hist_json)   # history.json 只在训练真正跑完时写
    if a.skip_train:
        log("--skip-train：跳过训练阶段")
    elif os.path.exists(best_pt) and trained and not a.force_train:
        log(f"{best_pt} 已存在且训练已跑完（history.json 在），跳过训练"
            f"（--force-train 可强制重训）")
    else:
        if os.path.exists(best_pt) and not trained:
            # train.py 是「每个 epoch 只要刷新纪录就写 best.pt」，所以训练被中断后
            # best.pt 一定已经存在。原来只看 best.pt 在不在就跳过训练 → 会把半成品
            # 直接送进评估，最后还打印"链条结束。产物：…"（半成品当成品报成功，
            # 和 .exit 那类事故同形）。
            log("⚠ best.pt 在、但没有 history.json：上一轮训练是中断的，"
                "不能拿半成品去评估 → 本轮重新训练（要强行跳过请用 --skip-train）")
        elif os.path.exists(best_pt) and a.force_train:
            log("--force-train：不管 best.pt 是否已存在，强制重训")
        run_or_die("训练", [
            sys.executable, os.path.join(REPO, "tools", "train.py"),
            "--dataset", dataset_csv, "--model", a.model,
            "--epochs", a.epochs, "--batch", a.batch, "--accum", a.accum,
            "--head-epochs", a.head_epochs, "--lr", a.lr,
            "--workers", a.train_workers, "--out", out_dir])

    # ---- 阶段 5：评估 ----
    if not os.path.exists(best_pt):
        log("没有 best.pt，跳过评估")
    else:
        if not os.path.exists(hist_json):
            log("⚠ 准备评估的 best.pt 没有配套 history.json —— 它可能不是一次完整训练的"
                "产物，评估数字请谨慎解读")
        run_or_die("评估", [
            sys.executable, os.path.join(REPO, "tools", "eval_report.py"),
            "--ckpt", best_pt, "--dataset", dataset_csv])

    log(f"链条结束。产物：\n  数据 {data_dir}\n  模型 {best_pt}\n"
        f"  评估 {os.path.splitext(best_pt)[0]}_eval.json")
    if not a.dry_run:
        log("回传本机时打包这两个目录即可：data/<region> 与 models/<region>_mnv4")


if __name__ == "__main__":
    main()

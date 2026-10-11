# -*- coding: utf-8 -*-
"""detb_pack_monitor 解析自测（无第三方依赖，直接 python 跑）。

为什么要有这个测试：监控台的判读直接决定外包同学"看到绿灯就放心关机"，而
踩过的两个坑都只在**特定日志形态**下才现形，肉眼看不出：
  1. yolox 的 after_train 是**无条件**打印 "Training of experiment is done and
     the best AP is X" 的——中途崩溃时照样打（且常是 0.00）。若"完成"判定排在
     "崩溃"之前，页面会把一具尸体显示成绿色"训练完成"。
  2. pycocotools 的 summarize 一次输出 6 行 AP + 4 行 AR，其中 area=large 那行
     数值最高。若正则不锁死 "IoU=0.50:0.95 | area=all"，best_ap 会被大目标行污染。
本测试用真实日志形态（loguru 单文件 train_log.txt、多行 message 续行无前缀）造
两份日志，验证这两条口径。

Usage: python tools/detector_b/test_detb_pack_monitor.py
"""
import os
import shutil
import sys
import tempfile

# Windows 控制台默认 GBK，所以本文件只用 GBK 能编出的字符（不用箭头/勾叉符号等），
# 并且不强行改 stdout 编码——那会把正常的 GBK 控制台打成乱码。只加容错。
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detb_pack_monitor as M  # noqa: E402

OK, BAD = "[OK]", "[FAIL]"

# pycocotools summarize 的真实输出形态（前 6 行 AP，area=large 故意给最高值）
AP_BLOCK = """
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.688
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
"""


def it_line(ts, ep, it, itot, loss=4.2, itime=1.20, size=640, eta="8:42:49"):
    return (f"{ts}.100 | INFO     | yolox.core.trainer:after_iter:270 - "
            f"epoch: {ep}/50, iter: {it}/{itot}, gpu mem: 4200Mb, mem: 12.1Gb, "
            f"iter_time: {itime}s, data_time: 0.031s, total_loss: {loss}, "
            f"iou_loss: 2.2, l1_loss: 0.0, conf_loss: 1.5, cls_loss: 0.5, "
            f"lr: 1.250e-03, size: {size}, ETA: {eta}")


def build_log(dirpath, n_ep_iters=40, itot=100, crash=False, n_evals=1):
    """造一份形态逼真的 train_log.txt。多行 message（AP 块）续行不带 loguru 前缀。"""
    os.makedirs(dirpath, exist_ok=True)
    L = []
    L.append("2026-10-06 19:43:40.356 | INFO     | yolox.core.trainer:before_train:132 - "
             "args: Namespace(experiment_name='yolox_bird_640', batch_size=8, fp16=True)")
    L.append("2026-10-06 19:43:40.359 | INFO     | yolox.core.trainer:before_train:133 - "
             "exp value:")
    L.append("┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓")
    L.append("┃ keys            ┃ values                ┃")
    L.append("┗━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━┛")
    for ep in range(1, 8):
        L.append(f"2026-10-06 19:4{ep}:00.000 | INFO     | "
                 f"yolox.core.trainer:before_epoch:218 - ---> start train epoch{ep}")
        for i in range(1, n_ep_iters + 1):
            L.append(it_line(f"2026-10-06 19:4{ep}:{(i * 10) % 60:02d}", ep, i, itot))
        if ep % 5 == 0 and (ep // 5) <= n_evals:
            L.append(f"2026-10-06 20:{ep:02d}:03.351 | INFO     | "
                     f"yolox.evaluators.coco_evaluator:evaluate_prediction:259 - "
                     f"Evaluate in main process...")
            L.append(f"2026-10-06 20:{ep:02d}:05.000 | INFO     | "
                     f"yolox.core.trainer:evaluate_and_save_model:381 - " + AP_BLOCK)
            L.append(f"2026-10-06 20:{ep:02d}:06.000 | INFO     | "
                     f"yolox.core.trainer:save_ckpt:402 - Save weights to ./YOLOX_outputs\\yolox_bird_640")
    if crash:
        # 真实崩溃形态：ERROR + after_train 仍然照打 "is done"（值 0.00）
        L.append("2026-10-06 20:43:04.667 | ERROR    | yolox.core.trainer:train:79 - "
                 "Exception in training: ")
        L.append("2026-10-06 20:43:04.667 | INFO     | "
                 "yolox.core.trainer:after_train:200 - Training of experiment is done "
                 "and the best AP is 0.00")
        L.append("2026-10-06 20:43:04.667 | ERROR    | yolox.core.launch:launch:98 - "
                 "An error has been caught in function 'launch'")
        L.append("Traceback (most recent call last):")
        L.append('  File "yolox/layers/jit_ops.py", line 83, in load')
        L.append("RuntimeError: Ninja is required for compilation")
    else:
        L.append("2026-10-06 23:59:59.000 | INFO     | "
                 "yolox.core.trainer:after_train:200 - Training of experiment is done "
                 "and the best AP is 0.412")
    path = os.path.join(dirpath, "train_log.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    # 伪造 ckpt：只测存在性/mtime，不需要真权重
    open(os.path.join(dirpath, "latest_ckpt.pth"), "wb").write(b"x" * 1024)
    if not crash:
        open(os.path.join(dirpath, "best_ckpt.pth"), "wb").write(b"x" * 2048)
    return path


def make_root(tmp, kind):
    exp = "yolox_bird_640"
    d = os.path.join(tmp, "yolox_repo", "YOLOX_outputs", exp)
    build_log(d, crash=(kind == "crash"), n_evals=1)
    return tmp, exp


FAILS = []


def check(cond, msg):
    print(("  %s " % OK if cond else "  %s " % BAD) + msg)
    if not cond:
        FAILS.append(msg)


def test_eval_schedule():
    """评测调度：进入 no_aug 阶段后 interval 变 1，不是固定每 5 轮。

    这是从 trainer.py:220/228/235 反推出来的真值——曾按 range(5,51,5) 硬编码，
    结果第 9 个 AP 块起 epoch 全错位、后 8 块直接丢标签。
    """
    print("[0] 评测调度（trainer 的 no_aug 折点）")
    ev = M.eval_epochs(50, 10, 5)
    check(ev == [5, 10, 15, 20, 25, 30, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
          f"ep50/no_aug10/interval5 共 18 次且折点在 40（实得 {len(ev)} 次: {ev}）")
    check(len(ev) != len(list(range(5, 51, 5))),
          "不等于旧的等差 10 次假设（否则说明折点没生效）")
    check(ev[:8] == [5, 10, 15, 20, 25, 30, 35, 40] and ev[8] == 41,
          "前 8 次每 5 轮、第 9 次起每轮（40→41 而非 45）")
    # no_aug=0 时应退化为纯每 5 轮
    check(M.eval_epochs(50, 0, 5) == list(range(5, 51, 5)),
          "no_aug=0 → 纯每 5 轮")
    # 边界：总轮数小于间隔时不该凭空造评测
    check(M.eval_epochs(3, 10, 5) == [], f"极短训练无评测（实得 {M.eval_epochs(3,10,5)}）")


def test_full_schedule_pairing(tmp):
    """满程日志：18 个 AP 块必须逐一配到正确的 epoch（本轮 bug 的回归测试）。

    造一份 50 轮、按 trainer 真实节奏（5,10,…,35 + 40..50）产出 AP 块的日志，
    AP 值故意设成可反查（epoch*1.0），配对错位会立刻被值不匹配抓到。
    """
    print("[6] 满程 18 次评测的 AP 与 epoch 配对")
    root = os.path.join(tmp, "full")
    exp = "yolox_bird_640"
    d = os.path.join(root, "yolox_repo", "YOLOX_outputs", exp)
    os.makedirs(d, exist_ok=True)
    sched = M.eval_epochs(50, 10, 5)
    L = ["2026-10-06 19:00:00.000 | INFO     | yolox.core.trainer:before_train:132 - args"]
    for ep in range(1, 51):
        L.append(f"2026-10-06 20:{ep:02d}:00.000 | INFO     | "
                 f"yolox.core.trainer:before_epoch:218 - ---> start train epoch{ep}")
        if ep == 40:
            L.append("2026-10-06 20:40:01.000 | INFO     | "
                     "yolox.core.trainer:before_epoch:221 - --->No mosaic aug now!")
        L.append(it_line(f"2026-10-06 20:{ep:02d}:10", ep, 50, 100))
        if ep in sched:
            # AP 值 = epoch 的千分之一，便于反查配到了哪一轮
            ap = ep / 1000.0
            block = (f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | "
                     f"maxDets=100 ] = {ap:.3f}\n"
                     f" Average Precision  (AP) @[ IoU=0.50      | area=   all | "
                     f"maxDets=100 ] = {ap + 0.1:.3f}\n"
                     f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | "
                     f"maxDets=100 ] = 0.999\n")
            L.append(f"2026-10-06 20:{ep:02d}:20.000 | INFO     | "
                     f"yolox.core.trainer:evaluate_and_save_model:381 - " + block)
    L.append("2026-10-06 23:59:59.000 | INFO     | yolox.core.trainer:after_train:200"
             " - Training of experiment is done and the best AP is 0.050")
    with open(os.path.join(d, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    open(os.path.join(d, "best_ckpt.pth"), "wb").write(b"x")

    r = M.collect(root, exp)
    hist = r["eval"]["ap_hist"]
    check(len(hist) == 18, f"认出全部 18 个评测块的 AP（实得 {len(hist)}）")
    check([h["epoch"] for h in hist] == sched,
          f"epoch 序列 = 真实调度（实得 {[h['epoch'] for h in hist]}）")
    # 每个块的 AP 应等于其 epoch/1000*100
    bad = [h for h in hist if abs(h["ap"] - h["epoch"] * 0.1) > 1e-6]
    check(not bad, f"每个 AP 值都配到自己的 epoch（不匹配项 {bad[:3]}）")
    check(all(abs(h["ap50"] - (h["epoch"] * 0.1 + 10)) < 1e-6 for h in hist),
          "AP50 也按同序配对")
    check(r["eval"]["best_ap"] == 5.0,
          f"best_ap=最大总 AP(50/1000*100=5.0)，不被 area=large 的 99.9 污染"
          f"（实得 {r['eval']['best_ap']}）")
    check(r["eval"]["finished"], "满程日志判为已完成")


def test_observed_epochs_beat_computed(tmp):
    """观测到的评测归属优先于推算：续训重跑会让同一 epoch 评两次。

    纯推算（按调度列表取第 k 个）遇到这种情况必然错位；按日志里 "start train
    epochN" 记下的归属则天然正确。
    """
    print("[7] 观测优先：续训重跑同一 epoch 的评测")
    root = os.path.join(tmp, "resume_dup")
    exp = "yolox_bird_640"
    d = os.path.join(root, "yolox_repo", "YOLOX_outputs", exp)
    os.makedirs(d, exist_ok=True)
    L = ["2026-10-06 19:00:00.000 | INFO     | yolox.core.trainer:before_train:132 - args"]
    for ep in (1, 2, 3, 4, 5):
        L.append(f"2026-10-06 20:0{ep}:00.000 | INFO     | "
                 f"yolox.core.trainer:before_epoch:218 - ---> start train epoch{ep}")
        L.append(it_line(f"2026-10-06 20:0{ep}:10", ep, 50, 100))
        if ep == 5:
            ap = 0.3
            L.append(f"2026-10-06 20:05:20.000 | INFO     | "
                     f"yolox.core.trainer:evaluate_and_save_model:381 -  Average "
                     f"Precision  (AP) @[ IoU=0.50:0.95 | area=   all | "
                     f"maxDets=100 ] = {ap}")
    # 崩了、resume、ep5 又评了一次（值不同）——真实场景：续训后重跑
    L.append("2026-10-06 21:00:00.000 | INFO     | "
             "yolox.core.trainer:resume_train:331 - loaded checkpoint 'True' (epoch 5)")
    L.append("2026-10-06 21:01:00.000 | INFO     | "
             "yolox.core.trainer:before_epoch:218 - ---> start train epoch6")
    L.append(it_line("2026-10-06 21:01:10", 6, 50, 100))
    L.append("2026-10-06 21:02:00.000 | INFO     | "
             "yolox.core.trainer:evaluate_and_save_model:381 -  Average Precision  "
             "(AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.5")
    with open(os.path.join(d, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    r = M.collect(root, exp)
    got = [(h["epoch"], h["ap"]) for h in r["eval"]["ap_hist"]]
    check(got == [(5, 30.0), (6, 50.0)],
          f"两次评测分别归到 ep5/ep6（实得 {got}）")
    check(r["eval"]["best_ap"] == 50.0, f"best_ap=50.0（实得 {r['eval']['best_ap']}）")


def main():
    tmp = tempfile.mkdtemp(prefix="detb_mon_test_")
    try:
        test_eval_schedule()

        print("[1] 正常完成路径（有真 AP 块 + 完成行）")
        root, exp = make_root(os.path.join(tmp, "ok"), "ok")
        d = M.collect(root, exp)
        check(d["status"]["level"] == "ok", f"状态为 ok（实得 {d['status']}）")
        check(d["eval"]["best_ap"] == 41.2,
              f"best_ap 取 area=all 的 0.412→41.2，不被 area=large 0.731 污染"
              f"（实得 {d['eval']['best_ap']}）")
        check(d["eval"]["ap_hist"] and d["eval"]["ap_hist"][0]["ap50"] == 68.8,
              f"AP50 取 area=all 的 0.688→68.8（实得 {d['eval']['ap_hist'][:1]}）")
        check(len(d["eval"]["ap_hist"]) == 1, "只认出 1 个评测块（6 行 AP 不能数成 6 次）")
        check(d["run"]["epoch"] == 7 and d["run"]["total_epochs"] == 50,
              f"epoch 采到 7/50（实得 {d['run']['epoch']}/{d['run']['total_epochs']}）")
        check(d["run"]["progress_pct"] and d["run"]["progress_pct"] > 0,
              f"总进度已算出（实得 {d['run']['progress_pct']}%）")
        check(d["eval"]["best_ckpt"] and d["eval"]["best_ckpt"]["exists"], "best_ckpt 存在")
        check(not d["error"]["is_error"], "无崩溃标记")
        rl = {r["name"]: r["exists"] for r in d["return_list"]}
        check(rl.get("best_ckpt.pth") and rl.get("train_log.txt"), f"回传清单齐（{rl}）")

        print("[2] 崩溃路径（Ninja/JIT 坑，after_train 仍打 'is done 0.00'）")
        root2, exp2 = make_root(os.path.join(tmp, "crash"), "crash")
        d2 = M.collect(root2, exp2)
        check(d2["status"]["level"] == "bad",
              f"崩溃必须显示为 bad，不能是绿色完成（实得 {d2['status']}）")
        check(d2["error"]["is_error"] and d2["error"].get("fatal"),
              "崩溃判为致命（traceback 之后没有迭代行）")
        check(d2["eval"]["best_ap"] == 41.2,
              f"best_ap 仍是崩溃前真评过的 41.2，不被假 0.00 覆盖"
              f"（实得 {d2['eval']['best_ap']}）")
        check(any("Ninja" in x or "Traceback" in x for x in d2["error"]["tail"]),
              "报错尾部带上了可读证据")
        rl2 = {r["name"]: r["exists"] for r in d2["return_list"]}
        check(not rl2.get("best_ckpt.pth"), "崩溃路径 best_ckpt 未生成 → 清单如实显示")

        print("[2b] 崩溃后已续训（traceback 之后又有迭代行）")
        d2b_dir = os.path.join(tmp, "resumed", "yolox_repo", "YOLOX_outputs",
                               "yolox_bird_640")
        build_log(d2b_dir, crash=True, n_evals=1)
        with open(os.path.join(d2b_dir, "train_log.txt"), "a", encoding="utf-8") as f:
            f.write("2026-10-07 00:10:00.000 | INFO     | yolox.core.trainer:resume_train:331"
                    " - loaded checkpoint 'True' (epoch 5)\n")
            f.write(it_line("2026-10-07 00:11:00", 6, 20, 100, loss=4.4) + "\n")
        d2b = M.collect(os.path.join(tmp, "resumed"), "yolox_bird_640")
        check(d2b["status"]["level"] == "warn" and "续训" in d2b["status"]["text"],
              f"崩过但已恢复 → warn + 提示续训，不再报红（实得 {d2b['status']}）")
        check(not d2b["error"].get("fatal"), "续训后不再判为致命")

        print("[3] 没开训（没有 train_log.txt）")
        d3 = M.collect(os.path.join(tmp, "empty"), None)
        check(d3["status"]["level"] == "idle" and not d3["log"]["exists"],
              f"空目录显示未启动（实得 {d3['status']}）")
        check(len(d3["return_list"]) == 0, "空目录不编回传清单")

        print("[4] 增量解析：追加只解析新增字节 + 半行跨轮拼接")
        root4 = os.path.join(tmp, "inc")
        d4dir = os.path.join(root4, "yolox_repo", "YOLOX_outputs", "yolox_bird_640")
        os.makedirs(d4dir, exist_ok=True)
        p4 = os.path.join(d4dir, "train_log.txt")
        with open(p4, "w", encoding="utf-8") as f:
            f.write(it_line("2026-10-06 19:00:00", 1, 10, 100, loss=5.0) + "\n")
        d4a = M.collect(root4, "yolox_bird_640")
        check(d4a["run"]["iter"] == 10 and len(d4a["loss_series"]) == 1,
              f"增量第 1 轮：iter 10 / 1 点（实得 {d4a['run']['iter']} / "
              f"{len(d4a['loss_series'])}）")
        # 追加一整行 + 一个**没有换行的半行**（真实 tail -f 形态）
        with open(p4, "a", encoding="utf-8") as f:
            f.write(it_line("2026-10-06 19:00:30", 1, 20, 100, loss=4.0) + "\n")
            f.write(it_line("2026-10-06 19:01:00", 1, 30, 100, loss=3.0))  # 半行，无 \n
        d4b = M.collect(root4, "yolox_bird_640")
        check(d4b["run"]["iter"] == 20,
              f"半行不该被当成完整行解析（应停在 iter 20，实得 {d4b['run']['iter']}）")
        check(len(d4b["loss_series"]) == 2,
              f"只多 1 个点（半行不算，实得 {len(d4b['loss_series'])}）")
        # 把半行补全（换行到了）→ 这次必须认出来
        with open(p4, "a", encoding="utf-8") as f:
            f.write("\n")
        d4c = M.collect(root4, "yolox_bird_640")
        check(d4c["run"]["iter"] == 30 and len(d4c["loss_series"]) == 3,
              f"半行补全后接力解析（应 iter 30 / 3 点，实得 {d4c['run']['iter']} / "
              f"{len(d4c['loss_series'])}）")

        print("[5] 日志被截断重写 → 增量缓存自愈，不留残影")
        with open(p4, "w", encoding="utf-8") as f:      # 截断：size 变小
            f.write(it_line("2026-10-07 09:00:00", 1, 5, 100, loss=9.9) + "\n")
        d5 = M.collect(root4, "yolox_bird_640")
        check(d5["run"]["epoch"] == 1 and d5["run"]["iter"] == 5
              and d5["run"]["loss"] == 9.9 and len(d5["loss_series"]) == 1,
              f"截断后按新内容重来（实得 ep {d5['run']['epoch']} iter "
              f"{d5['run']['iter']} loss {d5['run']['loss']} "
              f"pts {len(d5['loss_series'])}）")

        test_full_schedule_pairing(tmp)
        test_observed_epochs_beat_computed(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILS:
        print(f"{BAD} {len(FAILS)} 项失败")
        for f in FAILS:
            print("   -", f)
        return 1
    print(f"{OK} 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

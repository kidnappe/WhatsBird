# -*- coding: utf-8 -*-
"""COCO train2017 bird 子集下载（对策④ 小目标过采样的数据源）。

GT 直接用官方标注（instances_train2017.json 已在盘上，无需重下），图片按需下载。
⚠️ 训练过采样只能用 train2017——val2017 是 bench_detector_a 的评测集，混进训练
   会污染 PC 验收门槛（COCO bird 召回指标作废）。

输出 ROOT = E:\\code\\code tools\\data\\detector_b\\coco_bird\\：
  train2017_bird/<file>   图片（仅含 ≥1 个非 crowd bird 实例的图）
  labels/<stem>.txt       YOLO txt（class 0，cx cy w h 归一化，非 crowd bird 实例）
  manifest.json           {n_images, n_boxes, n_tiny_small, wh: {file: [w,h]}}

可重跑：已有图片/标签自动跳过。约 4~5k 图 / 700MB。
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _progress import update as prog_update  # noqa: E402  progress.json（监控台读）

COCO_ANN = r"E:\code\code tools\data\coco\annotations\instances_train2017.json"
ROOT = r"E:\code\code tools\data\detector_b\coco_bird"
IMG_DIR = os.path.join(ROOT, "train2017_bird")
LAB_DIR = os.path.join(ROOT, "labels")
URL = "http://images.cocodataset.org/train2017/{}"
TINY_FRAC, SMALL_FRAC = 0.0025, 0.01  # bench_detector_a 同口径


def fetch(fn):
    dst = os.path.join(IMG_DIR, fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    for _ in range(3):
        try:
            urllib.request.urlretrieve(URL.format(fn), dst)
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    """进程锁：同一时刻只允许一个下载实例（监控台/driver/手动三处都可能触发它）。"""
    import psutil
    os.makedirs(ROOT, exist_ok=True)  # 锁文件在 ROOT 下，必须先于锁创建目录
    lockf = os.path.join(ROOT, ".download.lock")
    if os.path.exists(lockf):
        try:
            old = int(open(lockf).read().strip() or 0)
            alive = psutil.pid_exists(old)
        except Exception:
            alive = False
        if alive:
            print(f"已有下载实例在跑（PID {old}），本实例跳过（幂等，图已存在的自动跳）")
            return 0
        print(f"发现陈旧锁（PID {old} 已不在），接管")
    with open(lockf, "w") as f:
        f.write(str(os.getpid()))
    try:
        return _run()
    finally:
        try:
            os.remove(lockf)
        except OSError:
            pass


def _run():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LAB_DIR, exist_ok=True)

    print(f"读 {COCO_ANN}（~1 分钟）...")
    with open(COCO_ANN) as f:
        ann = json.load(f)
    bird = [c["id"] for c in ann["categories"] if c["name"] == "bird"][0]
    imgs = {im["id"]: im for im in ann["images"]}
    by_img = {}
    for a in ann["annotations"]:
        if a["category_id"] == bird and not a.get("iscrowd"):
            by_img.setdefault(a["image_id"], []).append(a["bbox"])
    print(f"bird 图片 {len(by_img)}，实例 {sum(len(v) for v in by_img.values())}")

    wh, todo, n_box, n_ts = {}, [], 0, 0
    for iid, boxes in by_img.items():
        im = imgs[iid]
        fn, w, h = im["file_name"], im["width"], im["height"]
        wh[fn] = [w, h]
        stem = os.path.splitext(fn)[0]
        lab = os.path.join(LAB_DIR, stem + ".txt")
        if not os.path.exists(lab):
            with open(lab, "w") as f:
                for x, y, bw, bh in boxes:
                    f.write(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                            f"{bw / w:.6f} {bh / h:.6f}\n")
        n_box += len(boxes)
        if any(bw * bh / (w * h) < SMALL_FRAC for _, _, bw, bh in boxes):
            n_ts += 1
        if not (os.path.exists(os.path.join(IMG_DIR, fn)) and
                os.path.getsize(os.path.join(IMG_DIR, fn)) > 0):
            todo.append(fn)

    print(f"需下载 {len(todo)} 张（已有 {len(wh) - len(todo)}）...")
    prog_update(coco={"downloaded": len(wh) - len(todo), "target": len(wh),
                      "fail": 0, "done": False})
    fail = []
    with ThreadPoolExecutor(8) as ex:
        for i, (fn, ok) in enumerate(zip(todo, ex.map(fetch, todo))):
            if not ok:
                fail.append(fn)
            if (i + 1) % 200 == 0:
                prog_update(coco={"downloaded": len(wh) - len(todo) + i + 1,
                                  "target": len(wh), "fail": len(fail),
                                  "done": False})
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(todo)}")
    for fn in fail:  # 失败的重试一轮
        fetch(fn)
    fail = [fn for fn in fail if not os.path.exists(os.path.join(IMG_DIR, fn))]

    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump({"n_images": len(wh) - len(fail), "n_boxes": n_box,
                   "n_tiny_small": n_ts, "wh": wh}, f)
    print(f"完成：图片 {len(wh) - len(fail)} / 实例 {n_box} / tiny+small 图 {n_ts}，下载失败 {len(fail)}")
    prog_update(coco={"downloaded": len(wh) - len(fail), "target": len(wh),
                      "fail": len(fail), "done": True})
    return 1 if len(fail) > 50 else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""detB 数据链的 progress.json 合并写。

口径同 build_dataset_gbif 的 flush_progress：读-改-写，各脚本只写自己的字段
（pseudo_label 写 img_done/box/...，make_coco_bird_subset 写 coco 子块，
run_pipeline_when_free 写 stage），互不覆盖；读方是监控台 tools/dashboard/train_dashboard.py
的 update_detb()。原子替换写（tmp + os.replace），监控台并发读不会撕裂。

字段约定（E:\\code\\code tools\\data\\detector_b\\progress.json）：
  stage         总阶段：pseudo_label / calib / coco / done / failed（driver 管）
  img_done/img_total/box/low_conf/rescan_box/img_zero/rate_ms/eta_min/pl_done
                伪标签进度（pseudo_label 每 200 图刷一次）
  calib_files   校准清单张数（make_calib_b）
  coco          {downloaded, target, fail, done}（make_coco_bird_subset）
"""
import json
import os
import time

DB = r"E:\code\code tools\data\detector_b"
PJ = os.path.join(DB, "progress.json")


def update(**kw):
    os.makedirs(DB, exist_ok=True)
    try:
        cur = json.load(open(PJ, encoding="utf-8"))
    except Exception:
        cur = {}
    cur.update(kw)
    cur["ts"] = time.time()
    cur["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = PJ + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False)
    os.replace(tmp, PJ)

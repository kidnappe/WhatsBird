# -*- coding: utf-8 -*-
"""诊断 BRunner 口径：bench 报 0 召回但 export_b_onnx.smoke 在同图打出 0.77/IoU0.93。

两种可能，结论完全相反：
  A. 工具口径错（评测器/喂法/解码列序）  → 模型其实活着，是尺子坏了
  B. 模型真没响应                      → ep5 早停迹象，训练侧要查

跑法：python diag_b_ep5.py [n_images]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"E:\code\observebird\tools")
sys.path.insert(0, r"E:\code\observebird\tools\detector_b")

import bench_detector_a as A          # noqa: E402
from bench_detector_b import BRunner  # noqa: E402

DEC = r"E:\code\code tools\detectors\yolox\yolox_bird_512_dec.onnx"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    gt = A.load_bird_gt()
    runner = BRunner("bird512_ep5", DEC)

    # 与 export_b_onnx.smoke 完全同一批文件、同一排序
    files = sorted(gt.keys())[:n]

    print("\n--- 逐图：raw 分数分布 vs BRunner.detect 返回 ---")
    for fn in files:
        bgr = cv2.imread(os.path.join(A.COCO_DIR, "val2017", fn))
        if bgr is None:
            continue
        # (1) 裸跑：不做任何阈值过滤，只看 bird 分数的分布
        x, r = A.letterbox(bgr, runner.size)
        out = runner.sess.run(None, {runner.inp_name: x})[0][0]   # [N,6]
        bird = out[:, 4] * out[:, 5]
        obj, cls = out[:, 4], out[:, 5]
        # (2) 走评测器口径
        dets = runner.detect(bgr, score_th=A.STORE_TH)

        print(f"{fn}: n_rows={len(out)} "
              f"bird[max={bird.max():.4f} mean={bird.mean():.5f}] "
              f"obj[max={obj.max():.4f}] cls[max={cls.max():.4f}] "
              f"-> detect@0.2 返回 {len(dets)} 个"
              + (f" top={dets[0][0]:.3f}" if dets else ""))

    print("\n--- 阈值敏感度：同一批图召回随阈值怎么变 ---")
    all_bird = []
    for fn in files:
        bgr = cv2.imread(os.path.join(A.COCO_DIR, "val2017", fn))
        if bgr is None:
            continue
        x, r = A.letterbox(bgr, runner.size)
        out = runner.sess.run(None, {runner.inp_name: x})[0][0]
        all_bird.append((fn, out[:, 4] * out[:, 5], out))
    for th in (0.2, 0.05, 0.01, 0.001):
        tot = 0
        for _fn, bird, _o in all_bird:
            tot += int((bird >= th).sum())
        print(f"  score>={th:<6}: {tot:>6} 个候选框 "
              f"(占 {tot / max(sum(len(b) for _f, b, _o in all_bird), 1):.5f})")

    print("\n--- 输出列序确认：dec 头 6 列到底是什么 ---")
    _fn, _bird, o0 = all_bird[0]
    k = int(np.argmax(_bird))
    print(f"  最高分行的 6 列原始值 = {np.round(o0[k], 4).tolist()}")
    print(f"  其中 obj(第5列)={o0[k,4]:.4f}  cls(第6列)={o0[k,5]:.4f}  "
          f"乘积={o0[k,4]*o0[k,5]:.4f}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""B 模型 PC bench（评测器全部复用 bench_detector_a：同测试集/阈值/NMS/尺寸桶）。

结果可与 A 路线表格并排对比。PC 验收门槛（主文档阶段 3，2026-10-06 起 PC 口径）：
  tiny 桶召回 ≥2× 现役（6.7% → ≥15% 方向）、small ≥70%、FP/图 ≤0.3、
  华北检出率不大幅回退（现役 77.5%）。PC 不赢不出包。

Usage: python bench_detector_b.py <dec.onnx> [name] [--limit N] [--also-s512]
"""
import argparse
import json
import os
import sys

OBS_TOOLS = r"E:\code\observebird\tools"
sys.path.insert(0, OBS_TOOLS)
import bench_detector_a as A  # noqa: E402

OUT = os.path.join(A.TOOLS, "data", "detector_b", "results")


class BRunner:
    """单类 YOLOX（dec onnx [1,N,6]）。喂法/NMS 与 A.YoloxRunner 完全一致。"""

    def __init__(self, name, onnx_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        i = self.sess.get_inputs()[0]
        self.size = int(i.shape[2])
        self.inp_name = i.name
        self.name = name
        print(f"{name}: 输入 {i.shape}, 输出 {self.sess.get_outputs()[0].shape}")

    def detect(self, bgr, score_th=A.STORE_TH):
        h, w = bgr.shape[:2]
        x, r = A.letterbox(bgr, self.size)
        out = self.sess.run(None, {self.inp_name: x})[0][0]  # [N,6]
        bird = out[:, 4] * out[:, 5]
        m = bird >= score_th
        if not m.any():
            return []
        rows = out[m]
        sc = bird[m]                      # ⚠️ 必须过滤后的分数：i 是 rows 的局部下标，
        dets = []                        #    用 bird[i] 会取到全量数组的错位元素（曾恒为 0，见 A.YoloxRunner 同款写法）
        for i in range(len(rows)):
            cx, cy, bw, bh = (float(v) for v in rows[i, :4] / r)
            dets.append((float(sc[i]), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        return A.nms(dets, A.YOLOX_NMS_IOU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("name", nargs="?", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--also-s512", action="store_true", help="同表带现役 s512 对照")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    gt = A.load_bird_gt()
    runners = {}
    if args.also_s512:
        runners["s512"] = A.YoloxRunner("s512", "yolox_s_512_dec.onnx")
    name = args.name or os.path.splitext(os.path.basename(args.onnx))[0]
    runners[name] = BRunner(name, args.onnx)

    summary = {}
    for nm, runner in runners.items():
        print(f"\n=== [{nm}] COCO bird 子集 ===")
        viz = os.path.join(OUT, "viz", nm)
        os.makedirs(viz, exist_ok=True)
        res = {"coco": A.eval_coco(runner, gt, args.limit, viz)}
        print(f"=== [{nm}] 华北 200 ===")
        res["huabei"] = A.eval_huabei(runner)
        summary[nm] = res
        with open(os.path.join(OUT, f"{nm}.json"), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    for th in (A.SSD_SCORE_TH, 0.25):
        print(f"\n===== 汇总 score≥{th} =====")
        print(f"{'model':<14}{'recall':>8}{'tiny<0.25%':>12}{'small<1%':>10}{'large':>9}"
              f"{'FP/图':>7}{'华北检出率':>10}{'ms/图':>8}")
        for nm, res in summary.items():
            c = res["coco"][f"th{th}"]
            hu = res["huabei"][f"th{th}"]
            print(f"{nm:<14}{c['recall']:>8.3f}"
                  f"{c['by_size']['tiny<0.25%']['recall']:>12.3f}"
                  f"{c['by_size']['small<1%']['recall']:>10.3f}"
                  f"{c['by_size']['large>=1%']['recall']:>9.3f}"
                  f"{c['fp_per_img']:>7.2f}{hu['det_rate']:>10.3f}"
                  f"{res['coco']['ms_per_img']:>8.1f}")


if __name__ == "__main__":
    main()

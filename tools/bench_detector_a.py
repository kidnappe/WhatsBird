"""ObserveBird Route A: 检测器 PC 端对比评测（交接文档 docs/handoff/detector_a/HANDOFF-DETECTOR-A.md §3）。

对比对象：
  ssd    — App 现役 SSD-MobileNetV1 tflite（基线，喂法复刻 TfLiteDetector.kt：拉伸 300x300 + 原始 RGB）
  nano   — YOLOX-nano onnx（官方 0.1.1rc0 release，416 输入，letterbox + BGR/255，官方 demo 喂法）
  s      — YOLOX-s   onnx（同上，640 输入）

测试集：
  ① COCO val2017 含 bird GT 的全部图片（有 GT bbox，算召回/误检，小目标 = GT 面积 <1% 画面）
  ② 华北图随机 200 张（无 GT，算检出率，出可视化人工抽检）

产物：data/detector_a/{results/*.json, viz/<model>/*.jpg}
阈值策略：主口径 0.35（=ModelRegistry.MIN_DET_SCORE）；推理结果落盘到 0.2，离线可换阈值复盘。

Usage:
  python bench_detector_a.py            # 全量
  python bench_detector_a.py --limit 30 # 快速冒烟
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

TOOLS = r"E:\code\code tools"
COCO_DIR = os.path.join(TOOLS, "data", "coco")
HUABEI_DIR = os.path.join(TOOLS, "data", "huabei", "images")
OUT_DIR = os.path.join(TOOLS, "data", "detector_a")
SSD_PATH = r"E:\code\observebird\android\app\src\main\assets\models\ssd_mobilenet_v1_1_fp32_300x300.tflite"
YOLOX_DIR = os.path.join(TOOLS, "detectors", "yolox")

SSD_BIRD_ID = 15        # ModelRegistry.BIRD_CLASS_ID（⚠️ 与 labelmap 行号差 1，PC 实测）
SSD_SCORE_TH = 0.35     # ModelRegistry.MIN_DET_SCORE
YOLOX_BIRD_IDX = 14     # COCO 80 类顺序里 bird 的下标（person=0 起）
YOLOX_NMS_IOU = 0.45
STORE_TH = 0.20         # 落盘阈值：低于主口径、够复盘用
SMALL_FRAC = 0.01       # 交接文档定义：小目标 = GT 面积 < 1% 画面
TINY_FRAC = 0.0025      # 更细的桶：<0.25% 画面


# ---------------------------------------------------------------- COCO GT

def load_bird_gt():
    """返回 {file_name: {"wh": (w,h), "boxes": [[x,y,w,h],...]}}，仅含 bird 图片。"""
    with open(os.path.join(COCO_DIR, "annotations", "instances_val2017.json")) as f:
        ann = json.load(f)
    bird_cat = [c["id"] for c in ann["categories"] if c["name"] == "bird"]
    assert len(bird_cat) == 1, bird_cat
    bird_cat = bird_cat[0]
    imgs = {im["id"]: im for im in ann["images"]}
    gt = {}
    for a in ann["annotations"]:
        if a["category_id"] != bird_cat or a.get("iscrowd"):
            continue
        im = imgs[a["image_id"]]
        e = gt.setdefault(im["file_name"], {"wh": (im["width"], im["height"]), "boxes": []})
        e["boxes"].append(a["bbox"])  # COCO xywh 绝对像素
    print(f"COCO val2017 bird 图片数: {len(gt)}, bird 实例总数: {sum(len(v['boxes']) for v in gt.values())}")
    return gt


# ---------------------------------------------------------------- SSD 基线

class SsdRunner:
    """复刻 TfLiteDetector.kt：拉伸 resize 300x300、RGB、uint8 原始字节（或 fp32 等效归一化）。"""

    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter
        self.name = "ssd"
        self.it = Interpreter(model_path=SSD_PATH, num_threads=8)
        self.it.allocate_tensors()
        inp = self.it.get_input_details()[0]
        self.inp_idx = inp["index"]
        self.dtype = inp["dtype"]
        self.quant = inp.get("quantization", (0.0, 0))
        # 四个输出名字同为 TFLite_Detection_PostProcess（:1/:2/:3 后缀），classes/scores
        # 形状都是 [1,10]，名字和形状都区分不了——按位置映射，与 App 的
        # mapOf(0→boxes, 1→classes, 2→scores, 3→num) 实测顺序一致（2026-10-04 检查记录）。
        dets = self.it.get_output_details()
        assert tuple(dets[0]["shape"]) == (1, 10, 4) and tuple(dets[3]["shape"]) == (1,), \
            [tuple(d["shape"]) for d in dets]
        self.out = {"boxes": dets[0]["index"], "classes": dets[1]["index"],
                    "scores": dets[2]["index"], "num": dets[3]["index"]}
        print(f"SSD 输入 dtype={self.dtype.__name__} quant={self.quant}")

    def detect(self, bgr, score_th=SSD_SCORE_TH):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (300, 300)), cv2.COLOR_BGR2RGB)
        if self.dtype == np.uint8:
            self.it.set_tensor(self.inp_idx, rgb[None])
        else:  # fp32 资产：标准 [-1,1]（mean 127.5 std 127.5），与量化语义对齐
            x = (rgb.astype(np.float32) - 127.5) / 127.5
            self.it.set_tensor(self.inp_idx, x[None])
        self.it.invoke()
        boxes = self.it.get_tensor(self.out["boxes"])[0]   # [10,4] ymin,xmin,ymax,xmax 归一化
        classes = self.it.get_tensor(self.out["classes"])[0]
        scores = self.it.get_tensor(self.out["scores"])[0]
        num = int(self.it.get_tensor(self.out["num"])[0])
        dets = []
        for i in range(min(num, len(scores))):
            if int(classes[i]) != SSD_BIRD_ID or scores[i] < score_th:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            dets.append((float(scores[i]), float(xmin) * w, float(ymin) * h,
                         float(xmax) * w, float(ymax) * h))
        return dets


# ---------------------------------------------------------------- YOLOX

def letterbox(bgr, size):
    """官方 master preproc 口径：letterbox(114) + BGR + **原始 0-255，不归一化**。
    （0.1.1rc0 的权重即该口径训练；/255 会让所有类别响应归零——实测）"""
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(bgr, (nw, nh)).astype(np.float32)
    padded = np.full((size, size, 3), 114, dtype=np.float32)
    padded[:nh, :nw] = resized
    return padded.transpose(2, 0, 1)[None], r  # CHW, BGR, 0-255


def nms(dets, iou_th):
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    for d in dets:
        if all(_iou(d, k) < iou_th for k in keep):
            keep.append(d)
    return keep


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[1:]
    bx1, by1, bx2, by2 = b[1:]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class YoloxRunner:
    def __init__(self, name, onnx_file):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(
            os.path.join(YOLOX_DIR, onnx_file), so, providers=["CPUExecutionProvider"])
        i = self.sess.get_inputs()[0]
        self.size = int(i.shape[2])  # 固定方形输入
        self.inp_name = i.name
        self.name = name
        print(f"{name}: 输入 {i.shape}, 输出 {self.sess.get_outputs()[0].shape}")

    def detect(self, bgr, score_th=SSD_SCORE_TH):
        h, w = bgr.shape[:2]
        x, r = letterbox(bgr, self.size)
        out = self.sess.run(None, {self.inp_name: x})[0][0]  # [N,85] cx,cy,w,h,obj,80cls
        bird = out[:, 4] * out[:, 5 + YOLOX_BIRD_IDX]
        m = bird >= score_th
        if not m.any():
            return []
        rows = out[m]
        sc = bird[m]
        dets = []
        for i in range(len(rows)):
            cx, cy, bw, bh = (float(v) for v in rows[i, :4] / r)
            dets.append((float(sc[i]), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        return nms(dets, YOLOX_NMS_IOU)


# ---------------------------------------------------------------- 评测

def match_and_score(dets, gt_boxes, wh, iou_th=0.5):
    """贪心匹配：det 按分数降序，IoU≥th 配未占用的 GT。返回 (n_gt_hit, n_fp, per-gt hit) 。"""
    iw, ih = wh
    matched = [False] * len(gt_boxes)
    n_fp = 0
    for d in sorted(dets, key=lambda d: -d[0]):
        best, bi = iou_th, -1
        for gi, g in enumerate(gt_boxes):
            if matched[gi]:
                continue
            gx, gy, gw, gh = g
            iou = _iou(d, (0, gx, gy, gx + gw, gy + gh))
            if iou > best:
                best, bi = iou, gi
        if bi >= 0:
            matched[bi] = True
        else:
            n_fp += 1
    return sum(matched), n_fp, matched


def size_bucket(area_frac):
    if area_frac < TINY_FRAC:
        return "tiny<0.25%"
    if area_frac < SMALL_FRAC:
        return "small<1%"
    return "large>=1%"


def eval_coco(runner, gt, limit, viz_dir, bird_free_only_viz=False):
    files = sorted(gt.keys())
    if limit:
        random.Random(42).shuffle(files)
        files = files[:limit]
    ths = (SSD_SCORE_TH, 0.25)  # 主口径 0.35 + 对照 0.25

    def fresh_buckets():
        return {"tiny<0.25%": [0, 0], "small<1%": [0, 0], "large>=1%": [0, 0]}

    per = {th: fresh_buckets() for th in ths}
    tot = {th: {"gt": 0, "hit": 0, "fp": 0, "img_det": 0} for th in ths}
    stored = {}
    times = []
    for k, fn in enumerate(files):
        path = os.path.join(COCO_DIR, "val2017", fn)
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        e = gt[fn]
        t0 = time.perf_counter()
        dets = runner.detect(bgr, score_th=STORE_TH)
        times.append(time.perf_counter() - t0)
        stored[fn] = [[round(s, 4)] + [round(v, 4) for v in box] for s, *box in dets]
        for th in ths:
            d35 = [d for d in dets if d[0] >= th]
            hit, fp, matched = match_and_score(d35, e["boxes"], e["wh"])
            tot[th]["gt"] += len(e["boxes"])
            tot[th]["hit"] += hit
            tot[th]["fp"] += fp
            if d35:
                tot[th]["img_det"] += 1
            for gi, g in enumerate(e["boxes"]):
                frac = g[2] * g[3] / (e["wh"][0] * e["wh"][1])
                b = size_bucket(frac)
                per[th][b][1] += 1
                if matched[gi]:
                    per[th][b][0] += 1
        if k < 40 and viz_dir:
            _draw(bgr, [d for d in dets if d[0] >= SSD_SCORE_TH], e, path,
                  os.path.join(viz_dir, os.path.splitext(fn)[0] + ".jpg"))
        if (k + 1) % 50 == 0:
            t = tot[SSD_SCORE_TH]
            print(f"  [{runner.name}] {k + 1}/{len(files)} "
                  f"recall@0.35={t['hit']}/{t['gt']}={t['hit'] / max(t['gt'], 1):.3f} fp={t['fp']}")
    out = {"images": len(files), "gt": tot[SSD_SCORE_TH]["gt"]}
    for th in ths:
        t = tot[th]
        out[f"th{th}"] = {
            "recall": round(t["hit"] / max(t["gt"], 1), 4),
            "hit": t["hit"], "fp": t["fp"],
            "fp_per_img": round(t["fp"] / max(len(files), 1), 3),
            "img_with_det_frac": round(t["img_det"] / max(len(files), 1), 4),
            "by_size": {b: {"hit": h, "total": n, "recall": round(h / max(n, 1), 4)}
                        for b, (h, n) in per[th].items()},
        }
    out["ms_per_img"] = round(float(np.mean(times)) * 1000, 1)
    out["dets"] = stored
    return out


def _draw(bgr, dets, gt_entry, src_path, out_path):
    img = bgr.copy()
    for x, y, w, h in gt_entry["boxes"]:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 2)
    for s, x1, y1, x2, y2 in dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{s:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(out_path, img)


def eval_huabei(runner, n=200, viz_dir=None):
    files = sorted(
        f for f in os.listdir(HUABEI_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    files = sorted(random.Random(42).sample(files, min(n, len(files))))
    ths = (SSD_SCORE_TH, 0.25)
    n_det = {th: 0 for th in ths}
    cnt = {th: 0 for th in ths}
    times = []
    stored = {}
    for k, fn in enumerate(files):
        bgr = cv2.imread(os.path.join(HUABEI_DIR, fn))
        if bgr is None:
            continue
        t0 = time.perf_counter()
        dets = runner.detect(bgr, score_th=STORE_TH)
        times.append(time.perf_counter() - t0)
        stored[fn] = [[round(s, 4)] + [round(v, 4) for v in box] for s, *box in dets]
        for th in ths:
            d = [x for x in dets if x[0] >= th]
            cnt[th] += len(d)
            if d:
                n_det[th] += 1
        if k < 40 and viz_dir:
            _draw(bgr, [d for d in dets if d[0] >= SSD_SCORE_TH], {"boxes": []},
                  os.path.join(HUABEI_DIR, fn),
                  os.path.join(viz_dir, os.path.splitext(fn)[0] + ".jpg"))
        if (k + 1) % 50 == 0:
            print(f"  [{runner.name}] huabei {k + 1}/{len(files)} "
                  f"检出率@0.35={n_det[SSD_SCORE_TH] / (k + 1):.3f}")
    out = {"images": len(files), "dets_per_img": round(cnt[SSD_SCORE_TH] / max(len(files), 1), 3)}
    for th in ths:
        out[f"th{th}"] = {
            "det_rate": round(n_det[th] / max(len(files), 1), 4),
            "dets_per_img": round(cnt[th] / max(len(files), 1), 3),
        }
    out["ms_per_img"] = round(float(np.mean(times)) * 1000, 1)
    out["dets"] = stored
    return out


def calibrate_color(runner, gt, n=40):
    """信息性检查：官方 master preproc 为 BGR 不翻转不归一化。若 RGB 反而显著更高，
    打印警告供人工复核（不中断）。"""
    files = sorted(gt.keys())[:n]
    tot = {"bgr": 0.0, "rgb": 0.0}
    for fn in files:
        bgr = cv2.imread(os.path.join(COCO_DIR, "val2017", fn))
        if bgr is None:
            continue
        for tag, img in (("bgr", bgr), ("rgb", bgr[:, :, ::-1].copy())):
            x, r = letterbox(img, runner.size)
            out = runner.sess.run(None, {runner.inp_name: x})[0][0]
            sc = out[:, 4] * out[:, 5 + YOLOX_BIRD_IDX]
            tot[tag] += float(sc.max()) if len(sc) else 0.0
    winner = "BGR" if tot["bgr"] >= tot["rgb"] else "RGB"
    print(f"色彩检查 [{runner.name}]: BGR={tot['bgr']:.1f} RGB={tot['rgb']:.1f} → {winner}")
    if winner != "BGR":
        print("  ⚠️ RGB 更高，人工复核通道序！")
    return True


# ---------------------------------------------------------------- TFLite YOLO（量化资产评测）

class TfliteYoloRunner:
    """直接喂 tflite（含 int8 量化版）：NHWC float 输入，输出 [1,85,N]（onnx2tf 转置）。"""

    def __init__(self, name, tflite_path):
        from ai_edge_litert.interpreter import Interpreter
        self.name = name
        self.it = Interpreter(model_path=tflite_path, num_threads=8)
        self.it.allocate_tensors()
        inp = self.it.get_input_details()[0]
        self.inp_idx = inp["index"]
        assert inp["dtype"] == np.float32, inp["dtype"]
        self.size = int(inp["shape"][1])
        assert tuple(inp["shape"]) == (1, self.size, self.size, 3), inp["shape"]
        outs = self.it.get_output_details()
        assert len(outs) == 1
        self.out_idx = outs[0]["index"]
        print(f"{name}: 输入 {inp['shape']} {inp['dtype'].__name__} 输出 {outs[0]['shape']}")

    def detect(self, bgr, score_th=SSD_SCORE_TH):
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = round(h * r), round(w * r)
        p = np.full((self.size, self.size, 3), 114, np.float32)
        p[:nh, :nw] = cv2.resize(bgr, (nw, nh)).astype(np.float32)
        self.it.set_tensor(self.inp_idx, p[None])  # NHWC, BGR 0-255
        self.it.invoke()
        out = self.it.get_tensor(self.out_idx)
        c = out.reshape(85, -1).T if out.ndim == 2 else out[0].T if out.shape[1] == 85 else out[0]
        # c: [N,85] = cx_raw,cy_raw,w_raw,h_raw,obj,cls80（decode 前）
        grids, counts = [], []
        for st in (8, 16, 32):
            hw = self.size // st
            gy, gx = np.mgrid[0:hw, 0:hw]
            grids.append(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))
            counts.append(hw * hw)
        n = c.shape[0]
        assert n == sum(counts), (n, counts)
        g = np.concatenate(grids, 0)
        st = np.repeat(np.array((8, 16, 32), np.float32), counts)
        xy = (c[:, :2] + g) * st[:, None]
        wh = np.exp(c[:, 2:4]) * st[:, None]
        bird = c[:, 4] * c[:, 5 + YOLOX_BIRD_IDX]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r  # 原图像素
        m = bird >= score_th
        dets = [(float(bird[i]), float(boxes[i, 0]), float(boxes[i, 1]),
                 float(boxes[i, 2]), float(boxes[i, 3])) for i in np.where(m)[0]]
        return nms(dets, YOLOX_NMS_IOU)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="COCO bird 图限量（冒烟用）")
    ap.add_argument("--models", default="ssd,nano416,s416,s512,s640", help="逗号分隔")
    ap.add_argument("--skip-huabei", action="store_true")
    args = ap.parse_args()
    which = set(args.models.split(","))

    os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
    gt = load_bird_gt()

    runners = {}
    if "ssd" in which:
        runners["ssd"] = SsdRunner()
    if "nano416" in which:
        runners["nano416"] = YoloxRunner("nano416", "yolox_nano_416_dec.onnx")
        calibrate_color(runners["nano416"], gt)
    if "s416" in which:
        runners["s416"] = YoloxRunner("s416", "yolox_s_416_dec.onnx")
        calibrate_color(runners["s416"], gt)
    if "s512" in which:
        runners["s512"] = YoloxRunner("s512", "yolox_s_512_dec.onnx")
    if "s640" in which:
        runners["s640"] = YoloxRunner("s640", "yolox_s_640_dec.onnx")

    summary = {}
    for name, runner in runners.items():
        print(f"\n=== [{name}] COCO bird 子集 ===")
        viz = os.path.join(OUT_DIR, "viz", name)
        os.makedirs(viz, exist_ok=True)
        res = {"coco": eval_coco(runner, gt, args.limit, viz)}
        if not args.skip_huabei:
            print(f"=== [{name}] 华北 200 ===")
            res["huabei"] = eval_huabei(runner, viz_dir=os.path.join(OUT_DIR, "viz", name + "_huabei"))
        summary[name] = res
        with open(os.path.join(OUT_DIR, "results", f"{name}.json"), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    for th in (SSD_SCORE_TH, 0.25):
        print(f"\n===== 汇总 score≥{th} =====")
        hdr = (f"{'model':<9}{'recall':>8}{'tiny<0.25%':>12}{'small<1%':>10}{'large':>9}"
               f"{'FP/图':>7}{'华北检出率':>10}{'ms/图':>8}")
        print(hdr)
        for name, res in summary.items():
            c = res["coco"][f"th{th}"]
            hu = res.get("huabei", {}).get(f"th{th}", {})
            print(f"{name:<9}{c['recall']:>8.3f}"
                  f"{c['by_size']['tiny<0.25%']['recall']:>12.3f}"
                  f"{c['by_size']['small<1%']['recall']:>10.3f}"
                  f"{c['by_size']['large>=1%']['recall']:>9.3f}"
                  f"{c['fp_per_img']:>7.2f}"
                  f"{hu.get('det_rate', float('nan')):>10.3f}"
                  f"{res['coco']['ms_per_img']:>8.1f}")


if __name__ == "__main__":
    sys.exit(main())

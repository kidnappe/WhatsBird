r"""Copy inference models from E:\code\code tools into the Android assets dir.

Large .tflite files are gitignored (android/.gitignore), so after a fresh
clone you must run this once before building the app:

    python tools/copy_models_to_android.py

2026-10-13（APK 瘦身）：七个分类器改从 ``model_float16.tflite`` 取，经
``fp16_io32.convert`` 补 CAST 变成「fp16 权重 + fp32 IO」再落 assets——
端上 ``TfLiteClassifier`` 的 putFloat/FloatArray 一行不用改，体积对半。
逐图对照见 tools/apk_slim/eval_fp16_vs_fp32.py，说明见 android/models/README.md。
fp32 原件不进 APK，存放在 android/models/fp32/（随仓库分发）。
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "apk_slim"))
from fp16_io32 import convert as fp16_io32_convert  # noqa: E402

TOOLS = r"E:\code\code tools"
ASSETS = os.path.join(os.path.dirname(__file__), "..", "android", "app", "src", "main", "assets", "models")

FILES = {
    # 检测器 B（自训 YOLOX-s@640 单类，int8 两步法，2026-10-07；docs/handoff/detector_b/HANDOFF-DETECTOR-B-INT8-SOLVED.md）
    os.path.join(TOOLS, "data", "detector_b", "ckpt_sweep", "tflite", "yolox_bird_640_best", "yolox_bird_640_best_int8.tflite"): "yolox_bird_640_best_int8.tflite",
    # 检测器中档备位（512 ep35 int8，-1.1pp 过闸）：双档产品立项后由分档机制启用
    os.path.join(TOOLS, "data", "detector_b", "ckpt_sweep", "tflite", "yolox_bird_512_ep35", "yolox_bird_512_ep35_int8.tflite"): "yolox_bird_512_ep35_int8.tflite",
    # 分类器五区 rec 系（2026-10-07 五区接入，docs/handoff/root/HANDOFF-APP-MODELS-20261006.md §4-B；
    # labels 文件名去数字——类数从常量读，文件名不再背类数，§5.3）
    os.path.join(TOOLS, "models", "huabei_mnv4_rec", "export", "tflite", "model_float16.tflite"): "birdnet_huabei.tflite",
    os.path.join(TOOLS, "models", "huabei_mnv4_rec", "export", "tflite", "labels.txt"): "labels_huabei.txt",
    os.path.join(TOOLS, "models", "huanan_sw_mnv4_rec", "export", "tflite", "model_float16.tflite"): "birdnet_huanan_sw.tflite",
    os.path.join(TOOLS, "models", "huanan_sw_mnv4_rec", "export", "tflite", "labels.txt"): "labels_huanan_sw.txt",
    os.path.join(TOOLS, "models", "huadong_mnv4_300_rec", "export", "tflite", "model_float16.tflite"): "birdnet_huadong.tflite",
    os.path.join(TOOLS, "models", "huadong_mnv4_300_rec", "export", "tflite", "labels.txt"): "labels_huadong.txt",
    os.path.join(TOOLS, "models", "dongbei_mnv4_rec", "export", "tflite", "model_float16.tflite"): "birdnet_dongbei.tflite",
    os.path.join(TOOLS, "models", "dongbei_mnv4_rec", "export", "tflite", "labels.txt"): "labels_dongbei.txt",
    # 非鸟拒识门（2026-10-07 桌面误检事故的根解；docs/handoff/detector_b/ 下有验收记录）
    os.path.join(TOOLS, "models", "reject_mnv4_v1", "export", "tflite", "model_float16.tflite"): "birdnet_reject.tflite",
    os.path.join(TOOLS, "models", "reject_mnv4_v1", "export", "tflite", "labels.txt"): "labels_reject.txt",
    # 全国兜底 v2（384 类 = 五区并集 244 + 缺口补抓 140，2026-10-08 替换旧 cn373=373 类；
    # 验收 docs/reports/交付报告_national_v2_20261008.md）
    os.path.join(TOOLS, "models", "national_mnv4_v1", "export", "tflite", "model_float16.tflite"): "birdnet_national_v2.tflite",
    os.path.join(TOOLS, "models", "national_mnv4_v1", "export", "tflite", "labels.txt"): "labels_national_v2.txt",
    # 华中 25k 跨区借用版（17k iNat 版 + 8,178 张借入尾部数据），val 重排后 58.54%
    os.path.join(TOOLS, "models", "huazhong_mnv4_25k", "export", "tflite", "model_float16.tflite"): "birdnet_huazhong.tflite",
    os.path.join(TOOLS, "models", "huazhong_mnv4_25k", "export", "tflite", "labels.txt"): "labels_huazhong.txt",
}


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for src, name in FILES.items():
        if not os.path.exists(src):
            print(f"[MISS] {src}")
            continue
        dst = os.path.join(ASSETS, name)
        if name.endswith(".tflite"):
            # 分类器走 fp16：补 CAST 换成 fp32 IO（只改图首/图尾，权重字节原样）
            _, size = fp16_io32_convert(src, dst)
        else:
            shutil.copy2(src, dst)
            size = os.path.getsize(dst)
        print(f"[OK] {name}  {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

# android/models —— 不进 APK 的权重存档

这个目录里的 `.tflite` **不参与 APK 打包**，也不被 App 运行时加载。
它们随仓库分发（Git LFS），用途只有两个：**留档**与**离线复现**。
真正装进 App 的权重在 [`android/app/src/main/assets/models/`](../app/src/main/assets/models/)。

## fp32/ —— 七个分类器的 float32 原件

2026-10-13 APK 瘦身（蓝奏云单文件 100MB 上限，GitHub 下载太慢，118.86MB 装不下）：
App 里换成了**同源的 float16 权重**（fp32 体积对半），fp32 原件搬到这里存档。

| 文件 | 角色 | fp32 | fp16（App 现役） |
|---|---|---|---|
| `birdnet_national_v2.tflite` | 全国兜底，384 类 | 11.36 MB | 5.69 MB |
| `birdnet_dongbei.tflite` | 东北，150 类 | 10.22 MB | 5.12 MB |
| `birdnet_huadong.tflite` | 华东，150 类 | 10.22 MB | 5.12 MB |
| `birdnet_huazhong.tflite` | 华中，150 类 | 10.22 MB | 5.12 MB |
| `birdnet_huabei.tflite` | 华北，148 类 | 10.21 MB | 5.11 MB |
| `birdnet_huanan_sw.tflite` | 华南+西南，148 类 | 10.21 MB | 5.11 MB |
| `birdnet_reject.tflite` | 非鸟拒识门，2 类 | 9.49 MB | 4.76 MB |
| | **合计** | **71.93 MB** | **36.04 MB** |

App 里的 fp16 版是「**fp16 权重 + fp32 输入输出**」：端上 `TfLiteClassifier`
仍然 `ByteBuffer.putFloat` 灌输入、`Array(1){FloatArray(dim)}` 取 logits，
一行 Kotlin 都没改。逐图对照（每模型 1500 张 val，fp32 vs fp16）top-1 一致率
99.4%~100%、top-5 完全一致 97.3%~100%、最大 logit 偏差 0.19。
构建/复现口径见 `tools/apk_slim/`（`fp16_io32.py` 造权重、`eval_fp16_vs_fp32.py` 跑对照、
`smoke_load_models.py` 装载冒烟），对照数据与决策记录见项目内部的瘦身报告。

许可口径不变，见 [`LICENSE-MODELS.md`](../../LICENSE-MODELS.md)。

## offline-detectors/ —— App 不加载的离线检测器

| 文件 | 用途 | 说明 |
|---|---|---|
| `yolox_s_512_int8.tflite` | 退役的 COCO 80 类检测器 | 原资产名 `yolox_s_512.tflite`。2026-10-07 检测器 B 换装后 `ModelRegistry.DETECTOR_ASSET` 指向自训权重，此件只剩「回退」这一层名义价值，代码里已无任何引用，故移出 assets。伪标签打标走的是 PC 侧 `detectors/yolox/yolox_s_512_dec.onnx`，不是这份。 |
| `ssd_mobilenet_v1_1_fp32_300x300.tflite` | A 路线基线 | `tools/bench_detector_a.py` 的 `SSD_PATH`，只在 PC 上跑，从不进 App。 |

两份在数据盘 `E:\code\code tools\detectors\` 都有**逐位相同**的副本，
离线脚本现在直接读那边，本目录只作仓库留档。
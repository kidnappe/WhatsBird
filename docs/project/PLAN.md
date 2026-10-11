# 拾鸟 WhatsBird 开发方案 v2（基于已实测资产）

> 对标：一款设计型实时观鸟 App（参考资料榜 #34，296 评分 5.0，iOS 16+，安装包 287MB）
> 本文只写**已核实的事实**和**基于事实的决策**。推测和待验证项统一放在第六节。
> 最后更新：2026-10

---

## 一、已确认的资产（全部实测过）

### 1.1 已确认到手的资产

| 文件 | 大小 | 实测结果 | 授权 | 定位 |
|---|---|---|---|---|
| `bird-cn-373\model_v1base.pth` | 44.9 MB | **11.2 M 参数**（164 张量），架构 = ConvNeXt（stem + down2/3/4 + conv1–conv4 + norm + head），输入 224×224，输出 **373 类**；README 标top-1 62.16% / top-3 74.87% | MIT仓库 / **数据 NC** | 🔴 **核心资产**，中国鸟图像识别唯一主力 |
| `inat-models\INatVision_Small_2_fact256_8bit.tflite` | 21.17 MB | 507 类。实测内容 =北美常见**植物+昆虫+鸟+真菌+软体动物**混合（`Acer`/`Apis`/`Alligator`…）。作者 373 种中国鸟只命中 **5 种** | MIT | ⚪ **仅管线验证**，产品不用 |
| `inat-models\INatGeomodel_Small_2_8bit.tflite` | 0.64 MB | 1985 空间类。对中国鸟覆盖率同样只有 **5/373 = 1.3%** | MIT | ⚪ **仅管线验证**，产品不用 |
| `inat-models\taxonomy.csv` | 80 KB | 1985 分类单元，含 `leaf_class_id`(0–506) / `spatial_class_id`(0–1984) / `rank_level`(目科属种层级) | MIT | ⚪ 学名规范参考 |

**关键实测结论：iNat 三件套是"北美版 small model"，对中国场景几乎无效。** 唯一有价值的是它证明的技术链路（TFLite + geo model + taxonomy 三件套怎么配合）。

### 1.2 已确认可下载、授权干净（待下载）

| 项目 | 关键指标 | 授权 |
|---|---|---|
| **[birdnet-team/geomodel](https://github.com/birdnet-team/geomodel)** v3.0.4 | **14,082 种**（鸟 **11,157** + 哺乳 1,087 + 昆虫 566 + 两栖 540 + 爬行 11）；输入原始 `(lat, lon, week)`；多任务网络（多标签 BCE + 环境特征回归）；1.8M–36M 参数（默认 scale 0.75 ≈ 3.8M）；FP16 ONNX **7.5 MB**；`convert.py` 可导出 **ONNX / TFLite / TorchScript**；训练数据为 **GBIF occurrence**；附`taxonomy_v0.2-Jun2026.csv`（多语言）+ `gbif_dev.zip` 样例 | 代码 MIT + **模型 Apache 2.0**（独立 `LICENSE-MODELS.md`）✅ |

**官方 README 原文（已核对）**：
> The source code is licensed under the MIT License. Model artifacts—including checkpoints, converted models, and labels—are licensed under the **Apache License 2.0**.
> Acceptable Use Policy "does not add conditions to the Apache 2.0 model license".

🔴 **这是 geo model 问题的合法解** —— 注意 BirdNET 的**声学**模型是 NC，但**地理模型被单独授权成 Apache 2.0**。

### 1.5 检索补充发现（2026-10 第二轮）

#### ✅ 已下载并验证

| 资产 | 实测结果 |
|---|---|
| **`birdnet-team/geomodel` v3.0.4**（已下载） | `..._Global_14K_FP16.tflite` 7.35 MB / `..._INT8.tflite` **4.05 MB** / `..._FP16.onnx` 6.47 MB / `Labels.txt` 14,082 行 / `LICENSE-MODELS.md`。**release 直接提供 TFLite，无需自行转换** |

**🎯 验证项 #1 已通过：geomodel 对我们 373 种中国鸟覆盖 = 373/373 = 100%**
（对比：iNat small geo model 只有 5/373 = 1.3%）
已取得每个种在 geomodel 输出中的索引（如 棕脸鹟莺=11011、八哥=6324），可立即生成 `int[373]` 索引数组。

#### 🔥 关键架构决策：检测 + 裁剪 + 分类 三段式

来源：`wzf19947/Bird-Species-Classification` README 的 A/B 实验数据 + OSEA 论文（均为公开结论，思路可用）

```
用户拍照（鸟占画面一小块 + 大量背景干扰）
   ↓ ① 检测（YOLO / SSD-MobileNet / EfficientDet-D0）
   ↓ ② 目标框外扩 10~20%（避免切掉翅膀/头/尾）
   ↓ ③ crop_resize 224×224
   ↓ ④ 分类模型 → Top-5
   ↓ ⑤ geomodel 重排
```

**该作者的实测对比（训练数据方案）**：

| 训练方案 | 精度 | 原因 |
|---|---|---|
| **① 整图直接训练** | **最高** | 保留完整上下文，模型学到全局/局部/**环境**特征，且无检测框不准的问题 |
| ② 检测后直接 crop | **最低** | 丢失生态上下文（环境、姿态），检测框易切掉翅膀/头/尾 |
| ③ 检测后**外扩**再 crop | 折中 | 主体 + 部分上下文自然过渡，去掉无效背景 |

**最终结论（作者原话）**：使用「检测 → 目标框外扩 → crop_resize → 分类」的 pipeline 推理得分较高。

⚠️ **反直觉但正确**：**训练用整图（含背景），推理用裁剪图。** 两者不矛盾 —— 整图训练让模型学到"水鸟在水边"这类环境先验，裁剪推理消除背景干扰。
✅ **对本项目有利**：整图训练 ConvNeXt 显存吃紧，但换MobileNetV4-small / EfficientNet-Lite0 训整图在 6 GB 显存上完全可行。

参考实现（均为公开方法，代码 license 需单独确认）：
- **`wzf19947/Bird-Species-Classification`**：检测 mAP@0.5 **0.955** / 3.1 ms / 3.71 MB；识别 1486 类 Top-1 **81.3%** / Top-5 **91.2%** / 16.3 ms / 31.8 MB（Axera NPU 上实测）。⚠️ **无 LICENSE、无 Release、数据为 inat2021(NC)** → 代码与权重均不可用于商用，**只取其实验结论**
- **OSEA**（论文）：ResNet34 分类 + SSD-MobileNet 检测 + 地理过滤 + ONNX 量化，明确提供预训练模型，10,000+ 鸟种
- **BirdRecon**（Ecological Informatics 2025）：4 模型软投票集成，525 种 99.6%，导出 TFLite（待查 GitHub/license）
- 检测模型可选项：`fjsuarez/DLOD`（EfficientDet-D0，Apache/无 license 待确认）、LiteRT/Mediapipe 自带检测模型

#### 📊 可商用数据源（补充）

| 数据 | 内容 | 授权 |
|---|---|---|
| **中国鸟类的生活史和生态学特征数据集** | **1445 种中国鸟类 × 17 特征**：体重、体长、嘴峰长、翅长、尾长、跗蹠长、食性、窝卵数、卵大小、**巢址**、**巢的类型**、集群状况、**迁徙状况**、**是否特有种**、**地理分布范围**、**分布省份**。完整度 60.83%–100% | 《生物多样性》2021, 29(9):1149–1153，**DOI 10.17520/biods.2021201** → 数据论文，**引用即可商用** ✅ **已下载 → `tools/dataset_traits/`（见该目录 README）** |
| `tjysdsg/birds` 的 `bird_china_map_labeled.csv` | 物种英文名 / 中文名 / 稀有度 | ✅ 表可用（**事实性名称，非著作权对象**）；❌ 同仓库图片为 Google Images 爬取，**不可用** |

**该数据集直接支撑第三层重排的三个字段**：
- **迁徙状况** → 夏候鸟/冬候鸟/留鸟/旅鸟 → 月份维度过滤 ⚠️ **已实测打折，见下方第 4 条**：R 码可信（七区 222/222），S 码华北 92%，**P 码不可用（0–20%）**
- **分布省份** → 省级地理边界（精度仅次于 geomodel）
- **巢址** → 生境维度（geomodel 完全没有这个维度）

**⬇️ 已下载归档（2026-10-05）**：`tools/dataset_traits/`（xlsx + zip + 导出 CSV，两来源 SHA256 一致）。三条实测要点：

1. **正文里的旧 DOI/数据链接已失效**：`10.24899/do.202109003` 在 DOI 系统查无此号；`dataopen.info` 域名已下线。唯一活路是期刊站的两条 URL，所以本地归档是必需的，别再只存链接。
2. **本数据集没有「保护等级」**：只有迁徙状态（居留型 R/S/W/P/V，多值组合）、是否特有种、分布省份数、分布区代码。保护等级（国家一/二级、IUCN）走 `docs/legal/LICENSES.md` §D 的维基整理名录（中国鸟类名录 10.0，1501 种）。别把两件事混在一个字段里。
3. **拉丁名不能直接 join**：与 huabei 的 146 类交集 134/146，缺的 12 个是 2021 名录 vs GBIF backbone 的分类学改名（`Pica serica`、`Coloeus dauuricus`、`Periparus venustulus`、`Zosterops simplex` 等），接库前要做异名映射。
4. **已与 huabei 的 29,622 条 GBIF 记录做过交叉校验**（详见 `docs/guides/dataset_traits_README.md`「交叉校验结论」）：
   - 语言陷阱：**居留型是「繁殖季身份」，不是「哪季记录多」**；必须用「6–7 月有没有记录」判，用峰值季节比会得出大片假不一致（基线本身就被观测强度带偏：春 37.6%/冬 30.3%/夏 18.3%/秋 13.7%）。
   - 代码可信度：**R 留鸟七区 222/222 全对 → 可当硬约束**；S 华北 92%/华中 100%，但东北 59%/蒙新 67% → 只当软先验；W 只在南方三区可用；**P 旅鸟七区 0–20% 全废**——它把真旅鸟（黄眉柳莺）、冬候鸟（燕雀/红尾斑鸫/普通秋沙鸭/罗纹鸭，繁殖季 0 冬季 0.3–0.9）、夏候鸟（红脚鹬/泽鹬）混成一个桶。**按字面把「迁徙状况」接月份过滤会主动压低冬候鸟旺季权重，比不用更差。**
   - **别拿正在下的 GBIF 语料当物候基准**：`--per-species 300` 截的是 GBIF 默认排序前 300 条而非随机样本，与全量 `facet=month` 的月份占比中位 r 仅 **0.647**，「6–7 月有记录」判定**翻转 13.1%（19/145）**（鸳鸯 0 vs 0.107、震旦鸦雀 0 vs 0.074）。要月份先验就**一物种一次 API 拿全量 facet**（1445 种 ≈10 分钟、不占内存、不触发 §2.3 串行规则），缓存见 `tools/dataset_traits/phenology_cache.json`。
   - 仍须**按全体基线归一化**（全量数据里观鸟季相照样存在：麻雀 5 月 18.3%），否则是在拟合「观鸟人什么时候出门」；且 huabei 语料 29,128/29,622 行同属一个 iNat dataset_key，与训练/评测同源，验证重排增益要留独立区域或非 iNat 来源。

### 1.4 PoC 实测结果（2026-10，已完成）

脚本：`tools/poc_verify.py`（PyTorch / ONNX / TFLite 三方对比 + geomodel 重排）

**✅ 验证项 #2 通过：ConvNeXt → TFLite 数值完全正确**

| 对比 | max\|diff\| | cos 相似度 | argmax |
|---|---|---|---|
| PyTorch vs ONNX Runtime | 1.645e-04 | 1.000000 | 一致 |
| PyTorch vs TFLite | 1.311e-04 | 1.000000 | 一致 |

**导出链路上发现的 4 个坑（全部已解决）**：

| # | 坑 | 解决 |
|---|---|---|
| 1 | 模型有 **4 个 3维 LayerNorm**（`stem.1`=64×56×56、`down2.norm`=64×56×56、`down3.norm`=128×28×28、`down4.norm`=256×14×14），ONNX 里 `axis=(-3,)`；TFLite converter 不支持 | **必须走 `onnx2tf` 中转**（PyTorch 不能直转 TFLite）。`onnx2tf` 会把 LayerNormalization 分解成基础算子 |
| 2 | onnx2tf 输出 **NHWC**（`[1,224,224,3]`），与 PyTorch/ONNX 的 **NCHW**（`[1,3,224,224]`）不同 | Android 端需按tensor 声明的 shape 转置 |
| 3 | 模型输出是 **384 维**，但只有 373 类有标签（后 11 维未使用） | labelmap 只取前 373，重排时按 373 处理 |
| 4 | **geomodel 的 `.tflite` 需要 select-TF-ops（FlexErf），移动端不可用** | 🔴 **geo model 必须用 `.onnx` 版**（ONNX Runtime，MIT）|

**产物体积**：`birdnet_373_float32.tflite` 44.92 MB / `birdnet_373_float16.tflite` 22.49 MB（INT8 未做，待做）

**geomodel 实际规格**（实测）
```
input : ['batch', 3]      float32   ← (lat, lon, week)，week 为 1–48
output: ['batch', 14082]  float32   ← 概率 0.0000–0.9896
覆盖  : 373/373 = 100%
```

**重排实测**（北京 39.90/116.41，第 20 周 ≈ 5月中旬）

| | 纯图像 | geomodel 分 | 重排后 |
|---|---|---|---|
| 喜鹊 *Pica serica* | 74.46% | 0.9869 | **92.89%** |
| 松鸦 *Garrulus glandarius* | 0.21% | **0.4887** ↓ | 0.13% |
| 青藏喜鹊 *Pica bottanensis* | 2.44% | — | 被挤出Top-5 |

geomodel 给出的"北京第 20 周最可能出现的 10 种"= 环颈雉、家燕、麻雀、喜鹊、珠颈斑鸠、山斑鸠、灰喜鹊、戴胜、白头鹎、棕头鸦雀 —— **全部合理**，且把欧洲种松鸦正确降权。**链路成立。**

⚠️ 仍需 ≥200 张标注图才能量化 top-1/top-3 的实际提升幅度（验证 #3 未完成）。

### 1.3 已确认可复制的开源代码

| 项目 | 拿什么 | 授权 |
|---|---|---|
| **inaturalist/vision-camera-plugin-inatvision** (13⭐) | Android侧 TFLite 实时帧推理的 Java 实现 | MIT |
| **inaturalist/iNaturalistAndroid** (232⭐) | iNat 官方 Android App 全源码：记录/图鉴/地图/统计/离线队列 | MIT |
| **inaturalist/inatVisionAPI** (76⭐) | `generate_thresholds.py`（按类置信度阈值）、`geopackager.py` | MIT |
| **inaturalist/inatVisionTraining** (34⭐) | `export_tflite_geomodel.py`、训练脚本 | MIT |
| **inaturalist/SeekReactNative** (205⭐) | 端侧离线识物完整实现 + 模型放置说明 | MIT |
| **inaturalist/iNaturalistAPI** (154⭐) / **inaturalistjs** (44⭐) / **inaturalist-open-data** (157⭐) | API 客户端 + 官方开放数据文档 | MIT |
| **Firefly-ZJ/Bird-Classification** 代码 (`_BirdNet.py` / `_train.py` / `convnext.py`) | 训练与导出流程 | MIT |
| **android/architecture-samples**、**camera-samples** | Room+Hilt+Compose 分层、CameraX 最佳实践 | Apache-2.0 |
| **JetBrains/compose-multiplatform** (19k⭐) | 将来做 iOS 版时 UI 复用 | Apache-2.0 |
| **yoshino-s/birdrecord-cli** | 物种 taxonid 与《中国鸟类分类与分布名录·第四版》(郑四) 对齐 | MIT |
| **CKRainbow/commonBird** (32⭐) | 观鸟记录中心 ↔ eBird 双向转换思路（**仅参考，待核license**） | 待核实 |

### 1.4 确认不可商用

| 项目 | 问题 |
|---|---|
| BirdNET 声学模型（birdnet / BirdNET-Lite / whoBird-TFlite） | **CC BY-NC-SA 4.0** |
| woheller69/whoBIRD (919⭐) | **GPL-3.0** 强传染+ 内含 NC 模型 → 只读架构 |
| iNaturalist 图片 / i_naturalist2018/2021 / Firefly-ZJ 的 22 万张图 | 多为 **CC BY-NC** |
| CUB-200-2011 | Caltech 官网"非商业研究"vs Kaggle CC0，冲突 |
| xeno-canto 音频 | 逐条许可，多数 CC BY-NC |
| fjsuarez/DLOD / ousscher / josafatburmeister / OleksiiDubovyk / ProjectKestrel | 无 license 或待核实 → 只读思路 |

---

## 二、技术栈（已定）

| 层 | 选择 |
|---|---|
| 语言/UI | Kotlin + Compose M3；预留 Compose Multiplatform |
| 架构 | 单Activity + Navigation + MVVM，Hilt |
| 存储 | Room（含 **FTS**：支持"zsq"→朱背啄花鸟）+ DataStore |
| 相机 | CameraX（Preview / ImageCapture / ImageAnalysis） |
| **推理** | **双引擎**：图像模型走 **LiteRT**（TFLite，实测可用）+ geo model 走 **ONNX Runtime**（MIT，官方 TFLite 需 Flex delegate 不可用）|
| 网络 | Ktor + kotlinx.serialization |
| 图片 | Coil 3 |
| 地图 | MapLibre Native（底图用 OSM，需 ODbL 署名） |
| 动画/图表 | Lottie Compose / Vico |
| 后台 | WorkManager（批量识别、备份、同步） |
| 支付 | Google Play Billing |
| 模型训练 | PyTorch (CUDA) → ONNX → TFLite INT8；`tools/model-training/` |

**开发机实测**：Ryzen 7 6800H (8C/16T)、15.2 GB RAM（空闲仅 3.3 GB，需释放）、**RTX 3060 Laptop 6 GB VRAM**、CUDA 13.0、E盘 96.7 GB 空闲（数据全放E 盘）。已装 JDK 17 / Python 3.12/ Git。**缺 Android SDK 和 Gradle**。系统代理 `127.0.0.1:7877`（git 需 `-c http.proxy=` 指定）。

---

## 三、核心架构决策

### 3.1 模型可插拔（授权风险降级为"换文件"）

```kotlin
interface InferenceEngine {
    suspend fun classify(bitmap: Bitmap, geo: GeoContext?): List<Prediction>
}
class LiteRtEngine(bundle: AssetBundle) : InferenceEngine      // Apache-2.0
class OnnxEngine(bundle: AssetBundle) : InferenceEngine// 备选
// ModelRegistry: version → (tflite, labelmap, thresholds)
// 换模型 = 换 assets + 改版本号，业务代码零改动
```
模型走 **Play Asset Delivery按需下发**，避免头部竞品那种 287 MB 首包。识别记录写 `aiModelVersion` 可追溯。

### 3.2 三层重排（核心算法）

```
score = P_图像(373) × P_geomodel(373, lat,lon,week)^α × P_规则(373)^β
```
| 层 | 权重 | 阶段 | 说明 |
|---|---|---|---|
| geomodel | α = 1.0–2.0（val 扫最优） | **P0** | Apache 2.0，全球 14,082 种，48 周粒度，能区分上海/广州 |
| 生境/稀有度 | β = 0.5 | P1 | geomodel 没有的维度 |
| 用户锁定/排除 | 1.0 | P2 | learn from corrections |

**铁律：永不硬删除候选。** 除用户手动排除外，权重永不为 0。
反例：`黑鹤 = 仅黑龙江/江西` → 用户在河北拍到真鸟被硬删 → 用户永远看不到正确答案 → 卸载。
Merlin / iNat / Seek 都不用硬边界。

```kotlin
interface Reranker { val weight: Float; fun apply(i: RerankInput): FloatArray }
class GeoReranker : Reranker { override val weight = 1.0f }        // P0
class HabitatReranker : Reranker { override val weight = 0.5f }    // P1
class UserPreferenceReranker : Reranker { override val weight = 1.0f } // P2
```

### 3.3 推理链路：检测 → 裁剪 → 分类 → 重排

```
CameraX / PhotoPicker
   ↓ ① 检测模型（鸟框 + 外扩 10~20%）
   ↓ ② crop_resize 224×224
   ↓ ③ 分类模型 → Top-5 原始得分
   ↓ ④ geomodel 重排（α = 1.0–2.0）
   ↓ ⑤ 规则表重排（β = 0.5，P1）
   ↓ Top-5 候选 + 置信度 → 用户确认
```
依据见 §1.5 的 A/B 实验数据。**训练用整图（含背景），推理用裁剪图** —— 两者不矛盾。

### 3.4 实时取景识别（三档分层，架构基石）

**竞品对标**：头部竞品的商店描述主打「拍照识鸟 + 批量上传」，未强调取景实时识别。**实时取景是我们的差异化机会**（现场记录 vs 事后整理，两者不冲突，可都做）。

**分层策略**（不可能每帧都跑两个模型）：

| 档位 | 流水线 | 目标帧率 | 说明 |
|---|---|---|---|
| **快速扫视**（默认） | 只跑检测，画框，**不分类** | 10–20 fps | CameraX `ImageAnalysis` + `STRATEGY_KEEP_ONLY_LATEST`；检测到鸟才升级 |
| **识别中** | 检测到鸟 → crop → 分类 + geomodel 重排 | 3–6 fps | Top-5 候选实时刷新 |
| **锁定跟踪** | tracker 跟随（ML Kit Object Detection 的 `ObjectId`），**同一目标不重复分类** | 稳定 | 数量自动累加 →支撑「连续实时观察模式」和 `count` 字段 |

**实测单帧耗时**（Ryzen 7 6800H, 8 线程, `tools/bench_models.py`）：

| 环节 | 耗时 | 可信度 |
|---|---|---|
| geomodel (ONNX, 14082 类) | **0.55 ms** | ✅ 真实值（输入只有 3 个 float，端侧同样亚毫秒） |
| 分类器 BirdNet FP32 | **27.7 ms** | ✅ PyTorch oneDNN，代表「优化良好的 CPU 推理」 |
| 分类器 TFLite FP32 | 1075 ms | ❌ Windows 版 ai_edge_litert 无 XNNPACK，参考内核，**不代表手机** |
| SSD-MobileNet v1.1 检测 | 3068 ms | ❌ 同上 |

⚠️ **帧率上限必须在真机实测**。Android LiteRT 有 XNNPACK / GPU / NNAPI delegate，通常比本机参考内核快 1–2 个数量级。公开参考点：`wzf19947` 在 Axera NPU 上 检测 3.1 ms / 识别 16.3 ms（硬件加速，与手机 CPU 不可直接比较）。

**关键优化手段**（全部 Apache-2.0 / 官方）：
- **CameraX** `setBackpressureStrategy(STRATEGY_KEEP_ONLY_LATEST)` — 不堆积帧
- **ML Kit Object Detection**（Apache-2.0）— `Object` 自带 `ObjectId`，直接用于跨帧跟踪
- **LiteRT GPU delegate / NNAPI delegate** — 释放 CPU
- 分析流用**低分辨率**（640×480），拍照才用全分辨率
- **跳帧**：不必每帧分类（如每 3 帧一次）
- **热控降级**：`PowerManager.getCurrentThermalStatus()` 连续发热时自动降频

**已知问题**
- ⚠️ `onnx2tf` 生成的 **float16 TFLite 无法加载**（全 float16 图；TFLite 只支持 FP16 权重 + FP32 激活）。→ 短期用 FP32（44.92 MB），**必须做 INT8 量化**（目标 ~11 MB，量化后约 10–25 ms）
- ⚠️ onnx2tf 输出 **NHWC**，与 PyTorch/ONNX 的 NCHW 不同 → Android 端需按 tensor 声明转置
- ⚠️ 模型输出 **384 维**但只有 373 类有标签（后 11 维未使用）

### 3.5 交互：Top-5 候选 + 用户确认（不做单选）

图像模型 top-1 只有 62%，单选意味着 1/3 会错。参考 Merlin / Seek / iNat 全部给候选列表。
另配**按类置信度阈值表**（抄 `inatVisionAPI/generate_thresholds.py`）：高置信直接确认，低置信给候选。

---

## 四、功能与里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **W1 PoC** | 下载 geomodel v3.0.4；验证其 11,157 鸟种是否覆盖我们 373 种；装 PyTorch；`model_v1base.pth` → ONNX → TFLite；Python 脚本跑 **Top-5 + geo 重排** | 真机或桌面跑通；**实测重排前后 top-1 提升**；确认 TFLite 导出无算子缺失 |
| **W2 工程骨架** | Compose + Hilt + Room + 导航 +主题；CameraX 预览 + LiteRT 实时推理；geomodel 集成（生成 373 索引映射）；标签表（IOC/郑四合并，含拼音首字母） | 取景实时出 Top-5，geo 重排生效；1500+ 中国鸟种可搜索 |
| **W3–4 核心闭环** | 记录管理（时间/GPS/数量/生境/备注/照片/音频）；图鉴（已解锁/未解锁 + 鸟种详情）；时间线；基础统计 | 拍照→识别→确认→记录→图鉴→统计 全通|
| **W5–6 P0 收尾** | 「加新」羽毛飘落仪式感动画（Lottie + Compose 粒子）；免费额度 + Play Billing；空态/引导/无障碍 | MVP 内测版可跑 |
| **W7–8 P1** | 批量导入（Photo Picker 多选 + WorkManager 队列 + **EXIF 时间/GPS 回填**）；识别改判 + 反馈样本本地累积；地图足迹 heatmap + **敏感物种位置自动模糊**；分层 life list | 100 张批量导入不OOM、不掉进度 |
| **W9–10 P1** | **生境表**（373 种手工整理，一表两用：既做过滤维度又做图鉴文案）；高级统计；备份导出（含"去除位置"开关）；**导出到 eBird CSV / 观鸟记录中心格式** | 导出文件能被 eBird 导入成功 |
| **W11–14 P2** | 听音识鸟（whoBIRD 架构自写 + 自训声学模型）；植物/昆虫识别；Widget；分享长图 | — |

---

## 五、授权红线（写进代码规范）

| 规则 | 说明 |
|---|---|
| 只复制 **MIT / Apache-2.0 / BSD** 的代码 | GPL（whoBIRD）、无 license（DLOD、commonBird）一律只读思路自己重写 |
| **数据集训练数据与模型权重分离** | 权重文件头部必须写明训练数据来源与 license |
| 模型元数据强制字段 | `modelVersion` / `trainDataSource` / `trainDataLicense` / `codeLicense` |
| 设置页固定「数据来源与致谢」 | 署名 IOC（CC BY 4.0）、中国鸟类名录/郑四、Firefly-ZJ 代码 (MIT)、BirdNET geomodel (Apache 2.0)、BirdNET (引用) |
| 声学功能（P2）**禁止**引入 BirdNET 权重 | 只做自训，xeno-canto 仅用 CC0/CC-BY |
| 上架前必须做的替换 | 图像模型换自训（ImageNet 预训练 backbone，**不要从 `model_v1base.pth` 微调**，避免污染传递） |

---

## 六、待验证项（唯一的技术风险）

| # | 问题 | 影响 | 怎么验 | 何时 |
|---|---|---|---|---|
| ~~**1**~~ | ~~geomodel 是否覆盖 373种~~ | — | ✅ **已验证：373/373 = 100%** | **已完成** |
| **2** | **ConvNeXt → TFLite 导出是否干净？**（LayerNorm + 深度可分离卷积可能缺算子） | 可能需要 onnx2tf 或改用端侧原生骨干 | PyTorch → ONNX(opset17) → onnx2tf → TFLite，逐层对齐数值 | W1 |
| **3** | geo 重排对中国的**实际提升幅度** | 决定是否投P1 做精细 geo model | 拿 ≥200 张验证图，对比 top-1/top-3 前后 | W1 |
| 4 | iNat API 按 CC0/CC-BY 过滤后，中国 373 种能拉多少图？ | 决定**商用模型能否自训** | — | **⏸ 已挂起**（2026-10 决策：默认数据可用，不展开；用户提出时再讨论） |
| **5** | 商用6 GB VRAM 训小模型的实际耗时（建议 MobileNetV4-small / EfficientNet-Lite0，**整图训练**） | 决定自训排期 | 用小模型跑 1 epoch 实测 | W3 |
| **6** | birdfyi API 数据来源是否可商用 | 省3–4 周文案工作 | 查其数据出处 | W3 |
| ~~**7**~~ | ~~检测模型选型（YOLO vs SSD-MobileNet vs EfficientDet-D0）在 6 GB 显存下的可行性~~ | ✅ **已决（2026-10-05）**：YOLOX-s@512 int8（Apache-2.0）替换 SSD-MobileNetV1，commit `9d3231f` | **已完成** |

**如果 #4 的结果是"中国鸟 CC0/CC-BY 图片 < 3 万张"**，商用路线需要重新决策（加大众包自采 / 采购授权 / 减少物种数）。这是唯一可能推翻方案的点。

---

## 七、发布合规（个人开发者）

| 项 | 要求 | 备注 |
|---|---|---|
| Google Play 账号 | **$25** 一次性 | 个人账号 |
| **软著** | 国内安卓商店（华为/小米/OPPO/vivo/应用宝）**强制**要求 | 个人可申请，¥300–800，**30–60 工作日** → 建议立刻提交 |
| **App 备案** | 2023 年起中国大陆上线 App 必须完成（工信部），个人可申请，通过分发平台提交 | — |
| Google Play 结算 | 中国大陆个人需**Payoneer** 或境外收单卡 + **W-8BEN** 税务表 + 地址证明 | 资料准备比技术更耗时 |
| 隐私合规 | EXIF GPS = 精确位置，照片 = 个人信息 → 定位可关闭、敏感物种自动模糊、导出可去位置、隐私政策列清 | — |

---

## 八、现在做什么（W1）

|顺序 | 动作 | 产出 |
|---|---|---|
| 1 | 下载 `birdnet-team/geomodel` v3.0.4 release | FP16 ONNX 7.5 MB + labels.txt + LICENSE-MODELS.md |
| 2 | 用 labels.txt 匹配 373 学名 | **验证项 #1** |
| 3 | `pip install torch`（CUDA 版）+ 克隆 Firefly-ZJ 代码 | 环境 |
| 4 | 加载 `model_v1base.pth`，跑几张测试图 | 验证权重可加载 |
| 5 | PyTorch → ONNX → TFLite FP16 | **验证项 #2** |
| 6 | 桌面 PoC：图像 Top-5 + geomodel 重排 | **验证项 #3** |
| 7 | 并行：写 iNat API license 过滤脚本 | **验证项 #4** |
| 8 | 同步：申请软著 | 30–60 工作日启动 |

**W1 结束时能回答一个问题：整个方案成立吗？**
---

## 七、区域划分 v2（2026-10-04 确认）

W1 通过（华北 146 类 top-1 52.50%，重排后 53.81% > 50% 门槛）后，区域从 4 块扩为 **5 块**，
补上原规划缺失的华中（河南/湖北/湖南）。bbox 与参数见 `tools/regions.json`。

| 区域 | bbox（纬min 纬max 经min 经max） | 覆盖 | 状态 |
|---|---|---|---|
| 华北 | 36 43 110 120 | 京津冀晋蒙 | ✅ 完成：29,621 张，300/种补量后 top-1 59.56% |
| 东北 | 39 55 119 135 | 黑吉辽 | **未开始** |
| **华中** | 31 37 105 117 | **豫鄂湘（新增，含河南）** | ✅ 数据到位（9,230 张/145 种）；首训 51.12%，类别平衡重训进行中 |
| 华东 | 27 35 117 123 | 鲁苏皖浙赣闽 | **未开始** |
| 华南+西南 | 20 31 97 117 | 粤桂琼云贵川渝 | ✅ 外包机器完成：28,635 张/138 类，重排后 top-1 **62.07%**（best α=1.0），见 `data/huanan_sw/交付报告` |

单区域命令模板（新机器，`--region` 取值 = regions.json 的键：huabei / dongbei / huazhong / huadong / huanan_sw）：

```bash
python tools/pick_region_species.py --region <region> --n 150 --out species_<region>.csv
python tools/build_dataset_gbif.py --species species_<region>.csv --out data/<region> \
  --per-species 300 --max-pages 6 --workers 12 --api-timeout 60 --bbox <同上> --nice
```

说明（2026-10-04 补）：
- `pick_region_species.py` 的探针点已按各 bbox 取样并与 regions.json 对齐。bbox 是矩形近似，
  与省份注记有出入：**华中框(31-37N)实际=豫+鄂北**（湘、鄂中南落在 huanan_sw 框内）；
  华东框(27-35N)不含鲁北（鲁北在华北框）、闽中南（不在任何框内）。
- 预取缓存 `gbif_prefetch_cache.json` 与 `gbif_availability.csv` 写在各区域自己的 `--out`
  目录内（data/huabei/...），不再共用父目录 —— 可用记录数是按 bbox 查的，跨区共享会串台。
- **UI 解耦约定（2026-10-05，用户确认）**：华南（粤桂琼）与西南（云贵川渝）今后在 UI 层面
  **默认按两个独立大区呈现**——图鉴拆两册底册（`tools/split_huanan_sw_guide.py` 按物种分布点拆：
  华南册 127 种 / 西南册 115 种），相机页区域选择器与 GPS 路由同口径拆侧（框内按省界质心判侧）。
  推理层暂仍共用 138 类 `huanan_sw` 模型（不拆数据、不重训），解耦只发生在展示与记录 region 键上；
  今后新增 UI 功能不要再出现「华南+西南」合并大区。

已知限制：西北（新疆/青海/西藏/甘宁/内蒙古西部）暂未纳入规划，高原/荒漠特有种不在任何
区域模型内；geomodel 重排层为全物种，不受此限制。规划视全量数据表现决定是否增补第六区。

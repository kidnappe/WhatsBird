# 依赖清单：用途 + 商用授权（拾鸟 WhatsBird）

> 标注含义：
> - ✅ **可商用** — 允许用于商业闭源/收费 App
> - ⚠️ **有条件** — 附带义务（署名 / 同协议开源 / 非商业 / 数据来源受限）
> - ❌ **不可商用** — 含 Non-Commercial 条款，用于收费 App 属违约
> - 📖 **只可读思路** — 有传染性或不明确，**不能复制代码进项目**，只能自己重写
>
> 已核实 license 的条目我标了来源；未核实的明确写「待核实」，不要当结论用。

---

## 🔴 红线清单（先看这个）

| 东西 | 问题 | 结论 |
|---|---|---|
| **BirdNET 声学模型**（birdnet-team/birdnet、BirdNET-Lite、whoBird-TFlite 里的 tflite） | **CC BY-NC-SA 4.0**，明文禁止商业用途 | ❌ 收费 App 绝对不能用。P2 声学必须自训或谈 Cornell 授权 |
| **BirdNET geo model**（birdnet-team/geomodel） | ⚠️ **不是红线！** 它被单独授权为 **Apache 2.0**（见下节） | ✅ **可商用**。geo model 问题已合法解决 |
| **iNaturalist 观察图片**（含 Firefly-ZJ 那 22 万张、CUB-200 之外的 iNat 衍生集） | 多数观察者选的是 **CC BY-NC**（部分是 all-rights-reserved）。iNat 因此把**完整模型设为私有** | ❌ 不能直接用来训商用模型。必须换数据源 |
| **woheller69/whoBIRD 源码** | **GPL-3.0**，强传染 | 📖 只可读架构思路。一旦 copy 文件，你的项目被迫 GPL |
| **xeno-canto 音频** | 逐条许可，**多数 CC BY-NC** | ⚠️ 必须按 license 逐条过滤，只能用 CC0/CC-BY |
| **IUCN 红色名录**（`assets/guide/protection.json` 的 `iucn` 字段） | Terms of Use **仅允许非商业使用** | ⚠️ **不是禁用，是需独立声明**：该字段已排除在 Apache-2.0 之外，见 `LICENSE-MODELS.md` §3 |
| **DataV.GeoAtlas 行政区划边界**（`assets/geo/admin_county.ndjson`） | 服务条款**未明确授予再分发许可** | ⚠️ **知情决策：接受风险后随仓库分发**（2026-10-11），见 `LICENSE-MODELS.md` §2 |

---

## A. 推理 SDK / 框架（都是宽松许可，直接用）

| 项目 | 做什么 | 我们用在哪 | 授权 |
|---|---|---|---|
| **google-ai-edge/LiteRT**（原 TensorFlow Lite） | 在 Android 上跑 .tflite 模型的运行时；提供 Task Library、CPU/GPU/NNAPI delegate、INT8 量化、AOT 编译 | 图像分类 + geo model 端侧推理 | Apache-2.0 ✅ |
| **TensorFlow / Model Maker** | 训练与导出工具（PyTorch→TFLite 流程、精度回归） | `tools/model-training` 离线训练 | Apache-2.0 ✅ |
| **google-ai-edge/mediapipe** | 开箱即用的 CV 任务（分类/检测/分割/OCR） | 若不想训模型时的 PoC；后续植物识别 | Apache-2.0 ✅ |
| **microsoft/onnxruntime** | ONNX 模型运行时，CPU 优化好 | 模型来源更广时的备选推理后端 | MIT ✅ |
| **Tencent/ncnn** | 轻量推理框架，ARM 优化好 | 备选，包体更小 | BSD-3 ✅ |
| **alibaba/MNN** | 阿里推理引擎，国产化适配 | 备选 | Apache-2.0 ✅ |
| **berndporr/iirj** | 纯 Java 数字滤波库（带通/高通滤波） | 声学模块的音频预处理 | Apache-2.0 ✅ |
| **srikanth-lingala/zip4j** | Java zip 库 | 备份/解压导出 | Apache-2.0 ✅ |

---

## B. Android 基础库（Apache-2.0 为主，全都能商用）

| 库 | 做什么 |
|---|---|
| **Jetpack Compose / Material 3** | 全部 UI |
| **Navigation Compose** | 页面路由 |
| **Room** | 记录/图鉴数据库（含 FTS 全文搜索） |
| **Hilt**（Dagger） | 依赖注入 |
| **CameraX** | 相机预览、拍照、实时帧分析（识别入口） |
| **WorkManager** | 批量识别、缩略图、备份的可靠后台任务 |
| **DataStore** | 设置项 |
| **androidx.exifinterface** | 读 EXIF 拍摄时间 + GPS（批量导入自动填地点靠它） |
| **FusedLocationProvider**（Play Services） | 定位 |
| **Google Play Billing** | 订阅内购 |
| **Coil 3** | 图片异步加载与缓存 |
| **Ktor** 或 **Retrofit** + kotlinx.serialization | 网络 |
| **Lottie Compose** | 「加新」羽毛飘落动画 |
| **Vico** | 统计图表 |
| **MapLibre Native** | 地图渲染（足迹/热力/聚类） |
| **osmdroid** | 更轻的地图备选 |

**地图底图注意**：MapLibre/osmdroid 本身许可宽松，但**底图瓦片有独立许可**——
- OpenStreetMap 标准瓦片：**ODbL**，必须署名「© OpenStreetMap contributors」并保留在 App 里
- 高德/百度/天地图：需申请 key，各自的使用条款，不可商用场景要确认

---

## C. 模型文件（权重）—— 最需要注意的一类

## C-0. 🟢 地理先验模型（geo model）—— 已合法解决

| 项目 | 内容 | 授权 | 结论 |
|---|---|---|---|
| **[birdnet-team/geomodel](https://github.com/birdnet-team/geomodel)** v3.0.4（2025 新建） | **时空分布模型**：输入原始 `(lat, lon, week)` → 输出 14,082 个物种的出现概率。鸟 **11,157 种**、哺乳 1,087、昆虫 566、两栖 540、爬行 11。多任务网络（多标签 BCE + 环境特征回归作正则），1.8M–36M 参数（默认 scale 0.75 ≈ 3.8M），FP16 ONNX 仅 **7.5 MB**。`convert.py` 可导出 ONNX / TFLite / TorchScript。训练数据为 **GBIF occurrence**（CC0/CC BY 为主）。附多语言 `taxonomy_v0.2-Jun2026.csv` + `gbif_dev.zip` 样例数据 |代码 **MIT** + **模型 Apache 2.0**（独立 `LICENSE-MODELS.md`） | ✅✅ **可商用，这是 geo model 的标准解** |

**为什么这条很重要**：BirdNET 项目把**声学模型**（CC BY-NC-SA）和**地理模型**（Apache 2.0）**分开授权**了。官方 README 原文：

> The source code is licensed under the MIT License. Model artifacts—including checkpoints, converted models, and labels—are licensed under the **Apache License 2.0**.
> Acceptable Use Policy "does not add conditions to the Apache 2.0 model license".

**集成方式**：我们的图像模型 373 类，geomodel 输出 14,082 维→ 用 `labels.txt` 匹配学名，抽出对应 373 种的 373 列 → `score = P_图像(373) × P_geomodel(373)^α`。

**对比 iNat small geo model**（已下载，MIT 但仅 1985 类、北美版）：对作者 373 种中国鸟只覆盖 **5 种 = 1.3%**，对中国场景无效 → **降级为管线验证用**。

## C-1. 🟡 中国鸟图像模型（唯一核心资产，但授权有瑕疵）

| 文件 | 内容 | 授权 | 结论 |
|---|---|---|---|
| `model_v1base.pth`（**44.9 MB，已下载**） | 实测 **11.2 M 参数**，ConvNeXt 架构（stem + down2/3/4 + conv1–conv4 + norm + head），输入 224×224，输出 **373 类**，README 标 top-1 62.16% / top-3 74.87% | 仓库 **MIT**，但数据是 2025/03 从 **iNaturalist** 导出的 22万张 → **图片多为 CC BY-NC** | ⚠️ **MIT 声明有瑕疵**（作者无权把别人的照片授权为 MIT；iNat 自己就把完整模型设为私有）。**PoC / 自用可用；收费 App 不可用** |

**MIT 声明瑕疵的三个证据**：
1. 仓库 LICENSE 只写 "the Software"，未区分代码与权重
2. 作者的 `birdData/Note.md` **完全没有记录任何一张照片的 license**
3. iNaturalist 官方："keeping full species classification models private due to ... all-rights-reserved photos" —— 说明 iNat 认为这类模型无法自由授权

**可以安全拿走的部分**：
- ✅ 代码（`_BirdNet.py` / `_train.py` / `convnext.py`）—— MIT，明确
- ✅ 算法：两阶段微调流程、数据增强、训练配置
- ✅ **物种名录**：373 种学名 + 中文名 —— **学名/中文名是事实性名称，受《国际动物命名规则》约束，不是著作权保护对象**

**不可用**：权重本身（收费场景）、22 万张图片

**上架前的正确做法**：从 **ImageNet预训练 backbone** 重新训（**不要**从 `model_v1base.pth` 微调，避免 NC 污染传递），数据用 iNat API 按 `license_type=CC0/CC-BY` 过滤拉的 + 众包自采。

## C-2. 其余模型文件

| 文件 | 做什么 | 覆盖 | 授权 | 结论 |
|---|---|---|---|---|
| **`INatVision_Small_2_fact256_8bit.tflite`**（21 MB） | iNat 公开的小型图像分类器，输出 **507 个 leaf class** | 主要是北美常见种 + 其他类群 | model-files 仓库 **MIT**，官方公开声明可下载用于端侧测试 | ✅ 可商用，但**只适合当 PoC/技术验证**——507 类里中国鸟覆盖极差，不能当产品模型 |
| **`INatGeomodel_Small_2_8bit.tflite`**（0.64 MB） | 地理先验模型，输入 (lat, lon, week) 输出 **1985 类**在当地的分布概率 | 全球 | 同上 MIT | ✅ 可商用。**算法价值 > 模型本身价值**，我们要用它的做法自训 geo model |
| **`taxonomy.csv`**（80 KB） | 1985 个分类单元，含 `leaf_class_id`（图像模型输出索引）、`spatial_class_id`（geo 模型输出索引）、`iconic_class_id`（类群）、`rank_level`（目/科/属/种层级） | 全球 | MIT | ✅ 可商用 |
| **BirdNET tflite**（声学 6000+ 种） | 听声识鸟 | 全球 | **CC BY-NC-SA 4.0** | ❌ 不可商用 |
| **justinchuby/BirdNET-onnx**（HF） | BirdNET V2.4 转 ONNX + 优化版 | 全球 | HF 上只标 `cc`，**继承自 BirdNET 的 NC** | ❌ 按不可商用对待 |
| **k10z/birdvision-efficientnet-s**（HF） | EfficientNetV2-S，237 种北美鸟，ONNX，top-1 80.7% | 北美 | **CC BY-NC 4.0** | ❌ 不可商用（权重源自 iNat 数据） |
| **mvdmakesthings/coopcam-bird-classifier**（HF） | MobileNetV2 INT8，**仅 612KB**，168 种 | 北美 | MIT（数据源为 CC0） | ✅ 可商用。**价值是体积标杆**：证明端侧小模型完全可行，我们照这个体积目标做 |

**分类清楚**：
- 上面 iNat 三个文件属于**框架/官方公开资产**，MIT，能用
- 下面三个是**第三方基于 NC 数据训练的个人权重**，不能用在收费 App

---

## D. 数据集 / 名录

| 数据 | 做什么 | 授权 | 结论 |
|---|---|---|---|
| **GBIF occurrence** | 全球生物多样性记录（含 lat/lon/时间/物种），**是 birdnet geomodel 的训练数据源**。也可直接用于统计构建查表式 geo prior | 按记录不同：多数 **CC0 / CC BY**，需按 license 过滤 | ✅ **可商用**（按 license 过滤后）。**这是自建 geo 数据的正规来源** |
| **IOC World Bird List**（GBIF 下载） | 全球 11,227 种名录：拉丁名/英文名/科属目/亚种 | **CC BY 4.0** | ✅ 可商用，**必须署名** |
| ~~**中国鸟类列表（维基整理）**~~ | 原计划基于中国鸟类名录 10.0 / 郑四：1501 种 + **IUCN 红色名录等级 + 国家保护动物等级** | 维基 CC BY-SA（表格数据）；原始名录是出版物 | ⚠️ **2026-10-11：本机网络无法访问维基百科**（zh.wikipedia.org / api.wikimedia.org 全线超时），改用等价整理表，见下行 |
| **中国鸟类名录 v10.0 整理表**（GitHub `makenv/bird-list-en`，`v10_0.csv`，1499 种） | **已落地**：图鉴 384 种的 IUCN 等级（380/384）+ 国家重点保护级别（77 种有）。资产 `assets/guide/protection.json`，构建脚本 `tools/bird_protection/build_protection_asset.py`。⚠️ 维基来源本机不可达（2026-10-11，见上） | **按上游各自条款**（2026-10-11 定）：中间仓库无 LICENSE，故**不以它为准**，拆到三个上游分别认定 | 🟡 **部分可商用**——见下方三行逐项结论。原始 CSV 留档 `tools/bird_protection/raw/` |
| ├ **国家重点保护野生动物名录（调整版）** | 行政保护级别（77 种）的**唯一**出处：国家林草局、农业农村部**公告 2021 年第 3 号** | 政府**规范性文件**，公开发布，无许可限制 | ✅ **可商用，无需署名义务**（政府规范性文件；且保护级别属法律事实，不受著作权保护）。⚠️ 名录若日后调整须重新同步 |
| ├ **中国鸟类名录 v10.0**（中国观鸟年报整理） | 分类学：中文名 / 学名 / 异名映射基准 | 公开发表的出版名录 | ✅ **可商用**，按惯例**署名出处**（"中国鸟类名录 v10.0"） |
| └ **IUCN 红色名录** | IUCN 等级代码（CR/EN/VU/NT/LC/NR） | IUCN Red List **Terms of Use：数据免费用于非商业用途；商业使用须与 IUCN 另行签署授权** | ⚠️ **已决策（2026-10-11）：独立声明为非商业。** 等级代码本身是公开事实、非著作权客体，但 `protection.json` 属整理后的数据集。`LICENSE-MODELS.md` §3 已把 `iucn` 字段**单独排除在 Apache-2.0 之外**，只按 IUCN ToU 的非商业条款提供——这样既不丢功能，也不替下游授予 IUCN 没给的权利。商用下游需自行联系 IUCN，或移除该字段（`cn` 与分类学口径无此限制） |
| **中国鸟类的生活史和生态学特征数据集**（王彦平等 2021，《生物多样性》29(9):1149–1153，DOI 10.17520/biods.2021201） | 1445 种 × 17 特征：形态（体重/体长/嘴峰/翅/尾/跗蹠，分雌雄）、食性、窝卵数、卵大小/体积、巢型、巢址、集群、**迁徙状态（居留型）**、特有种、分布区、分布省份数 | 数据论文，**引用即可商用** | ✅ 可商用，**引用王彦平等 (2021) + DOI**。**已归档 `tools/dataset_traits/`**（正文旧链接 dataopen.info / 10.24899 DOI 均已失效）；**不含保护等级**；拉丁名为 2021 名录口径，与 GBIF 有异名差（huabei 146 类命中 134） |
| **中国生物多样性红色名录·鸟类卷**（2021，郑光美主编，科学出版社） | 中国**本土**物种评估 CR/EN/VU | **专著**，无公开机读数据源；dataopen.info / 10.24899 旧 DOI 均已失效 | ❌ **拿不到，已放弃**（2026-10-11 用户定：UI 整行删除，不留「暂缺」占位）。要用得找纸质书或联系出版社 |
| **birdfyi API** | REST API，11251 种，含生境/区域分布/保护状态/**简繁中文 i18n** | 客户端 MIT；**数据来源未核实** | ⚠️ 先查它数据从哪来再商用 |
| **eBird API + 年度 taxonomy.csv** | 物种主数据 + `ebird_regional_extents.csv`（按地区分布，可当免费地理先验数据源） | eBird 有独立使用条款 | ⚠️ **个人非商业免费；商业使用需向 Cornell 申请**，别默认可用 |
| **中国观鸟记录中心（birdreport.cn）** | 物种 taxonid 与郑四对齐；权威中文名口径 | 官方 API 有加密签名 | ⚠️ 只读参考，别在 App 里直接调 |
| **Pl@ntNet observations（GBIF）** | 植物观察 + 图片（已过滤为作者同意 CC 授权的部分） | **CC BY 4.0** | ✅ 可商用（P2 植物识别数据源） |
| **plantnet/PlantNet-300K** | 植物 1081 种 / 30.6 万张；长尾分布处理技巧可借鉴 | 代码 **BSD-2**；数据集 Zenodo 上 CC BY 4.0 | ✅ 可商用 |
| **CUB-200-2011** | 200 种 / 11,788 张，带 bbox + 312 属性 | Kaggle 标 CC0，**但 Caltech 官网是「非商业研究用」** | ⚠️ 冲突。**只用于学术验证，不要进产品训练管线** |
| **Birds 525（gpiosenka, Kaggle）** | 525 种 / 84,635 张 | 未明确声明 | ⚠️ 待核实 |
| **Firefly-ZJ/Bird-Classification** 的数据集 | 中国 373 种 / 22 万张 | **仓库代码 MIT ✅，但数据是 2025/03 从 iNaturalist 导出的 → 图片多为 CC BY-NC** | ⚠️ **代码可商用，数据不可直接商用** |
| **xeno-canto** | 全球鸟声库（P2 训练数据） | 逐条许可，**多数 CC BY-NC** | ⚠️ 必须按 license 过滤 |
| **i_naturalist2018/2021**（TFDS） | 8,142/10,000 种，官方竞赛集 | 数据说明 CC BY 4.0，但图片许可为 CC BY-NC | ❌ 不进商用训练 |

---

## E. 可复制的开源代码 vs 只可读思路 ⚠️ 重点

### 可以直接复制/改写进商业闭源 App（Apache-2.0 / MIT，保留版权头）
| 项目 | 做什么 | 授权 |
|---|---|---|
| **inaturalist/iNaturalistAndroid** | iNat 官方 Android App 全源码（Java）：观察记录、图鉴、地图、统计、离线队列——**业务逻辑最完整的参考** | MIT ✅ |
| **inaturalist/vision-camera-plugin-inatvision** | Android 侧 **TFLite 实时帧推理**的 Java 实现：图像预处理、delegate 选择、结果回调 | MIT ✅ |
| **inaturalist/inatVisionAPI** | geo model 分片打包（`geopackager.py`）、**按类置信度阈值生成**（`generate_thresholds.py`）、评估工具 | MIT ✅ |
| **inaturalist/inatVisionTraining** | 训练 + 导出：`export_tflite_geomodel.py`（geo model → Android TFLite）、`export_coreml_geomodel.py` | MIT ✅ |
| **inaturalist/SeekReactNative** | 端侧离线识物（图+声+AR）完整实现；README 有 Android 模型/taxonomy 放置说明 | MIT ✅ |
| **inaturalist/iNaturalistReactNative** / **INaturalistIOS** | iNat 客户端（React Native / ObjC） | MIT ✅ |
| **inaturalist/iNaturalistAPI** / **inaturalistjs** / **inaturalist-open-data** | iNat API 客户端 + **官方开放数据文档（搞清数据授权先读这个）** | MIT ✅ |
| **android/architecture-samples** | Room + Hilt + Compose 分层官方范式 | Apache-2.0 ✅ |
| **android/camera-samples** / **google/jetpack-camera-app** | CameraX 官方最佳实践；App 的变焦挡位药丸（ZoomBar.kt）与变焦状态管理（ZoomAnimator.kt）改编自其 ZoomBarComponents/ZoomStateManager | Apache-2.0 ✅ |
| **JetBrains/compose-multiplatform**（19k⭐） | 若之后要做 iOS 版，UI 层可复用 | Apache-2.0 ✅ |
| **aregpetrosyan/Jetpack-Compose-Boilerplate** | Compose+Hilt+Room+Retrofit 骨架 | MIT ✅ |
| **BoltUIX/compose-mvi-2025** | Compose + MVI 骨架 | MIT ✅ |
| **yoshino-s/birdrecord-cli** | birdreport.cn CLI；**物种 taxonid 与郑四对齐** | MIT ✅ |

### 只可读思路，不能复制（GPL / 传染 / 无 license）
| 项目 | 做什么 | 问题 | 结论 |
|---|---|---|---|
| **woheller69/whoBIRD**（919⭐） | Android 版 BirdNET 实时听声识鸟 6000+ 种，AudioRecord+iirj+TFLite 流式推理 | **GPL-3.0** + 内含 NC 模型 | 📖 **读架构，自己用 Kotlin 重写**。不要 copy 任何文件 |
| **birdnet-team/birdnet** / **BirdNET-Lite** | 声学模型 + 14082 类地理范围模型 | CC BY-NC-SA 4.0 | 📖 读论文/文档，权重 ❌ |
| **fjsuarez/DLOD** | 最小 CameraX+ImageAnalysis+TFLite+Compose 实时检测 Demo（**架构思路很好**） | **未标注 license** = 版权保留 | 📖 照着结构自己写，别复制文件 |
| **CKRainbow/commonBird**（32⭐） | 观鸟记录中心 ↔ eBird 双向转换 + 鸟种名映射 | 有 LICENSE 文件，**待核实具体条款** | 📖 思路极有价值（证明导出是刚需），核实后再决定 |
| **TaQini/qBird** | birdreport.cn API 逆向（签名/加密） | 无 license | 📖 仅理解协议，别在 App 里用 |
| **sun-jiao/birdreportcn-to-ebird** | 早期转换工具 | 待核实 | 📖 |
| **SanjaySoniLV/ProjectKestrel**（51⭐） | 鸟类照片自动分组、按锐度排序、自动物种标签、剔除废片 | 待核实 | 📖 借鉴「批量导入后自动整理」的功能设计 |
| **ousscher/bird-recognition-dl** | CUB 上 ResNet18+Attention，82% top-1；含 bbox 裁剪预处理 | 无 license | 📖 只看方法 |
| **josafatburmeister/BirdSongIdentification** | Xeno-canto → mel 频谱 → CNN/RNN 完整声学 pipeline | 无 license | 📖 只看方法 |
| **mvrl/bird-audio-detection** | 声学检测 baseline（含 SoundNet 特征 + SVM） | 待核实 | 📖 |
| **OleksiiDubovyk/eBird-taxonomy** | eBird/Clements 2019–2024 年度 taxonomy CSV | 无 license | ⚠️ 数据本身来自 eBird，代码不可复制 |
| **louistrue/birdwatch-ai** | 「图像检测+分类+声学」三合一系统参考 | 待核实 | 📖 |

---

## F. 在线 API（省事但有代价）

| 服务 | 做什么 | 条款 | 建议 |
|---|---|---|---|
| **Pl@ntNet API** | 植物识别（7 万种图库），有官方离线嵌入式模型参考 | 有免费额度，商业使用需申请 key/付费 | P2 植物识别的最快路径 |
| **iNaturalist API** | 观察数据查询/提交 | 官方 API，速率受限；商用需遵守其条款 | 可用于拉数据，但**图片 license 仍是观察者选的** |

---

## G. 我还没核实的（建议你或我下一步查清）

1. `birdfyi` 的数据来源与许可（11251 种含中文，省 3–4 周文案工作，值得查）
2. eBird API 对商业用途的具体条款
3. CUB-200-2011 到底是 CC0 还是仅非商业研究（Kaggle 与 Caltech 官网冲突）
4. `CKRainbow/commonBird`、`SanjaySoniLV/ProjectKestrel` 的具体 license
5. BirdNET 的**商业授权报价**（Cornell / kahst）——这决定 P2 声学是自训还是采购
6. 中国观鸟记录中心对第三方 App 导出/引用其名录的态度（法务上更稳的做法是引用郑四出版物并署名）

---

## H. 分发与变现边界（2026-10-06 调研，非法律意见）

> 结论先行：**「全网多渠道免费分发」本身不构成 NC 意义上的商用**；**免费 App + 爱发电纯自愿
> 赞助链接**通常也仍在安全区。红线始终是数据/权重的 NC 部分，不是分发渠道。

| 问题 | 结论 |
|---|---|
| 全网分发（GitHub/酷安/市场/Play）算商用吗 | 分发渠道本身不改变许可定性。Apache-2.0/MIT 组件闭源免费分发无障碍（履行 NOTICE/版权/署名义务即可，渠道无关） |
| App 内放爱发电链接 | ① CC NC 视角：CC 官方口径看「主要目的是否指向商业利益/金钱对价」——纯自愿、不与任何功能/回报绑定的赞助一般不算 monetary compensation（CC FAQ）；**赞助页只放"支持开发者"裸档位**，爱发电的"会员/回报"档位有被判数字商品销售的风险；② Google Play：100% 归开发者且不授予任何数字内容/服务的款项按 tip 处理，**不强制走 Play 结算**（Payments 政策 FAQ）；禁止的是引导绕过结算买数字商品 |
| 本项目组件对位 | 检测器 YOLOX（Apache-2.0）✅；geomodel（模型件单独 Apache-2.0，§C-0）✅；分类器自训（GBIF CC0/CC-BY + iNat 按 license 过滤，dataset.csv 逐行带 license）✅ 沿用现有过滤口径；**红线不变**：BirdNET 声学模型权重（CC BY-NC-SA）、iNat NC 图片直接训练——免费+赞助模式下这些也**仍然不可用**（NC 与是否收钱无关的解读不能反过来用：NC 数据训出的模型只要"主要目的"沾商业就有风险，稳妥做法是根本不用） |
| 若未来收费 | NC 权重/数据必须先替换或获授权（这是唯一硬门槛）；Play 上数字内容必须走 Play 结算；MIT/Apache 代码闭源收费均可 |

> **2026-10-11 落地**：App 侧按上表执行——设置页「关于」组内一行 `ActionRow("支持拾鸟")`
> （排在"数据位置"那行长说明之前），点击跳 `https://afdian.com/a/zhouxuano6`
> （常量集中在 `ui/SettingsScreen.kt` 的 `AFDIAN_URL`）。
> App 内**不出现任何档位名与价格**，档位文案只活在爱发电页面上；跳转失败走页内提示，不静默。
> ⚠️ 仍未核实的硬前提：**爱发电侧的档位必须只有"纯赞助"档，无会员/回报档**（见上表第 2 行）。

先例：Merlin Bird ID（康奈尔，免费 + 资助/捐赠）、Seek（非营利免费无广告）、LocalSend（App 内
Donate 入口，Play 正常在架）。声学模型若上 P2：BirdNET 权重 NC（免费+捐赠是安全区、收费不行），
可考虑 Apache-2.0 的 Google Perch v2 替代。

## 一句话总结

- **能放心用的**：所有 Android/Compose 库 + LiteRT/ONNX Runtime/ncnn/MNN（Apache/MIT/BSD）+ iNat 官方公开三件套（MIT）+ IOC 名录（CC BY，署名）+ Pl@ntNet GBIF 数据（CC BY）
- **只能借鉴不能抄的**：whoBIRD（GPL）、DLOD/ProjectKestrel/commonBird（无 license 或待核实）
- **绝对不能用的**：BirdNET 声学模型（NC）、iNat 观察图片衍生的模型权重（NC）
- **最大待解问题**：**商用模型需要「可商用训练数据」**。可行路线：① 众包自采（App 内上传即授权协议）② 只用 CC0/CC-BY 子集 ③ 采购 iNat/eBird 媒体授权
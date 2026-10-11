# 模型权重与数据资产许可（拾鸟 WhatsBird）

Copyright 2026 The WhatsBird Authors

本文件覆盖本仓库分发的**训练产物与生成型数据资产**。仓库**源代码**另按
Apache-2.0 授权，全文见 [`LICENSE`](LICENSE)。

本文件与 `LICENSE` 同为 Apache-2.0，单独成文只为把「代码」与「权重 / 数据」的边界写清楚：
权重不是代码，它的可再分发性取决于**训练数据**的授权；数据资产同理。这一层必须显式声明。

---

## 🔴 先看这个：不是全部 Apache-2.0

[`LICENSE`](LICENSE)（Apache-2.0）**不覆盖**下面两项。它们随仓库分发，但附带额外限制，
商用前请自行确认：

| 路径 | 限制 |
|---|---|
| `android/app/src/main/assets/guide/protection.json` 的 `iucn` 字段 | IUCN Red List Terms of Use **仅允许非商业使用**——见 §3 |
| `android/app/src/main/assets/geo/admin_county.ndjson` | 来源条款未明确授予再分发许可。本项目**接受该风险后分发**，但这不构成对下游的授权担保——见 §2 |

**除这两项之外**，下列权重与数据资产按 **Apache License 2.0** 授权：可自由使用、修改、
再分发，**包括商业用途**，条件是保留本文件与 `LICENSE` 中的版权与归属声明。

---

## 覆盖范围

| 路径 | 内容 | 来源 | 许可 |
|---|---|---|---|
| `android/app/src/main/assets/models/*.tflite` | 区域分类器（全国 v2 + 华北 / 华东 / 华中 / 华南西南 / 东北）、拒识门、YOLOX 检测器双档 | 本项目自训 / 上游 Apache-2.0（§1） | ✅ Apache-2.0 |
| `android/models/fp32/*.tflite` | 上面七个分类器的 **float32 原件**（不进 APK，仓库存档，见 `android/models/README.md`） | 同上，与 assets 里那份同源同权重 | ✅ Apache-2.0 |
| `android/models/offline-detectors/*.tflite` | YOLOX-s COCO 80 类 int8（已退役）、SSD-MobileNet v1 基线；**App 运行时不加载** | 上游 Apache-2.0 | ✅ Apache-2.0 |
| `android/app/src/main/assets/models/labels_*.txt` | 物种名录（学名 + 中文名） | 事实性名称，不受著作权保护 | ✅ 无限制 |
| `android/app/src/main/assets/prior/*.json` | 地理先验查表（由 BirdNET geomodel 蒸馏裁剪） | 上游 Apache-2.0 | ✅ Apache-2.0 |
| `android/app/src/main/assets/geo/admin_county.ndjson` | 离线行政区划边界（7.6 MB） | DataV.GeoAtlas | ⚠️ 见 §2 |
| `android/app/src/main/assets/guide/protection.json` | 图鉴保护状态（381 种 × IUCN 等级 + 行政保护级别） | 三源合并 | ⚠️ **部分非商业**，见 §3 |
| `android/app/src/main/assets/guide/` 其余 JSON | 图鉴点亮、性状、稀有度数据 | 本项目整理 | ✅ Apache-2.0 |

## 上游归属

- **YOLOX**（检测器架构）— Apache-2.0，Megvii Inc.
- **LiteRT / TensorFlow**（推理运行时与训练工具链）— Apache-2.0，Google
- **MobileNetV4**（区域分类器与拒识门架构）— Apache-2.0，Google；经 `timm` 提供的
  ImageNet 预训练权重（见 §1）
- **birdnet-team/geomodel v3.0.4**（地理先验模型）— 代码 MIT、**模型 Apache-2.0**
  （上游另有独立的 `LICENSE-MODELS.md`）
- **SSD MobileNet V1**（基线检测器）— Apache-2.0，TensorFlow Object Detection API

## 数据来源声明

分类器训练数据按**逐条 license 过滤**（GBIF CC0 / CC BY 为主，iNaturalist 仅取
CC0 / CC BY 子集），过滤口径见 `docs/legal/LICENSES.md` §H 与
`docs/guides/CLEANING-GUIDE.md`。逐条过滤记录随数据集保留，不在本仓库内。

---

## §1 区域分类器的训练血统 —— ✅ 已结案（2026-10-11）

**结论：六个部署的分类器与 `bird-cn-373/model_v1base.pth` 没有任何权重传递关系，
不存在 NC 污染。**

背景：上游 `model_v1base.pth`（44.9 MB，ConvNeXt，373 类）的 MIT 声明有瑕疵——
其训练数据是 2025/03 从 iNaturalist 导出的 22 万张图，**多为 CC BY-NC**，
作者无权将其授权为 MIT。`docs/project/PLAN.md` §七 因此要求「上架前必须换自训模型，
**不要**从 `model_v1base.pth` 微调，避免污染传递」。

证据（五条，各自独立成立）：

1. **架构不同，物理上无法传递。** 上游是 ConvNeXt / 11.2 M 参数 / 373 类；
   六个部署模型是 **MobileNetV4-conv-small / 2.68 M 参数** / 148–384 类。
   没有对应张量可搬。
2. **训练脚本只从 ImageNet 起。** `tools/train.py` 是全仓唯一建模型处：
   `timm.create_model(name, pretrained=True, num_classes=...)`；该文件**没有任何
   加载外部权重的选项**（无 `load_state_dict`、无 `--ckpt`）。
3. **实际训练命令留档**（`docs/session/SESSION-20261004-1520.md`）：
   `--model mobilenetv4_conv_small.e2400_r224_in1k`——后缀 `e2400_r224_in1k`
   即 timm 官方 **ImageNet-1k** 预训练权重。
4. **labels 行数与模型类别数逐一对上**：148 / 148 / 150 / 150 / 150 / 384 / 2。
5. **真正由该权重导出的 `birdnet_373_float32.tflite`（44.92 MB）已退役**，
   并从 assets 与拷贝脚本移除。全仓对 `model_v1base.pth` 的剩余引用只在 PoC /
   评测脚本，以及"取物种名单"的脚本——后者取的是**学名 + 中文名清单**，
   属事实性名称，不受著作权保护（`docs/legal/LICENSES.md` §C-1）。

⚠️ 唯一无法从仓库判断的：是否在机器上做过**知识蒸馏**（拿该 ConvNeXt 当 teacher）。
仓库内无此记录，`train.py` 亦无相关代码。若确未做过，本项即为干净。

## §2 离线行政区划 —— ⚠️ 知情决策：直接分发（2026-10-11）

`android/app/src/main/assets/geo/admin_county.ndjson`（7.62 MB / 2,874 个区县多边形）
来源为 **DataV.GeoAtlas 公开边界 API**（阿里，免费无 key），坐标是 GCJ-02。

**问题**：DataV.GeoAtlas 提供的是**服务条款**，不是开放许可，**未明确授予再分发权利**。
`docs/legal/LICENSES.md` 因此一直未对这条做授权认定。

**决策**：**接受该风险，随仓库直接分发。** 理由是 7.6 MB 换 clone 即用——
否则离线逆地理功能需要使用者先跑一次 `tools/admin_geo/start_generate.bat`
（全量约 400 请求 / 3 分钟）。

**因此**：本文件**不**对 `admin_county.ndjson` 的可再分发性作任何担保。下游若需
更稳妥的授权路径，可自行运行上述 bat 重新生成，或替换为其他边界数据源
（注意：GADM 明确禁止商用；Natural Earth 为公有领域但无中国区县精度）。

需要重新生成时仍可跑那个 bat——行政区划年度调整时用。

## §3 图鉴保护状态 —— ⚠️ IUCN 部分独立声明

`android/app/src/main/assets/guide/protection.json`（381 种）是**三个上游合并**的产物，
逐项授权如下：

| 字段 | 含义 | 来源 | 结论 |
|---|---|---|---|
| `cn` | 一级 / 二级（《国家重点保护野生动物名录》2021 调整版） | 国家林草局、农业农村部**公告 2021 年第 3 号** | ✅ 可商用，无需署名（政府规范性文件；保护级别属法律事实） |
| `cn` | 三有（《有重要生态、科学、社会价值的陆生野生动物名录》2023） | 国家林草局**公告 2023 年第 17 号** | ✅ 同上 |
| 分类学口径 | 中文名 / 学名 / 异名映射基准 | **中国鸟类名录 v10.0**（中国观鸟年报整理，公开发表的出版名录） | ✅ 可商用，按惯例署名 |
| **`iucn`** | **CR / EN / VU / NT / LC / NR** | **IUCN 红色名录** | ⚠️ **仅非商业使用** |

### IUCN 部分的独立声明

**IUCN Red List Terms of Use 允许数据免费用于非商业用途；商业使用须与 IUCN 另行签署授权。**

- **拾鸟 App 自身的使用属非商业**：应用免费、无广告、无内购、无云端上传。
- **但本仓库源码按 Apache-2.0 发布，而 Apache-2.0 明确允许商用**——若不加声明，
  等于替下游用户授予了一个 IUCN 并未给出的权利。所以 `protection.json` 的 `iucn`
  字段**不适用 Apache-2.0**，只按 IUCN Terms of Use 的非商业条款提供。
- 等级代码本身是公开事实、非著作权客体，但 `protection.json` 是**整理后的数据集**，
  按数据集对待。
- **需要商用的下游用户**，请自行联系 IUCN 取得授权，或移除 `iucn` 字段后使用
  （`cn` 字段与分类学口径均无此限制）。处理记录见
  `docs/reports/GUIDE-PROTECTION-STATUS-20261011.md`。

### 附：两个体系为什么必须分开显示

`iucn` 与 `cn` 是**两套独立体系**，App 里刻意用两套配色，不能互相推导：

- 金雕、白尾海雕：IUCN **无危 LC**，却是**国家一级**保护
- 红头潜鸭、尖尾滨鹬、田鹀：IUCN **易危 VU**，却**未列入**保护名录
- 77 个受保护种里有 **59 个（76%）IUCN 是无危**——所以"无危"绝不能读作"不受保护"

381 种实测分布：IUCN 受威胁（CR/EN/VU/NT）36 种；行政保护级别 一级 17 / 二级 60 /
三有 302。**36 个受威胁种全部同时有行政保护级别**，因此移除 `iucn` 字段不会让任何
受威胁物种失去标记，代价只是丢掉上表那类"全球无危但国家一级"的对比。

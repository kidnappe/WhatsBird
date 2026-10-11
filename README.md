# 拾鸟 WhatsBird

> 拍照认鸟的 Android 应用。**识别全程离线**，照片与观测记录只留在你自己的手机里。

拾鸟面向中国的观鸟场景：举起手机对准鸟，App 告诉你它是什么。没有账号、没有云端上传、
没有广告——飞行模式下也能用。

## 核心能力

| 能力 | 说明 |
|---|---|
| **拍照识别** | 相机实时帧分析 + 三段式推理（检测 → 框选裁剪 → 分类 Top-5），逐帧给出候选 |
| **区域模型** | 全国兜底 + 华北 / 华东 / 华中 / 华南西南 / 东北 五套区域分类器，按 GPS 自动路由，也可手动指定 |
| **地理先验重排** | 用 BirdNET geomodel 的时空分布先验对分类结果加权重排（`score = P_图像 × P_先验^α`） |
| **非鸟拒识门** | 独立的拒识分类器挡住非鸟目标，抑制误检 |
| **多鸟识别** | IoU 跟踪器 + 主目标点选，群鸟场景逐只标框 |
| **观测记录** | 按日流水，可改判 / 加备注 / 删除；CSV 全量导出（SAF，不申请存储权限） |
| **图鉴点亮** | 384 种中国鸟的收集册，含性状档案与保护状态（IUCN + 国家重点保护） |
| **离线逆地理** | 内置行政区划边界，无网也能把经纬度解析到区县 |
| **纸感设计** | 全 app 统一的 Paper Almanac 视觉体系，五套配色 + 夜间派生 |

## 技术栈

- **语言 / UI**：Kotlin + Jetpack Compose（Material 3）
- **推理**：LiteRT（TensorFlow Lite），int8 量化，纯 CPU
- **架构**：Hilt（DI）+ Room（含 FTS4 全文检索）+ DataStore + CameraX
- **检测器**：YOLOX 单类 bird 自训，int8 双档（640 旗舰 / 512 中档，按设备内存自动选择）
- **最低版本**：Android 8.0（API 26）· targetSdk 35 · JDK 17

## 快速开始

### 直接安装（普通用户）

**下载页：<https://kidnappe.github.io/WhatsBird/>**（手机浏览器打开，点一下就能下 APK）。

或者直接到 [Releases](https://github.com/kidnappe/WhatsBird/releases) 下载 APK，传到手机上点开安装即可。

- **要求 64 位设备**（`arm64-v8a`）——2016 年以后的手机基本都是；32 位老机与 x86 模拟器
  请从源码构建（见下）。
- Android 8.0+，不需要 root。模型全在包里，装完就能断网用。
- 文件：`WhatsBird-v1.0-arm64-v8a.apk`，约 119 MB。

### 从源码构建（开发者）

```bash
git clone https://github.com/kidnappe/WhatsBird.git
cd WhatsBird/android

# Windows
gradlew.bat assembleDebug
# macOS / Linux
./gradlew assembleDebug
```

前置条件：JDK 17、Android SDK（compileSdk 35）。

### 模型资产（Git LFS）

模型权重与部分生成型数据资产约 **114MB**（17 个文件），通过 **Git LFS** 分发。
克隆前请先装好 [Git LFS](https://git-lfs.com/)：

```bash
git lfs install
git clone <repo-url>
```

若已克隆但权重是几十字节的指针文件：

```bash
git lfs pull
```

涉及的资产：`assets/models/*.tflite`（区域分类器 / 检测器 / 拒识门）、
`assets/prior/*.json`（地理先验查表）、`assets/geo/*.ndjson`（离线行政区划）。

> ⚠️ **这些资产不是全部 Apache-2.0。** 有两项附带额外限制，商用前请读
> [`LICENSE-MODELS.md`](LICENSE-MODELS.md) 开头那张表：
>
> - `assets/guide/protection.json` 的 `iucn` 字段 —— IUCN Red List Terms of Use
>   **仅允许非商业使用**（`cn` 行政保护级别字段不受此限）
> - `assets/geo/admin_county.ndjson` —— 来源条款未明确授予再分发许可，
>   本项目**接受该风险后分发**，不构成对下游的授权担保
>
> 其余权重与数据资产按 Apache-2.0 分发，含商用。

## 项目结构

```
android/                 Android 应用
  app/src/main/java/com/whatsbird/
    ui/                  Compose 界面（取景 / 日志 / 图鉴 / 设置）
    vision/              推理管线（检测器 / 分类器 / 跟踪 / 先验重排 / 区划索引）
    data/                Room 数据库、DataStore 偏好
    di/                  Hilt 模块
  app/src/main/assets/   模型权重、地理先验、图鉴数据、离线区划
docs/                    项目文档（设计规范 / 指南 / 授权 / 开发方案）
tools/                   数据清洗、模型训练、监控台、看图器等配套工具
site/                    下载页（GitHub Pages：仓库 Settings → Pages → main 分支 /site）
dist/                    交付包（已 gitignore）
```

## 文档

文档在 [`docs/`](docs/) 下，按主题分目录：

| 目录 | 内容 |
|---|---|
| [`docs/design/`](docs/design/) | 设计体系（视觉 / 动效 / 图标） |
| [`docs/guides/`](docs/guides/) | 操作指南（数据清洗、IOC 名单、离线区划、包名迁移） |
| [`docs/legal/`](docs/legal/) | 依赖与素材授权清单 |
| [`docs/project/`](docs/project/) | 开发方案与技术选型 |

几个入口：

- [开发方案](docs/project/PLAN.md) — 技术选型、架构决策、里程碑
- [设计体系](docs/design/DESIGN.md) — 纸感视觉规范
- [依赖授权清单](docs/legal/LICENSES.md) — 商用授权红线（**分发前必读**）
- [数据清洗指南](docs/guides/CLEANING-GUIDE.md) — 训练数据从哪来、怎么过滤
- [离线行政区划指南](docs/guides/OFFLINE-ADMIN-GEO.md) — 无网逆地理怎么做的、踩了哪些坑

## 隐私

- 照片与观测记录**只写本机**（`files/records/` + Room），无任何上传接口。
- 定位用于地理先验与记录坐标，**可关闭**；无定位时先验退化为月度均值。
- 地址解析走**内置离线区划**，不走第三方地理服务。
- 导出的 CSV 由你自己选保存位置，App 不申请存储权限。

## 许可

- **源代码**：[Apache License 2.0](LICENSE)
- **模型权重与数据资产**：大部分同为 Apache-2.0（含商用），**两项例外**——
  `protection.json` 的 IUCN 字段仅非商业、`admin_county.ndjson` 未获明确再分发许可。
  完整声明见 [`LICENSE-MODELS.md`](LICENSE-MODELS.md)
- **第三方组件**：完整清单与商用授权结论见 [`docs/legal/LICENSES.md`](docs/legal/LICENSES.md)

Copyright 2026 The WhatsBird Authors

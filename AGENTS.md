# 拾鸟 WhatsBird — 贡献者与 AI 助手工作约定

这份文件面向**在本仓库里干活的人**（也包括 AI 助手）。项目维护者的本机约定不在
这里——本文只写外部贡献者需要知道的事。

## 项目是什么

一个**全程离线**的 Android 拍照认鸟应用。举起手机对准鸟，App 告诉你它是什么。
没有账号、没有云端上传、没有广告，飞行模式下也能用。

推理分三段：**YOLOX 检测器**找出鸟的位置 → 按比例扩框裁剪 → **区域分类器**给出
Top-5 → 再用 **BirdNET geomodel** 的时空分布先验重排。区域分类器有全国兜底
加华北 / 华东 / 华中 / 华南西南 / 东北五套，按 GPS 自动路由。

## 历史与命名（看代码前先读）

项目在 2026-10-09 更名并改过包名，所以仓库里会同时出现两套名字——**它们是同一个项目**：

| 你看到的 | 是什么 |
|---|---|
| **拾鸟 WhatsBird** / `com.whatsbird` | 现名，现行包名 |
| **ObserveBird** / `com.observebird` | 旧名与旧包名，只应出现在历史记录里 |

- [`docs/guides/MIGRATION-20261009.md`](guides/MIGRATION-20261009.md) **按设计保留旧名**——
  它记录的正是这次改包名，别顺手"改干净"。
- `tools/` 里的脚本带**开发机绝对路径**（`E:` 盘上一个叫 `observebird` 的目录——项目旧名），
  换机器跑要先改成你自己的路径。它是目录名，不是产品名。
- 包名是 `com.whatsbird`，**不要改**（原因见下面「代码约定」）。

## 仓库结构

```
android/                     Android 应用（Gradle 根在 android/，不在仓库根）
  app/src/main/java/com/whatsbird/
    ui/                      Compose 界面（取景 / 日志 / 图鉴 / 设置）
    vision/                  推理管线（检测 / 分类 / 跟踪 / 先验重排 / 区划索引）
    data/                    Room、DataStore
    di/                      Hilt 模块
  app/src/main/assets/       模型权重、地理先验、图鉴数据、离线区划
docs/                       项目文档（见下）
tools/                      配套工具：数据清洗、训练、监控台、看图器
```

## 构建

前置：**JDK 17**、Android SDK（compileSdk 35）。

```bash
cd android

# Windows
gradlew.bat assembleDebug
# macOS / Linux
./gradlew assembleDebug
```

模型权重与部分生成型数据资产走 **Git LFS**。克隆前先装 [Git LFS](https://git-lfs.com/)：

```bash
git lfs install
git clone https://github.com/kidnappe/WhatsBird.git
# 已克隆但拿到的是指针文件：
git lfs pull
```

改完代码至少跑一次编译，别只靠 IDE 报错：

```bash
cd android && ./gradlew compileDebugKotlin
```

### 出正式包要自己签名

仓库里**没有**正式签名密钥（`android/keystore.properties` 与 `.jks` 都不入库）。所以：

- `assembleDebug` 随便跑，得到 debug 签名包；
- `assembleRelease` 得到的是**未签名**包——这是预期行为，不是 bug。

要出可发布的正式包，自己生成一个密钥库，然后在 `android/keystore.properties` 里填：

```properties
storeFile=/绝对路径/your-release.jks
storePassword=…
keyAlias=…
keyPassword=…
```

Gradle 检测到该文件就会自动给 release 签名；文件不存在则退回未签名。

## 代码约定

- Kotlin + Jetpack Compose（Material 3），包名 `com.whatsbird`。**不要改包名**——
  改包名会让用户的 Room 库和照片路径全部失效。
- 最低 Android 8.0（API 26），targetSdk 35，纯 CPU 推理，int8 量化。
- 界面注释里常见的 `docs/design/DESIGN.md §x.y` 是设计依据。**改行为时同步改
  那份文档**，别让注释指向已经不存在的小节。
- 视觉体系是自研的「纸感收集册（Paper Almanac）」，规范全在
  [`docs/design/DESIGN.md`](design/DESIGN.md)：token、五套配色、状态机、
  动效参数、图标清单。加新界面先读它。

## 文档

公开文档按主题分四个目录：

| 目录 | 放什么 |
|---|---|
| [`docs/design/`](design/) | 设计体系（视觉 / 动效 / 图标） |
| [`docs/guides/`](guides/) | 操作指南（数据清洗、IOC 名单、离线区划、包名迁移） |
| [`docs/legal/`](legal/) | 依赖与素材授权清单 |
| [`docs/project/`](project/) | 开发方案与技术选型 |

新增文档放进对应目录，**并在根 `README.md` 的「文档」节登记**。
文档之间的相对链接必须有效——改路径时全仓搜一遍引用。

## 批处理脚本（`.bat`）的三条硬约束

这不是风格偏好，是踩过的坑：

1. **必须 CRLF 行尾**。`cmd.exe` 解析 LF 会吞字符、报「'xxx' 不是内部或外部命令」。
   [`.gitattributes`](../.gitattributes) 已用 `*.bat text eol=crlf` 强制，别绕过它。
2. **注释和 echo 用纯 ASCII**。中文经 GBK / UTF-8 转换容易炸。
3. **开头 `chcp 65001 >nul`**。

另外：**不要用 `sed` 之类批量替换 bat 里的 Windows 路径**——`\t` / `\d` 会被当转义
序列吃掉，整行报废。

## PowerShell 脚本要存成 UTF-8 **带 BOM**

Windows PowerShell 5.1 会把**无 BOM** 的 `.ps1` 按系统代码页（简体中文下是 GBK）解析，
中文字面量在解析期就被毁，报 `Unexpected token`。加了 BOM 才正常。
（PowerShell 7 默认 UTF-8，不受影响——但脚本要给 5.1 用就得带 BOM。）

## 依赖下载（中国大陆网络）

直连经常超时，装包请带镜像参数：

| 用途 | 镜像 |
|---|---|
| pip | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| HuggingFace | `HF_ENDPOINT=https://hf-mirror.com` |
| npm | `--registry=https://registry.npmmirror.com` |

**例外**：`torch` / `torchvision` 的 CUDA 轮子走
`--index-url https://download.pytorch.org/whl/cu128`——这是 Blackwell（sm_120）
必需的，装 cu121 及以下会报 `no kernel image`。

下载类脚本设镜像请写成 `setdefault` / 可被环境变量覆盖的形式，别硬覆盖用户已有的值。

## 资产与许可红线

**这是本项目最容易踩的坑，动手前先读
[`docs/legal/LICENSES.md`](legal/LICENSES.md)。**

- **不要提交 NC（非商用）授权的模型或数据。** 本项目是免费 App，但源码按
  Apache-2.0 发布（允许商用），一旦混入 CC BY-NC 的权重或训练数据，
  整个分发包的授权就说不清了。
- **权重不是代码。** 源码 Apache-2.0 不代表权重也能按 Apache-2.0 分发——
  权重的可再分发性取决于**训练数据**的授权。边界单独声明在
  [`LICENSE-MODELS.md`](../LICENSE-MODELS.md)。
- **仓库不是全部 Apache-2.0。** 有两项资产附带额外限制，别当成 Apache-2.0 用：
  `assets/guide/protection.json` 的 `iucn` 字段（IUCN Terms of Use，**仅非商业**）、
  `assets/geo/admin_county.ndjson`（来源条款未明确授予再分发许可）。
  新增资产遇到同类情况，**照这个模式在 `LICENSE-MODELS.md` 里单独声明**，
  不要默默并进 Apache-2.0。
- **第三方名录 / 论文原件不入库。** 别人的 PDF、Word、Excel 原件（名录、补充材料）
  重新分发属于侵权。仓库里只保留脚本从原件抽出的**事实性小表**
  （物种名 + 保护级别这类，不含原件的表达）。需要原件请按对应工具文档里的
  来源 URL 自行下载。
- 新增任何模型权重或数据集，先确认训练血统和逐条 license，再决定能不能提交。

## 提交前自检

```bash
# 1. 编译过
cd android && ./gradlew compileDebugKotlin

# 2. 文档没断链（示例：根 README 里的相对链接都指得到）
#    改过文档路径就全仓搜一遍旧路径

# 3. 没有 NC 资产混进来
git status --short
```

提交信息用祈使句写清**为什么**改，别只写「fix bug」。

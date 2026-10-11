# 拾鸟 WhatsBird 项目概述 & GBIF 记录清洗指南

> 写给要接手 GBIF 记录清洗工作的读者。任务：用生活史数据集（`tools/dataset_traits/`）
> 反向校验训练数据里 GBIF 来源记录的物种鉴定合理性。
> 日期基准：2026-10-05。相关文档：`docs/project/PLAN.md`（方案）、
> `docs/guides/dataset_traits_README.md`（数据集结构与交叉校验结论）。

---

## 1. 项目概述（30 秒）

对标头部设计型竞品的观鸟记录 App（Android/Kotlin/Compose）。技术核心是**按地理区域分别训练鸟类图像分类模型**（MobileNetV4-conv-small，2.68M 参数），GPS 路由到区域模型，App 端 TFLite 推理 + Top-5 候选人工确认。

| 区域线 | 状态 | val top-1 |
|---|---|---|
| 华北 huabei | ✅ 已接入 App | 59.56% |
| 华中 huazhong | ✅ 57.07%（17k 版）接入 | 51.12→57.07% |
| 华南+西南 huanan_sw | ✅ 已接入 App | 60.67% |
| 华东 huadong | 🔄 自动管线运行中 | - |
| 东北 dongbei | 未启动 | - |

W1 已判定区域化路线成立（>50% 门槛）。训练数据规模：每区域 1.5万~3万张，每类中位 ~90-300 张。

## 2. 数据资产地图

```
<数据工作目录>\<region>\        ← 外部管线目录，不在本仓库
  dataset.csv            ← 清洗对象（主清单）
  gbif_prefetch_cache.json  每物种 GBIF 可用量缓存（key=学名, n=记录数, tk=taxonKey）
  images\*.jpg           图片本体（短边256，文件名在 dataset.csv 的 file 列）
  dataset.csv.*.bak      历史清洗备份
  gbif_build.log.*       下载日志
```

**dataset.csv 列**：`file, scientific_name, common_name, birdnet_index, gbif_key, occurrence_key, lat, lon, year, month, license, dataset_key, attribution`

**数据来源三条腿（清洗必须区分来源！）**：
1. **GBIF**（build_dataset_gbif.py 下载）——主体。社区上传+社区鉴定，**鉴定合理性未知 = 本指南的清洗对象**
2. **iNat 直拉**（fetch_inat_region.py，research-grade）——华中 17k 扩充用
3. **Commons**（fetch_commons_region.py）——部分补充

⚠️ 已知事实：华中 17k 版里有 ~3% CC-BY-NC 和 ~11% 许可未知的行（iNat 不限许可拉的）。**商用合规（NC/未知剔除）和鉴定校验是两件事，都做，分开报告。**华北（纯 GBIF 来源）可直接作为鉴定校验的干净试验田。

## 3. 已有清洗基础设施（先看再用）

| 工具 | 作用 | 状态 |
|---|---|---|
| `tools/clean_dataset.py` | 结构性清洗：属级 key 污染回查/删类、跨种共享 observation 删除（防泄漏），原地重写+自动 .bak | ✅ 已跑过两轮 |
| `tools/audit_taxon_keys.py` | 预取缓存 taxonKey 等级审计（GENUS 级检出） | ✅ |
| `tools/build_dataset_gbif.py` | 下载器，已内置 taxon_key rank 校验 | ✅ |

**已抓到的结构性污染案例**（理解你要防的错误的量级）：GBIF `species/match?strict=false` 在种名失配时静默退化属级 key，曾把苍鹭/大白鹭记录灌进 `Ardea coromanda` 类，累计删过 511 行。rank 校验堵住了这种形态；**你要做的是查堵住之后剩下的"key 正确但鉴定内容错误"的记录——至今未量化，这是本任务的核心目标。**

## 4. 清洗任务定义

**用生活史数据集（文献口径）反向校验 GBIF 记录的物种鉴定**：书说不可能出现在此地/此季的鸟却有一堆记录 → 可疑。数据集结构、区间字符串解析（`~`/`-`/`～` 混用）、双行表头处理见 `docs/guides/dataset_traits_README.md`；物候交叉校验方法学（为什么必须用"6-7 月繁殖窗口 + 观测强度基线归一"而不是"峰值季节"）README 已论证过，照其结论执行。

### 规则集（v1，宁漏勿冤）

| 规则 | 判据 | 处置 |
|---|---|---|
| R1 地理越界 | 记录 lat/lon 所在省 ∉ 数据集分布省份，**且** 该种非 V（迷鸟）码 | 标记 suspect，高置信 |
| R2 物候冲突 | 纯 W 冬候鸟的记录 ≥30% 落在 6-7 月（或纯 S 反季），按 README 基线归一化后判定 | 标记 suspect |
| R3 名录外 | scientific_name 不在 1445 正名单且无别名可映射 | 先过 §5 别名映射，映射失败的才报告 |
| **豁免** | 迁徙状态含 P 的种**不做 R2**（README 已证 P 码对 GBIF 物候仅 5% 可信）；含 V 或分布省份数 ≤3 的种**不做 R1 删除建议**（真迷鸟风险） | — |

## 5. 执行顺序（硬顺序，不可并行跳步）

1. **别名映射表**：产出 `tools/dataset_traits/species_name_map.csv`（our_sci → dataset_sci，映射依据：改名/拼写差异）。已知华北清单 12 例（Pica serica、Spilopelia chinensis、Zosterops simplex、Chloris sinica 等）。**映射率 <95% 就停下来报告，不要带病进规则引擎。**
2. **只报不删**跑华北（29,621 行，纯 GBIF，最干净的试验田）：输出 `data/huabei/clean_audit_<日期>.csv`（file/sci/month/lat/lon/触发规则/数据集依据）+ 汇总报告（各规则命中率、按物种聚合的 top 可疑名单）。
3. **报检率，人做决定**。参考阈值：可疑率 <1% → 仅归档；1~3% → 删高置信（R1+R2 双触发）保留迷鸟嫌疑；>3% → 停，重新评估数据源策略。**不许自作主张删训练数据。**
4. 抽检支持：可疑记录的图片在 `images/` 里按 file 列可直接打开，按物种分组整理给人工验收。
5. 华东数据落盘后（同管线产出），复制步骤 2 跑第二轮——检测率跨区一致才谈固化为管线前置（`auto_region_pipeline.py` 的 clean 步骤之后加一步，**改动前先问**）。

## 6. 红线与坑（都踩过，别再踩）

- **训练/管线进程可能在跑**——清洗期间别 kill 训练进程、别写 `models/`。
- 清洗若获批执行：规则**同时作用于全量 dataset.csv**（train/val 由 `train.py` 的 grouped_split 自动重切），**禁止只清 train**——同源同规则才有可比性。
- 冻结验证集：重训对比前，先用当前 dataset.csv 按 `train.py --seed 0` 的 split 逻辑导出固定 val 观察集存盘；新旧模型都在冻结集上评测，否则分数差异掺"换题"效应。
- 迷鸟 ≠ 错误：删错一条迷鸟记录的代价可能大于删十条错标签（稀有种本来就缺图）。宁可标记。
- Windows 环境：`PYTHONIOENCODING=utf-8` 必设（GBK 控制台遇 emoji 会崩进程）；E 盘 NVMe 但和训练共用，大 IO 错峰。
- 数据集是 2021 口径：分布漂移（鹭类北扩等）会制造假 R1——所以报告而非删除。
- **本指南针对鉴定合理性；iNat 扩充里的 NC/未知许可行是并行的合规问题**（华中 ~14%），报告里单列一节，别和鉴定混淆。

## 7. 交付物清单

1. `species_name_map.csv` + 映射率报告
2. `data/huabei/clean_audit_*.csv` + 检测率汇总（按规则/物种/来源分解）
3. NC/许可合规审计报告（华中，单列）
4. 决策建议书（按 §5.3 阈值给结论，等人类批准）

基线数字备查：华北 59.56% / 华中 17k 57.07% / 华南西南 60.67%（val top-1，清洗后重训用同配置 `--epochs 30 --model mobilenetv4_conv_small.e2400_r224_in1k`，不加任何 balance 参数——类平衡已证伪）。

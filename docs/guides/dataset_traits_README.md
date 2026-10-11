# 中国鸟类的生活史和生态学特征数据集（本地归档）

**用途**：拾鸟 WhatsBird 第三层重排（物种特征先验）的数据源之一，对应 `docs/project/PLAN.md` §1.3「可商用数据源」。

**当前消费方（2026-10-10）**：
1. **图鉴详情页「物种档案」卡片** —— 已落地，把本表的 15 项性状显示给用户（点亮/未点亮都显示）。
   覆盖 381/384 种；生成器在 `tools/guide_traits/`。
2. 第三层重排（物种特征先验）—— 规划中。

## 出处

| 项 | 值 |
|---|---|
| 标题 | 中国鸟类的生活史和生态学特征数据集 / A dataset on the life-history and ecological traits of Chinese birds |
| 作者 | 王彦平, 宋云枫, 钟雨茜, 陈传武, 赵郁豪, 曾頔, 吴亦如, 丁平 |
| 期刊 | 《生物多样性》Biodiversity Science, 2021, 29(9): 1149–1153 |
| DOI | **10.17520/biods.2021201**（cstr 32101.14.biods.2021201） |
| 下载时间 | 2026-10-05（本机） |
| 授权口径 | 数据论文，**引用即可商用** ✅（见 `docs/legal/LICENSES.md` §D） |

论文正文里写的旧数据链接已失效，别再照抄：

- `http://doi.org/10.24899/do.202109003` → **DOI Not Found**（该前缀未在 DOI 系统注册）
- `http://dataopen.info/home/datafile/index/id/222` → **DNS 解析失败，站点已下线**

## 文件

| 文件 | 说明 |
|---|---|
| `supplFile_art_20220117115032.xlsx` | **主文件**（344,619 B）。期刊「补充材料」页下载：`https://www.biodiversity-science.net/fileup/1005-0094/SUPPL/supplFile_art_20220117115032.xlsx` |
| `2021201.zip` | 期刊「数据链接」里的打包版（339,799 B）：`https://www.biodiversity-science.net/fileup/1005-0094/DATA/2021201.zip`，解包后为 `Chinesebirdsdata.xlsx` |
| `china_bird_traits.csv` | 由主文件导出的 UTF-8-SIG CSV（表头合并成 `一级|二级`），便于直接读 |

两个来源的文件 **SHA256 完全一致**：`3B2765766300779E73A4C097B77E3E4BD4489302E081EAA8707B6C0DF4F3624D`
→ 互为校验，内容可信。

## 结构（实测）

- 单 sheet `Chinesebirdsdata`，范围 `A1:AD1476`；**第 1–2 行是双行表头**，数据从第 3 行开始
- **1,445 行物种，拉丁名 1,445 个唯一值，中文名 1,445 个唯一值**（与论文口径一致；表尾有少量空行，末两列 AD/AC 为空列）
- 17 个特征：体重、体长、嘴峰、翅长、尾长、跗蹠长（6 项均分雌雄 → 12 列）、食性、窝卵数、卵大小、卵体积、巢型、巢址、集群状况、迁徙状态、是否特有种、分布区及其迁徙状态、分布省份数

⚠️ **数值列是区间字符串**，不是数字：体重 `325~430`、窝卵数 `1-2`/`1～2`、卵大小 `102-108*61-63`、省份数 `34`。
用之前需要写解析器（范围 → min/max，缺失为 `NA`）。分隔符混用 `~`、`-`、`～`。

⚠️ **本数据集不含「保护等级」**。它给的是：迁徙状态（= 居留型：R 留鸟 / S 夏候鸟 / W 冬候鸟 / P 旅鸟 / V 迷鸟，且是多值组合如 `RSWP`）、是否特有种、分布省份数、分布区代码（`IA(P),IIA(P),…`）。
保护等级（国家一/二级、IUCN）在 `docs/legal/LICENSES.md` §D 的「中国鸟类列表（维基整理，基于中国鸟类名录 10.0，1501 种）」里 —— 另取。

## 与本项目物种清单的对接

以华北区域的 `dataset.csv`（外部管线目录，不在本仓库）146 个拉丁名为例：**134 命中 / 12 未命中**。
未命中的 12 个基本是分类学改名（2021 名录 vs GBIF backbone），例如 `Pica serica`、`Coloeus dauuricus`、`Periparus venustulus`、`Suthora webbiana`、`Zosterops simplex`、`Spilopelia chinensis` 等 —— 接库时必须做**异名映射**，不能直接 join 拉丁名。

## 交叉校验结论（2026-10-05，用 huabei 已下的 29,622 条 GBIF 记录做的）

**先说方法上的关键点**：中国鸟类学里 留鸟R/夏候鸟S/冬候鸟W/旅鸟P 说的是**在该区的繁殖季身份**，
不是「哪一季记录最多」。而 GBIF 记录的季节分布被观测强度主导（全体基线 春 37.6% / 冬 30.3% / 夏 18.3% / 秋 13.7%）。
所以判别量必须取**繁殖窗口(6–7月)有没有记录**，并把基线归一化；用「峰值季节」去比会得出大片假不一致。

### 1. 迁徙状态代码的可信度（逐动物地理区，vs GBIF 实测物候型）

| 区 | R 留鸟 | S 夏候鸟 | W 冬候鸟 | P 旅鸟 |
|---|---|---|---|---|
| I 东北 | **26/26** | 38/64 (59%) | 0/1 | 0/13 |
| II 华北 | **33/33** | 22/24 (92%) | 0/1 | **2/37 (5%)** |
| III 蒙新 | **33/33** | 28/42 (67%) | 2/2 | 0/15 |
| IV 青藏 | **31/31** | 19/26 (73%) | 5/6 | 2/30 |
| V 西南 | **38/38** | 14/16 | 21/29 (72%) | 3/14 |
| VI 华中 | **34/34** | 8/8 | 29/37 (78%) | 2/18 |
| VII 华南 | **27/27** | 1/1 | 27/47 (57%) | 0/7 |

- **R（留鸟）222/222 全对，七个区无一例外 → 这个码可以放心当硬约束用**（永远不要因为月份把留鸟过滤掉）。
- **S（夏候鸟）在华北/华中最可信（92%/100%），在东北/蒙新只有 59%/67%** → 只能当软先验。
- **W（冬候鸟）在南方三区可用（57%–78%），在北方几乎无样本** —— 合理，冬候鸟本来就是南方概念。
- **P（旅鸟）在七个区全是 0–20% → 这个码不可用**，原因见下。
- union（全国口径）代码：单值只占 835/1445 = 57.8%，其余是 `RSWP` 这种多值组合，**月份维度上等于没信息**。

### 2. 「P 旅鸟」是个垃圾桶，直接接月份过滤会反向伤效果

华北 bbox 内被区划 II 标为 P 的 37 种里，有一批是**繁殖季 6–7 月零记录、冬季占比却很高**的典型冬候鸟：

```
燕雀 Fringilla montifringilla    union=SWP    繁殖季 0.000  冬季 0.310
红尾斑鸫 Turdus naumanni          union=WP     繁殖季 0.000  冬季 0.933
普通秋沙鸭 Mergus merganser       union=RSWPV  繁殖季 0.000  冬季 0.457
斑头秋沙鸭 Mergellus albellus     union=SWPV   繁殖季 0.000  冬季 0.511
水鹨 Anthus spinoletta           union=RWPV   繁殖季 0.000  冬季 0.489
锡嘴雀 Coccothraustes coccothraustes union=RWP 繁殖季 0.000  冬季 0.456
红胁蓝尾鸲 Tarsiger cyanurus      union=SWP    繁殖季 0.000  冬季 0.333
罗纹鸭 Mareca falcata            union=SWP    繁殖季 0.000  冬季 0.275
```
反向的也有：`红脚鹬`/`泽鹬`/`反嘴鹬` 标 P 但繁殖季占比 0.42–0.52（其实是夏候鸟）。
同时 `黄眉柳莺`（繁殖季 0 / 冬季 0）是**真旅鸟** —— 所以不是整个 P 码错，是 P 把三类混在一起了。
**结论：`docs/project/PLAN.md` §1.3 里「迁徙状况 → 月份维度过滤」这条按字面实现会主动压低冬候鸟在旺季的权重，比不用还差。**

### 3. 更大的坑：别拿正在下的这份 GBIF 语料当物候基准

对 145 个物种比较「本项目截断语料（每物种 min(300,可用)）」vs「GBIF 全量 `facet=month`」的 12 维月份占比：

- 中位 **r = 0.647**，p10 = 0.150，90 分位 = 0.919；**r ≥ 0.9 只有 20/145，r < 0.7 有 91/145**
- 「6–7 月有记录」这个判定的**翻转率 13.1%（19/145）**，大样本种也翻：
  `鸳鸯 Aix galericulata` 语料 0.000 vs 全量 0.107（r=0.234）、`震旦鸦雀` 0.000 vs 0.074（r=−0.108）
- 语料 n=90–390 vs 全量 n=1,275–37,301（`大山雀` 37,301 → 390）——截断丢掉的正是物候信号

**原因**：`--per-species 300 --max-pages 6` 截的是 GBIF 默认排序的前 300 条，不是随机抽样，
月份构成被排序而非物候决定。**所以「一月一次请求」直接从 GBIF 拿全量 facet，比用这份语料既便宜又准**：

```
https://api.gbif.org/v1/occurrence/search?taxonKey=<key>&decimalLatitude=36,43&decimalLongitude=110,120&facet=month&facetLimit=12&limit=0&hasCoordinate=true
```
1445 种 ≈ 10 分钟、无图片、内存可忽略 → 不触发「训练与下载串行」的既有约束。
本目录 `phenology_cache.json` 已存 huabei 145 种的全量月份直方图。

⚠️ **仍然必须按全体基线归一化**：观鸟/采集季节性在**全量数据里同样存在**（麻雀 5 月 18.3%、苍鹭 5 月 16.2%），
不归一化就是在拟合「观鸟人什么时候出门」。

⚠️ **同源循环风险**：huabei 语料 **32,512/33,214** 行来自**同一个 dataset_key（iNaturalist research-grade）**，
年份 2024–2026。而本项目的训练图片也来自 GBIF/iNat、评测集也是 iNat 系。
> 2026-10-09 更新：原文写的 29,128/29,622 是 **10-05 rec 补量前**的基线；
> rec 补量后 `data/huabei/dataset.csv` 已涨到 33,214 行。**下面 §交叉校验的所有比例
> （占全体的百分比、物候翻转 13.1% 等）都是按 29,622 算的，换基数需重算。**
在这份数据上拟合月份先验、又在这份数据上评估重排增益，数字会虚高。至少要留一个区或一个非 iNat 来源做验证。

### 4. 哪些维度根本没法靠 GBIF 交叉校验

- **体重/体长/食性/窝卵数/卵/巢型/巢址**：GBIF 没有性状字段 → 无从校验，也**没法替代**。这是这份数据集的独有价值。
- **分布省份数**：可以校（坐标→省界需要 shapefile），但 GBIF 采样偏差使它**系统性偏低**，只能当「假阳性探测器」
  （比如声称分布 30 省却在任一区都 0 记录），不能当计数基准。
- **金翅雀 Chloris sinica** 是已知的反例（华北 bbox 内 0 记录）→ **缺失不是证据**，不一致只能进人工复核队列，永不自动改码。

## 复现命令

```powershell
python tools/dataset_traits/dump_dataset.py tools/dataset_traits/supplFile_art_20220117115032.xlsx tools/dataset_traits/china_bird_traits.csv
python tools/dataset_traits/check_values.py   # 抽样取值格式 + 与 huabei 清单交集
python tools/dataset_traits/inspect_dataset.py <xlsx>   # 原始表头/前后几行
# 交叉校验（需先有区域 dataset.csv）
python tools/dataset_traits/crosscheck_migrant_status.py <xlsx> "<区域>/dataset.csv" II   # 峰值季节口径（会误导，见上）
python tools/dataset_traits/crosscheck_v2.py          <xlsx> "<区域>/dataset.csv" II   # 繁殖窗口口径（推荐）
python tools/dataset_traits/crosscheck_regions.py     <xlsx> "<区域>/dataset.csv"      # 七区命中率矩阵
python tools/dataset_traits/phenology_vs_api.py                                        # 截断语料 vs GBIF 全量 facet
```

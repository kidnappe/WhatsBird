# IOC 名字对照表 · 制作指导文档

> 面向"另起一个窗口继续做"的场景。**核心步骤我已经全部实测验证过**，实测数字写在文中，
> 照做即可。
>
> 建立日期：2026-10-05 晚 ｜ 数据源：IOC World Bird List **v15.2**（2026-04-13）

---

## 0. 一句话目标

**以 IOC 世界鸟类名录为唯一权威，把"我们的名字"映射到 IOC 的有效名 + GBIF 的 taxonKey。**
它同时解决三件事：① 名字对不上 ② geomodel 索引缺失 ③ 扩池的分类学混乱。

---

## 1. 为什么需要它（问题本质）

同一个种在不同系统里有不同名字，而且**学名的唯一性只对"某一本名录、某一时刻"成立**。
分三类，别混：

| 现象 | 真实例子 | 后果 |
|---|---|---|
| **属被拆/并**（同一只鸟换姓） | `Charadrius dubius` → `Thinornis dubius`；`Ixobrychus eurhythmus` → `Botaurus eurhythmus` | 名字对不上 → 取不到图 / 选种被跳过 / 重排记 0 分 |
| **种被拆分/合并**（"有几个种"变了） | `Bubulcus ibis` → `Ardea coromanda`；`Saxicola torquatus` → `maurus` + `stejnegeri` | 更难：两套名录各有一批名字参赛 |
| **拼写/词尾** | `Phylloscopus sibilatrix` → `sibillatrix`；`Accipiter trivirgatus` → `Lophispiza trivirgata` | 机械可解 |

**关键事实（已实测）**：
- **IOC 用新分类学** —— `Thinornis dubius` ✓、`Antigone vipio` ✓、`Chloris sinica` ✓、`Ardea coromanda` ✓
- **GBIF 用旧名** —— `Charadrius dubius` ✓、`Grus vipio` ✓、`Carduelis sinica` ✓、`Bubulcus ibis` ✓
- **geomodel 和 373 池子跟 IOC 一边**（都是新名）
- **生活史数据集（2021 口径）跟 GBIF 一边**（旧名）

→ 所以 IOC 不只是"另一本名录"，它是**和 geomodel 口径一致的那一本**，这才是它的价值。

---

## 2. 数据源（已下载到本地，列已验证）

存放目录：`tools/ioc/`（xlsx 不入库，按下表 URL 自行下载）

| 文件 | 大小 | 内容 | 用途 |
|---|---|---|---|
| `Multiling_IOC_15.2.xlsx` | 5.2 MB | sheet `List`：**11,228 行 × 50 列** | ⭐ **主力** |
| `master_ioc_list_v15.2.xlsx` | 2.0 MB | sheet `Master`：33,716 行（**含亚种**） | 需要分布范围/亚种时用 |

**下载地址（幂等，直接重下即可）**

```
https://worldbirdnames.org/master_ioc_list_v15.2.xlsx
https://worldbirdnames.org/Multiling%20IOC%2015.2.xlsx
```

其他可选（未下载，需要时再说）：
```
https://www.worldbirdnames.org/IOC_Names_File_Plus-15.2.xlsx       # Life List+
https://www.worldbirdnames.org/IOC_15.2_vs_other_lists.xlsx        # 与 Clements 等世界名录的对照（6 MB）
http://www.worldbirdnames.org/master_ioc-names_xml.14.2.xml        # XML（只到 v14.2）
```

### `Multiling_IOC_15.2.xlsx` 的实际列（第 1 行是表头）

```
seq | Order | Family | IOC_15.2 | English | Catalan | Chinese | Chinese (Traditional) | Croatian | Czech | Danish | Dutch | Esperanto | Finnish | ...（共 50 列，多语言）
```

**要用的四列**：
- **`IOC_15.2`** = 当前有效学名（如 `Thinornis dubius`）
- **`Chinese`** = 中文名（如 `金眶鸻`）← **桥接的关键**
- **`English`** = 英文名
- `Order` / `Family` = 目/科（可用于"同目同科"这类交叉验证）

**实测覆盖**：11,227 个种级条目，其中 **11,204 个有中文名**；不同中文名 11,204 个。

### `master_ioc_list_v15.2.xlsx` 的实际列（**第 4 行**才是表头！）

```
行1-3 是标题/统计，第 4 行: Infraclass | Parvclass | Order | Family (Scientific) |
Family (English) | Genus | Species (Scientific) | Subspecies | Ext. | Authority |
Species (English) | Breeding Range | Breeding Range-Subregion | Nonbreeding Range | ...
```
⚠️ 行 1–3 不是表头，直接 `next(rows)` 会取到标题行 —— **必须跳过 3 行**。

### 授权

- worldbirdnames.org 页脚写 **CC BY 3.0 Unported**（Attribution）
- GBIF 上那份 IOC checklist（DOI `10.15468/anfqn6`）标 **CC BY 4.0**
- **两者都是"署名即可"**，商用 OK。文档里记录版本（v15.2 / 2026-04-13）和来源即可满足署名。
  ⚠️ 与现在那份 **NC 血统的 373 池子**不是一回事 —— 这是换 IOC 的附带收益。

---

## 3. 核心策略：**中文名桥**（已实测，99.0% 可解）

不要靠学名字符串硬猜。正确顺序：

```
我们的名字 ──┬─ ① 学名在 IOC 里精确存在 ────────────→ 直接用（IOC_15.2 就是有效名）
             ├─ ② 学名不在，但中文名命中 IOC.Chinese ──→ 用 IOC_15.2
             └─ ③ 都不中 → 列出来人工处理（**绝不猜**）
                                    ↓
                    IOC 有效名 → GBIF species/match → taxonKey
                                    ↓
                    验证：rank==SPECIES 且 class==Aves
```

### 实测基线（用现有 `species_name_map.csv` 的 1,494 个名字）

| 匹配方式 | 数量 |
|---|---|
| ① 学名精确命中 IOC | **1,337** |
| ② **学名不中、中文名命中** | **142** ← 桥的价值 |
| ③ 两者都不中 | **15** |
| **合计可解** | **1,479 / 1,494 = 99.0%** |

**⚠️ 重要**：② 那 142 个里，**中文名在 IOC 里没有一个多义**（0 个多重匹配）—— 所以中文名桥是**无歧义**的，可以直接自动采纳。

### 附带收益：解决 geomodel 索引缺失（实测）

```
原本 149 个名字没有 geomodel 索引
  经 IOC 后，133 个能在 geomodel 的 14,082 个标签里命中
  仍找不到: 16
  且这 133 个【全部】是靠 IOC 的新名命中的（用原名一个都命不中）
```

样例（我们的旧名 → IOC 新名 → geomodel 里就叫这个）：

```
Accipiter trivirgatus → Lophospiza trivirgata      Bubo blakistoni  → Ketupa blakistoni
Accipiter gentilis    → Astur gentilis             Bubulcus ibis    → Ardea coromanda
Babax koslowi         → Pterorhinus koslowi        Anser cygnoid    → Anser cygnoides
Accipiter badius      → Tachyspiza badia           Brachypodius atriceps → Microtarsus melanocephalos
```

→ **扩池的前置障碍（149 个无索引）直接从 149 降到 16。**

---

## 4. 实施步骤

### 步骤 1：加载 IOC，建两个索引

```python
import collections, openpyxl
p = "tools/ioc/Multiling_IOC_15.2.xlsx"   # 相对仓库根
wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
it = wb["List"].iter_rows(values_only=True)
hdr = [str(c or "").strip() for c in next(it)]
i_sci, i_zh, i_en = hdr.index("IOC_15.2"), hdr.index("Chinese"), hdr.index("English")
i_ord, i_fam = hdr.index("Order"), hdr.index("Family")

sci2ioc = {}                      # 小写学名 -> (有效名, 中文, 英文, 目, 科)
zh2ioc = {}                       # 中文名   -> 有效名
for r in it:
    if not r or not r[i_sci]:
        continue
    s = str(r[i_sci]).strip(); z = str(r[i_zh] or "").strip()
    sci2ioc[s.lower()] = (s, z, str(r[i_en] or "").strip(),
                          str(r[i_ord] or "").strip(), str(r[i_fam] or "").strip())
    if z and z not in zh2ioc:     # 实测无多义，出现冲突要记下来人工看
        zh2ioc[z] = s
wb.close()
```

### 步骤 2：解析我们的名字

```python
def to_ioc(name, zh):
    """→ (ioc_valid_name, how) 或 (None, None)"""
    if name.lower() in sci2ioc:
        return sci2ioc[name.lower()][0], "exact"
    if zh and zh in zh2ioc:
        return zh2ioc[zh], "via-zh"
    return None, None
```

输入就是我们现有的 `tools/dataset_traits/species_name_map.csv`（`query_name` + `zh` 两列）。
**不要重新从 xlsx 抽名字** —— 那 1,494 个名字已经汇总好了，包含 traits1445 + 373 池子的并集。

### 步骤 3：IOC 有效名 → GBIF key

**必须复用已有的解析器，不要另写一套**（它已经处理了下面 §5 的所有坑）：

```
tools/dataset_traits/build_name_map.py
    · match_one()          —— EXACT 或「FUZZY 且规范名相同」，且 rank==SPECIES
    · is_aves_species()    —— 只看 class==Aves + rank==SPECIES，**不看 status**
    · search_full_name()   —— 认非 backbone 的 key（金翅雀 key=6101015）
    · ep_variants()        —— 拉丁词尾/双写辅音变体
    · name_overrides.csv   —— 人工确认的映射（阶梯 0，最高优先级）
    · resolve()            —— 上面这些组成的阶梯
```

**建议做法**：给 `resolve()` 加一个**阶梯 0.5**（在 override 之后、match 之前）：
拿去 IOC 查到的有效名再走一遍 `match_one()`；命中就用它，并在 `resolved_by` 里写 `via-ioc`。
这样 IOC 表成了新的主桥，而原来的 5 级阶梯退化成兜底。

### 步骤 4：产出

| 文件 | 内容 |
|---|---|
| `tools/ioc/ioc_map.csv` | 人看：`our_name, zh, ioc_valid_name, how(exact/via-zh/none), gbif_name, gbif_key, gbif_cn_records, in_geomodel, geomodel_index, note` |
| `tools/ioc/ioc_map.json` | 机器用：`{our_name: {ioc, gbif_name, gbif_key, geomodel_index}}` |
| `tools/ioc/ioc_unresolved.csv` | ① ② 都不中的（预计 **15** 个）+ 中文名多义的（预计 **0** 个），**列出来给人看** |

进度、缓存、并发都照抄 `build_name_map.py` 的做法（每 50 条落盘、8 线程、`--retry-failed`）。

---

## 5. 必须遵守的坑（都是今天真踩过的）

| # | 坑 | 正确做法 |
|---|---|---|
| 1 | **网络失败被记成"这个分类单元无效"** | 抓取失败写 **`FETCH_FAILED`**，绝不写 `NONE`。这条已经删掉了华南 6 个种的行（4,138 条可用记录），图片还在盘上 |
| 2 | **"没问到" ≠ "不存在"** | 任何"查不到"都要能区分"网络失败"和"确实没有"。`audit_taxon_keys.py` + `clean_dataset.py` 已改成 fail-safe |
| 3 | **种加词会撞车** | 实测：纯"同种加词"匹配 geomodel 得到 133 个"命中"，绝大多数是**昆虫/蛙/蝙蝠/植物/别的鸟**（`Dicaeum agile`→螽斯、`Garrulax chinensis`→青蛙、`Turdinus crispifrons`→植物）。所以**必须**加：在 IOC 里精确存在 + GBIF 确认鸟纲的种 |
| 4 | **不能要求 `status == ACCEPTED`** | 会漏掉 `Chloris sinica` key=6101015（status=None，却有 35,263 条中国记录）。只要求 `class==Aves` + `rank==SPECIES` |
| 5 | **`rank == SPECIES` 绝不能放宽** | 用属级 key 会把整个属的记录标成同一个种名 → 两个种并进一类。华北 3 个种就是这样（缓存里 `tk=` 属级、`n=` 整个属的记录数 1551/287/241） |
| 6 | **GBIF 搜索要完整词** | `q=dubi` 无结果，`q=dubius` 才有 |
| 7 | **别按"记录最多"选候选** | `dubius` 会让红眉朱雀(3577) 排在金眶鸻(209) 前面；`sinensis` 会让白头鹎(101089) 排第一 → 会选中**别的鸟**。要同属约束 + 中文俗名验证 |
| 8 | **`master_ioc_list` 的表头在第 4 行** | 前 3 行是标题/统计 |
| 9 | **一次只能跑一个下载** | GBIF 限流 + 15GB 内存约束。并发查询会得到 `SSL UNEXPECTED_EOF`（我今天就这样触发过） |
| 10 | **改数据前冻结验证集** | 唯独**从未训过**的区（华东、东北）不需要 |

---

## 6. 验收标准

| 项 | 标准 | 实测基线 |
|---|---|---|
| 覆盖率 | ≥ 97% 的我们的名字能拿到 IOC 有效名 | **99.0%**（1,479/1,494） |
| 反查一致性 | 每个 IOC 有效名在 GBIF 都能 `match` 到 `rank=SPECIES` + `class=Aves` | 待测 |
| 中文名多义 | 必须为 0，否则逐条人工看 | **0** |
| geomodel 索引 | 补齐数 ≥ 120 | **133**（原缺 149，剩 16） |
| 未解清单 | 必须产出文件，不许静默丢弃 | 预计 15 + 16（两者可能有重叠） |
| 抽样人工核对 | 随机 30 条，人工确认真鸟 | — |

**自检脚本必须做的三件事**（照 `render_test.js` 的思路，把它做成可复跑的测试）：
1. `len(ioc_map) + len(unresolved) == 输入的 1,494`
2. 每条 `ioc_valid_name` 都在 `sci2ioc` 里
3. 每条有 `gbif_key` 的行，`/species/{key}` 都返回 `rank=SPECIES` + `class=Aves`

---

## 7. 不要做什么

- ❌ **不要让本地小模型（`qwen2.5:local7b`）拍板**。它只能在"① ② 都不中"的十几条上**提议**，
  而且提议必须过三道验证（IOC 里精确存在 / GBIF 确认鸟纲的种 / 中文俗名一致）。
  实测纯相似度匹配的准确率极低（见 §5 第 3 条）。
- ❌ **不要动** `dataset.csv` / `progress.json` / `gbif_prefetch_cache.json` / images/。
  这个任务只产出"名字映射表"。
- ❌ **不要为了 IOC 去重下数据**。映射表是纯查表工作。
- ❌ **不要扩池**。等 IOC 表做完再谈 —— 现在扩池等于把两套分类学混在一起参赛（见下）。

### 为什么"扩池"要等 IOC

实测：华北旧名单有 `Saxicola stejnegeri`（黑喉石䳭），扩池后**两个石䳭名都没进** —— 因为
扩池把另一套名录的分类学意见也拉进来参赛，`Saxicola maurus`（黑喉石䳭）和
`Saxicola stejnegeri`（东亚石䳭）**各自的名次都掉到 150 名之外**，结果丢了一只鸟。

**先统一名录（本任务），再谈扩池。** 扩池的收益原本就小（实测每区只换 1~11 个名额，
合计 26/750 = 3.5%，影响约 1% 的观测），加上这个下行风险就不划算了。

---

## 8. 相关文件与实测数字（省得重查）

### 已有产出

| 文件 | 说明 |
|---|---|
| `tools/dataset_traits/species_name_map.csv` | **1,494 个名字**的现有对照（解析路径、中国记录数、geomodel 索引、候选） |
| `tools/dataset_traits/species_name_map.json` | 同上，机器用（含 `excluded` 标记） |
| `tools/dataset_traits/name_overrides.csv` | 人工确认的 3 条映射 + 1 条排除（海南画眉，GBIF 只有亚种） |
| `tools/dataset_traits/build_name_map.py` | 解析器本体（要复用它） |
| `tools/ioc/` | **IOC 原始文件（已下载）** |
| `docs/project/PLAN.md` | 技术方案与数据源选型 |

### 关键数字

```
IOC v15.2              11,227 个种级条目，11,204 个有中文名
我们名单               1,494 个（traits1445 + 373 池子的并集，去重后）
学名精确命中 IOC        1,337
中文名桥命中            142（零多义）
两者都不中              15
现有 geomodel 命中      1,345 / 1,494（90.0%）
经 IOC 后预计 geomodel  1,345 + 133 = 1,478（98.9%）
```

### 环境

- Python：需装 openpyxl（实测 3.1.5 可用），**不需要 pandas**
- 必须 `$env:PYTHONIOENCODING='utf-8'`（本机默认 cp936/gbk，否则中文乱码）
- PowerShell 脚本必须存成 **UTF-8 带 BOM**（否则按 GBK 读 → 引号错位 → 整个脚本解析失败）

---

## 9. 做完之后能解锁什么

| # | 事项 | 依赖 IOC 的什么 |
|---|---|---|
| 1 | **修好现有 6 类名字问题** | IOC 有效名 |
| 2 | **补齐 133 个 geomodel 索引** | IOC 的新名（原 149 缺 → 剩 16） |
| 3 | **统一分类学口径，然后才谈扩池** | 删掉 IOC 不认的名字 / 合并同种异名 |
| 4 | **授权干净**（替掉 NC 血统的池子） | IOC 是 CC BY |
| 5 | **图鉴/rare 表的中文名与 IUCN** | IOC 的 `Chinese` 列（IUCN 列 IOC 没有，另找 GBIF 的 `iucnRedListCategory`） |
| 6 | `regen_labels_guide.py`（防类别表错位） | 需要稳定的中文名来源 |

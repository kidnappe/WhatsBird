package com.whatsbird.vision

import android.content.Context

/**
 * 模型注册表：所有推理资产与区域路由的常量。
 * 5 区域策略（PLAN §七）：GPS 落在某区域 bbox 内 → 区域模型；否则回退全国兜底 v2（384 类）。
 */
object ModelRegistry {
    // 检测器双档自动分档（旗舰走 640-best，其余走 512-ep35；阈值见 DETECTOR_TIER_MIN_BYTES 注释）：
    // 512 ep35 int8 已过闸（整体 -1.1pp），small 桶 -2.2pp 为该尺寸量化固有敏感度
    // （含鸟校准重量化实测无效），双档上线按此口径接受。
    // selftest_bird.jpg PC 基准：640=0.8776/dets=1，512=0.7663/dets=1（logcat 比对用）。
    //
    // ⚠️ 2026-10-10 补：上面那两个数是「512 家族 fp32 → int8」的对比，
    //   **不是「中档 vs 旗舰」**。低内存设备实际拿到的是中档，代价如下
    //   （落盘 data/detector_b/results/rescore_th045_20261009.json，th0.45 行）：
    //     旗舰 int8-640-best   recall 0.492 / tiny 0.254 / small 0.719 / FP 0.952
    //     中档 int8-512-ep35   recall 0.443 / tiny 0.177 / small 0.686 / FP 0.832
    //     差                   recall −4.9pp / **tiny −7.7pp（相对 −30%）** / small −3.3pp
    //   tiny 正是 10x 小个体麻雀那一档，也正是升分辨率重抓（C）想治的弱项。
    //   分档门槛本身是既定决定，此处只把**代价**记清楚，别再被「−1.1pp」误导成差距很小。
    const val DETECTOR_ASSET = "models/yolox_bird_640_best_int8.tflite"
    const val DETECTOR_ASSET_MID = "models/yolox_bird_512_ep35_int8.tflite"
    const val MIN_DET_SCORE = 0.45f

    // ⚠️ 2026-10-10 阈值由 12 GiB 改为 9 GiB，依据是**真机实测**（不是推理）：
    //   小米15（**标称 12GB**）：/proc/meminfo MemTotal = 11,388,316 kB
    //   Android totalMem      = 11,661,635,584 B = **10.86 GiB**（= 11.66 GB 十进制）
    //   原阈值 12 GiB → 差 1.14 GiB，**标称 12GB 的机器被判成"低内存"走中档**。
    //   而中档 tiny 0.177 vs 旗舰 0.254（相对 −30%），tiny 正是 10x 小个体麻雀那一档。
    //
    //   取 9 GiB 的依据（MemTotal ≈ 标称 × 0.95~0.97，各家 ROM 保留比例略有差异）：
    //     标称  8 GB → MemTotal ≈ 6.9~7.0 GiB  → 低于 9 → 中档 ✅
    //     标称 12 GB → MemTotal ≈ 10.8~10.9 GiB → 高于 9 → 旗舰 ✅（本机实测 10.86）
    //     标称 16 GB → MemTotal ≈ 14.4~14.6 GiB → 高于 9 → 旗舰 ✅
    //   9 GiB 落在 8 与 12 的正中间，两侧余量各约 22%，对保留比例差异不敏感。
    //   ⚠️ 只有 12GB 那一行是实测；8/16GB 两行是按比例推算，等有真机再校。
    //   ⚠️ `totalMem` 不是物理内存，别拿它当"手机多少G"对外说（见 YoloDetector 日志注释）。
    const val DETECTOR_TIER_MIN_BYTES = 9L * 1024 * 1024 * 1024

    /** 档位名（设置页三选一）。 */
    const val TIER_AUTO = "auto"
    const val TIER_FLAG = "flag"   // 旗舰 yolox_bird_640_best_int8（79ms/图）
    const val TIER_MID = "mid"     // 中档 yolox_bird_512_ep35_int8（48ms/图）

    /** 按设备 RAM 选检测档位（auto 模式）。 */
    fun detectorAsset(totalMemBytes: Long): String =
        if (totalMemBytes >= DETECTOR_TIER_MIN_BYTES) DETECTOR_ASSET else DETECTOR_ASSET_MID

    /** 按档位偏好选资产；[tier] 不是 flag/mid 时退回 auto 的 RAM 判定。 */
    fun detectorAssetFor(tier: String, totalMemBytes: Long): String = when (tier) {
        TIER_FLAG -> DETECTOR_ASSET
        TIER_MID -> DETECTOR_ASSET_MID
        else -> detectorAsset(totalMemBytes)
    }
    // 显示门（幽灵框防线③，用户提案）：轨迹均分达此值才画框——低分漂移检测一闪一灭
    // 不再上屏；相机逻辑（追焦/自动放大）仍吃未过门的原始检测，真鸟充能期间分数上来框自然出现
    const val DISPLAY_MIN_SCORE = 0.55f
    // 裁剪口径对齐（2026-10-07 家燕案例归因 → tools/eval_crop_regime.py 扫描定值）：
    // 五区模型按整图训练，紧裁剪推理是分布外——加大外扩单调恢复，本值
    // （=评测脚本 frac2.0，裁剪宽=5×框宽、鸟占 ~20%≈GBIF 构图）。
    // 换算：device ratio = 2× 评测 frac。取 4.0 不取 6.0（评测集多为单鸟图，
    // 裁剪越大越多鸟入镜风险评测看不到，4.0 是留了安全余量的膝点）。
    //
    // ⚠️ 2026-10-10 已修正本注释的依据（值一个字未动）：
    //   原注「top1 只有 23~39%」「三区追平整图口径」出自**每区 300 张**的扫描，
    //   且 23~39% 这个数经项目审计已查明**无出处**（0.15 时代的结果被覆盖）。
    //   全量复测（~3,700/区，frac=2.0 与本值同口径）：
    //     无重排 → ratio 4.0：华北 61.96→63.32 / 华中 59.63→61.04 / 华东 59.15→58.95
    //                     / 华南西南 65.43→67.80 / 东北 60.57→62.19
    //   即**部署口径已追平甚至反超整图**（四区反超、华东 −0.2pp），
    //   且 ratio 6.0 相对 4.0 是 −0.1/−0.5/+0.3/−1.0/−0.8，四区变差且都在噪声内
    //   → 4.0 已定案，不要再改。
    //   ⚠️ 本值至今**未在真机回归过**（多鸟邻居串标 / 拒识门 / 显示门三点），
    //   但拒识门那项已用部署口径实测代偿：外扩后真鸟误拒 0.46%，与整图 0.52% 持平，
    //   剩多鸟串标与显示门待真机。
    // PLAN §8.4「外扩 10-20% 折中」就此作废。
    const val BOX_EXPAND_RATIO = 4.0f
    const val BIRD_CLASS_ID = 15 // YOLOX COCO 类表里 bird=15

    // 非鸟拒识门（桌面误检事故根解）：二分类 __bird__/__not_bird__，检测框 crop 先过它，
    // 判非鸟（top1=not_bird 且 ≥REJECT_THRESHOLD）→ 不给候选名。实测 FP 拒识率 99.1% /
    // 真鸟误拒 0.6% @0.8（tools/detector_b/eval_reject.py，2026-10-07）。
    // ⚠️ 2026-10-10 补：上面那个 0.6% 是**整图**口径的数字，而端上喂进来的是
    //   BOX_EXPAND_RATIO 外扩后的 crop，口径确实不同，故已补测（n=18,618）：
    //   部署口径真鸟误拒率 **0.46%**，与整图的 0.52% 基本相同（−0.06pp）
    //   ⇒ **外扩没有带来额外误拒，本阈值不需要动**。
    //   残留信号：裁剪短边 <64px 的 4.3% 样本误拒 5.20%（其余 0.21%）。
    //   负样本侧因原始视频不在盘上仍无部署口径数据 ⇒ 将来若要调阈值，须先补那段。
    const val REJECT_ASSET = "models/birdnet_reject.tflite"
    const val REJECT_LABELS = "models/labels_reject.txt"
    const val REJECT_DIM = 2
    const val REJECT_THRESHOLD = 0.8f

    // 全国兜底 v2（384 类 = 五区并集 244 + 缺口补抓 140，2026-10-08 训练；
    // 验收：零泄漏同题 top1 60.56%
    // vs 旧 cn373 57.71%，gap 桶 +6.2pp）。旧 373 类退役，路由 key 沿用 "cn373"。
    // 旧版"384 维只有 373 有标签"的坑不复存在：v2 输出 384 维与 labels 全对齐。
    const val CN373_ASSET = "models/birdnet_national_v2.tflite"
    const val CN373_LABELS = "models/labels_national_v2.txt"
    const val CN373_TENSOR_DIM = 384
    const val CN373_DIM = 384

    // 华北 148 类（rec 补量版，val top-1 60.21%；旧 huabei_mnv4_300 146 类已淘汰）
    const val HUABEI_ASSET = "models/birdnet_huabei.tflite"
    const val HUABEI_LABELS = "models/labels_huabei.txt"
    const val HUABEI_DIM = 148
    // 华北 bbox：纬 36~43，经 110~120（tools/regions.json，区域划分 v2）
    val HUABEI_BBOX = doubleArrayOf(36.0, 43.0, 110.0, 120.0)

    // 华中 150 类（25k 跨区借用版，val top-1 57.18%）：豫鄂湘（含河南全境）
    // ✅ 实测确认（2026-10-05，直接解析 tflite flatbuffer，不靠注释）：
    //    输出张量 = [1,150]，labels_huazhong.txt = 150 行，export_meta.json n_species = 150。
    //    三者一致，HUAZHONG_DIM 就是 150。
    //    ⚠️ 本行原先写着「模型张量/labels/训练元数据都是 145，实测 [1,145]」—— 那是**错的**，
    //    照它把 HUAZHONG_DIM 改成 145 会让 FloatArray(145) 与张量 [1,150] 不匹配、
    //    抛异常后被 RegionClassifierRouter 的 catch(Throwable) 吞掉，华中分类**静默全废**。
    //    历史上的 145 是 17k 旧版模型的类数，25k 借用版补齐 5 个种后已是 150。
    const val HUAZHONG_ASSET = "models/birdnet_huazhong.tflite"
    const val HUAZHONG_LABELS = "models/labels_huazhong.txt"
    const val HUAZHONG_DIM = 150
    // 华中 bbox：纬 31~37，经 105~117；与华北框重叠带（36~37°N）优先华北
    val HUAZHONG_BBOX = doubleArrayOf(31.0, 37.0, 105.0, 117.0)

    // 华南+西南 148 类（rec 补量版，val top-1 61.27%）：粤桂琼云贵川渝
    // ⚠️ UI 层已按约定解耦为 华南(huanan)/西南(xinan) 两个大区（PLAN §七），
    //    但推理模型仍是这一个合并 148 类模型，两侧共用。
    const val HUANAN_SW_ASSET = "models/birdnet_huanan_sw.tflite"
    const val HUANAN_SW_LABELS = "models/labels_huanan_sw.txt"
    const val HUANAN_SW_DIM = 148
    val HUANAN_SW_BBOX = doubleArrayOf(20.0, 31.0, 97.0, 117.0)

    /**
     * 合并框内 GPS 判侧：huanan（粤桂琼）/ xinan（云贵川渝）。
     * 按省界质心最近距离归省，公式与 tools/split_huanan_sw_guide.py 的图鉴拆侧一致
     * （纬度距离加权 1.2x）。bbox 矩形区分不开粤桂与黔渝的重叠带，故用质心法。
     */
    fun huananSwSide(lat: Double, lon: Double): String {
        fun d2(la: Double, lo: Double) = (la - lat) * (la - lat) * 1.2 + (lo - lon) * (lo - lon)
        val south = minOf(d2(23.7, 108.8), d2(23.3, 113.8), d2(19.0, 109.8)) // 桂/粤/琼
        val sw = minOf(d2(25.0, 101.8), d2(26.6, 106.9), d2(30.3, 103.0), d2(29.7, 107.0)) // 滇/黔/川/渝
        return if (south <= sw) "huanan" else "xinan"
    }

    // 华东 150 类（huadong_mnv4_300_rec，val top-1 56.12%）：鲁苏皖浙赣闽。
    // ⚠️ 用 rec 不用 huadong_mnv4_300：那版是 iNat 补量前的 149 类旧表（55.52%）
    const val HUADONG_ASSET = "models/birdnet_huadong.tflite"
    const val HUADONG_LABELS = "models/labels_huadong.txt"
    const val HUADONG_DIM = 150
    // 华东 bbox：纬 27~35，经 117~123（tools/regions.json）
    val HUADONG_BBOX = doubleArrayOf(27.0, 35.0, 117.0, 123.0)

    // 东北 150 类（dongbei_mnv4_rec 补量版，val top-1 58.82%）。
    // ⚠️ 用户拍板（2026-10-06 晚）：用 rec 尽管比 v1（59.18%）低 0.36 点——此决定推翻
    // 「低于 59.18% 就锚回 v1」的早期指示后来者别改回去（2026-10-06 用户拍板）。
    const val DONGBEI_ASSET = "models/birdnet_dongbei.tflite"
    const val DONGBEI_LABELS = "models/labels_dongbei.txt"
    const val DONGBEI_DIM = 150
    // 东北 bbox：纬 39~55，经 119~135（tools/regions.json）。
    // 与华北的重叠带 39–43°N×119–120°E（蒙东一角）→ Router 先判华北，该带留华北侧
    val DONGBEI_BBOX = doubleArrayOf(39.0, 55.0, 119.0, 135.0)

    // 分类软门槛：框到了但分类器 Top-1 低于此值 → 视为"不像鸟"，不显示候选
    const val MIN_CLASS_PROB = 0.15f

    // geomodel 先验查表重排：assets/prior/<region>.json
    // 只重排不删候选（软掩码已实测否决）
    const val PRIOR_DIR = "prior"

    /**
     * 先验权重指数，**逐区**取值。
     *
     * 第一次改（2026-10-09 上午）：原来五区写死 1.0f，但那个值只测过**华北一个区**
     * （早期交接写「0.5~1.0 差距 <0.1」——该差距只对华北成立）。
     * 第二次改（同日下午）：全部按 **rank 方式**重定（见 PRIOR_SCALE），因为
     * 部署口径下 rank 的最优 α 与 prob 不同。
     *
     * 部署口径**全量**实测（`tools/eval_crop_regime.py --detect --n-per 4000` +
     * `--eval --prior --fracs 2.0`，frac=2.0 即端上 BOX_EXPAND_RATIO=4.0，
     * 每区 3,613~3,779 张，2026-10-09 21:30），top-1 绝对值：
     *
     *   区          无重排   rank@0.5   rank@1.0   取值    全量最优档
     *   华北       63.3%      64.2     **64.5**   1.0     rank@1.0  ✅
     *   华中       61.0%    **62.1**     61.7     0.5     rank@0.5  ✅
     *   华东       59.0%      59.3     **59.4**   1.0     rank@1.0  ✅
     *   华南+西南  67.8%      68.7      68.9     0.5     prob@1.0（69.1，差 0.2pp）
     *   东北       62.2%    **63.0**     62.6     0.5     rank@0.5  ✅
     *
     * ⚠️ 2026-10-10 已修正本注释的依据（值一个字没动）：
     *   ① 原表引用的是每区 **300 张**的噪声表（基线比全量高 3.6~6.2pp），
     *      项目审计已判定那张表「五区最优增益全部落在噪声内（σ<1.1）」，
     *      §8 用 ~3,700/区 全量重跑过——现表即全量结果。
     *   ② 原注「华东不可信，维持 1.0」：**前提已作废**。华东验证集当天已按训练切分补记为
     *      6,905 张 / 150 种、判分 56.16%。华东现为可复算数据，
     *      rank@1.0 确实是该区最优。
     *   ③ 原注「东北未测，0.5 依据整图口径」：**已补测**。
     *      dongbei_mnv4_rec\best_eval_g49_nearest.json（2026-10-09 19:22）整图口径
     *      rank@0.5 = 59.53% 为该区最优；部署口径也已补，上表 63.0% 即其最优档。
     *   ④ 五区取值全部与全量最优一致（华南西南那 0.2pp 差在 ±0.73pp 噪声内），
     *      **本轮复核后代码无需改动**，改的是依据。
     *   ⚠️ prob 与 rank 的差距逐区 ≤0.5pp < 噪声，故两种都不算错，保持 rank 不动。
     */
    private val PRIOR_ALPHA_BY_REGION = mapOf(
        "huabei" to 1.0f,
        "huazhong" to 0.5f,
        "huanan" to 0.5f, "xinan" to 0.5f,
        "huadong" to 1.0f,
        "dongbei" to 0.5f,
    )

    /** 该区的先验指数；未登记的区回退 1.0（等价于旧行为，不引入新风险）。 */
    fun priorAlpha(region: String): Float = PRIOR_ALPHA_BY_REGION[region] ?: 1.0f

    /**
     * 先验的乘入方式：`"rank"`（默认）或 `"prob"`。
     *
     * `prob` = 直接乘表里的数字。表是 geomodel 压到 49 个网格点压出来的，
     * **绝对值不可靠，且各区尺度差很多**（实测中位数：华东 0.357 / 华中 0.244 /
     * 东北 **0.057**）。数字小的区里，连最该出现的那种鸟都被打上一刀
     * ——这正是东北用 prob α=1.0 会变差的机制。
     *
     * `rank` = 只看「在这一堆鸟里排第几名」，第 1 名记 1.0、第 k 名记 k/类数，
     * 并列取平均秩（与 `scipy.stats.rankdata` 默认一致，保证与 PC 侧同口径）。
     * 第 1 名永远不被数字大小牵连，α 的含义因此跨区稳定。
     *
     * 代价：rank 下手没那么狠，压不到 prob 的 1e-6 地板
     *（prob 靠这个拿到过更高峰值）。但**部署口径下
     * rank 全面胜出**（§3.3），故默认 rank。要切回旧行为只改这一个常量。
     */
    const val PRIOR_SCALE = "rank"

    // ===== 区域显示文案（类数运行时派生，不写死）=====
    // UI 里的「华北 148 类」不再手写死数：类数 = labels 资产行数运行时派生，
    // 换模型类数变了 UI 自动跟上；行数读不到降级为纯区名（log.w 可见，降级必须可见）。

    /** 区域 key（UI/路由口径）→ 中文名。huanan/xinan 共用合并模型（PLAN §七 UI 解耦约定）。 */
    val REGION_NAMES = mapOf(
        "huabei" to "华北", "huazhong" to "华中", "huanan" to "华南", "xinan" to "西南",
        "huadong" to "华东", "dongbei" to "东北", "cn373" to "全国",
    )

    /** 区域 key → labels 资产路径（类数派生数据源）。 */
    fun labelsAssetOf(region: String): String? = when (region) {
        "huabei" -> HUABEI_LABELS
        "huazhong" -> HUAZHONG_LABELS
        "huanan", "xinan" -> HUANAN_SW_LABELS
        "huadong" -> HUADONG_LABELS
        "dongbei" -> DONGBEI_LABELS
        "cn373" -> CN373_LABELS
        else -> null
    }

    /** labels 资产有效行数（非空行；区名显示与启动自检共用的类数唯一来源）。 */
    fun labelCount(context: Context, labelsAsset: String): Int? = try {
        context.assets.open(labelsAsset).bufferedReader().readLines().count { it.isNotBlank() }
    } catch (t: Throwable) {
        android.util.Log.w("ModelRegistry", "labels 行数读取失败，UI 降级为纯区名: $labelsAsset", t)
        null
    }

    /** 区域显示名（"华北 148 类"）。行数读不到时降级为纯区名。 */
    fun regionDisplayName(context: Context, region: String): String {
        val name = REGION_NAMES[region] ?: region
        val asset = labelsAssetOf(region) ?: return name
        val n = labelCount(context, asset) ?: return name
        return "$name $n 类"
    }
}

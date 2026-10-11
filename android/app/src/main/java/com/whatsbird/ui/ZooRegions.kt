package com.whatsbird.ui

/**
 * 中国鸟类地理区划（一级 I–VII 区）→ 中文名。
 *
 * ## 口径来源（两个独立来源，逐字一致）
 *
 * 1. **国家林业和草原局**《森林与人类》2021-04-27：
 *    https://www.forestry.gov.cn/c/www/dw/150295.jhtml
 *    > 「在两大动物地理界下，学者将中国划分为 7 个 I 级地理区，分别是
 *    > **东北区、华北区、蒙新区、青藏区、华中区、横断山区、华南区**，前 4 区属古北界，后 3 区属东洋界。」
 *
 * 2. **360 百科《中国动物地理区划》**（源：中国科学院蓝色动物学）：
 *    https://baike.so.com/doc/6720670-6934721.html
 *    > 「下分七个区：东北区、华北区、蒙新区、青藏区、西南区、华中区、华南区。」
 *
 * ## 为什么不用「七大地理分区」（华北/东北/华东/华中/华南/西南/西北）
 *
 * * 那套是**地理课本 / 行政经济地理**的约定，不是政府发布的区划，也**没有权威出处文件**；
 *   里面的「华东」在鸟类学界**根本不是一个区名**（江苏皖北归华中、闽浙南归华南）。
 * * 它**没有「蒙新/青藏」**，而蒙新、内蒙与河西走廊、准噶尔盆地、塔里木盆地正是我们最缺的模型覆盖区。
 * * 本数据集（王彦等 2021）用的就是这一套 I–VII 编码，与区划**天然对齐**，无需二次映射。
 *
 * ## 关于「西北」
 *
 * 用户关心的西北五省（新疆/青海/甘肃/宁夏/西藏）在这一体系里有两个家：
 * * [ZOO_MONXIN] 蒙新区 = 内蒙古 + 鄂尔多斯高原 + 阿拉善（含**河西走廊**）+ **塔里木** + 柴达木 + **准噶尔盆地** + **天山山脉**
 * * [ZOO_QINGZANG] 青藏区 = **青海**（柴达木盆地除外）+ **西藏** + **四川西部**，北由昆仑山/阿尔金山/祁连山环绕
 *
 * ## 与「识别区」的关系
 *
 * 本表是**地理标签**（这个种分布在哪），与识别区的模型覆盖**完全解耦**：
 * 识别区来自 `guide_<region>.json`（= 各区模型的类别表），由 bbox + GBIF 采样决定，
 * **与区属统计无关**（已实测：区划代码在六区间的分布差异 ≤1.7pp）。
 * 西北没有区域模型 → 识别落到全国兜底模型，但**地理标签照常正确**。
 */
object ZooRegions {

    /** 一级区：罗马数字前缀 → 中文名。 */
    private val TOP = mapOf(
        "I" to "东北区",
        "II" to "华北区",
        "III" to "蒙新区",
        "IV" to "青藏区",
        "V" to "西南区",
        "VI" to "华中区",
        "VII" to "华南区",
    )

    /** 展示序（沿用数据集的 I→VII 编号序，不按地理方位重排）。 */
    val ORDER = listOf("I", "II", "III", "IV", "V", "VI", "VII")

    /**
     * 一级区代码集合。数据集里出现过 `I` 单独出现（1 种），
     * 统一归到 I 区。
     */
    private val CODE_TO_TOP = listOf(
        "VII" to "VII", "VI" to "VI", "IV" to "IV", "III" to "III",
        "II" to "II", "V" to "V", "I" to "I",
    ).toMap()

    /** 区划代码（如 `VIIB`）→ 所属一级区代码（`VII`）。 */
    fun topOf(code: String): String? {
        for ((prefix, top) in CODE_TO_TOP) {
            if (code == prefix || code.startsWith(prefix)) return top
        }
        return null
    }

    fun name(top: String): String = TOP[top] ?: top

    /**
     * 一组区划代码 → 一级区中文名列表（去重、按 [ORDER] 排序）。
     * @return 无法解析时返回空列表（调用方据此显示"暂缺"，不要编造归属）
     */
    fun displayNames(codes: Collection<String>): List<String> {
        val tops = codes.mapNotNull { topOf(it) }.toSortedSet(
            compareBy { ORDER.indexOf(it).let { i -> if (i < 0) Int.MAX_VALUE else i } },
        )
        return tops.map { name(it) }
    }

    // ---------------------------------------------------------------- 识别模型绑定

    /**
     * 一级区 → 区域模型 key。**value 仍是旧的模型 key**，所以
     * 持久化的 `manualRegion` 语义不变、路由代码一行都不用改，只是显示名换成新体系。
     *
     * 蒙新区 / 青藏区**没有区域模型**（`labelsAssetOf` 返回 null），
     * 按约定在识别页下拉里**不出现**——用户在西北本来就只能落全国兜底模型。
     */
    data class ModelBinding(val zooTop: String, val modelKey: String)

    val MODEL_BINDINGS: List<ModelBinding> = listOf(
        ModelBinding("I", "dongbei"),     // 东北区 → 东北模型 150 类
        ModelBinding("II", "huabei"),     // 华北区 → 华北模型 148 类
        ModelBinding("V", "xinan"),       // 西南区 → 与华南共用 huanan_sw 模型
        ModelBinding("VI", "huazhong"),   // 华中区 → 华中模型 150 类
        ModelBinding("VII", "huanan"),    // 华南区 → 与西南共用 huanan_sw 模型
    )

    /** 有区域模型的一级区代码（供识别页下拉过滤）。 */
    val WITH_MODEL: List<String> = MODEL_BINDINGS.map { it.zooTop }

    fun modelKeyOf(zooTop: String): String? =
        MODEL_BINDINGS.firstOrNull { it.zooTop == zooTop }?.modelKey

    /** 旧模型 key → 新一级区代码（用于把已持久化的 `manualRegion` 显示成新名字）。 */
    fun zooTopOfModelKey(modelKey: String): String? =
        MODEL_BINDINGS.firstOrNull { it.modelKey == modelKey }?.zooTop
}
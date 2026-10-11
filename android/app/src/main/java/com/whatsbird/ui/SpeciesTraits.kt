package com.whatsbird.ui

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale

/**
 * 物种性状（图鉴详情页「物种档案」卡片的数据模型）。
 *
 * 数据源：**中国鸟类的生活史和生态学特征数据集**（王彦平等，《生物多样性》2021, 29(9):1149–1153，
 * DOI 10.17520/biods.2021201）。数据论文，引用即可商用 ✅ —— 出处、授权、交叉校验见
 * `docs/guides/dataset_traits_README.md`，`docs/legal/LICENSES.md` §D。
 *
 * 资产 `assets/guide/traits.json` 由 `tools/guide_traits/build_traits_asset.py` 生成，
 * 按学名索引；384 种图鉴骨架里覆盖 381 种，异名映射的逐条证据见 `tools/guide_traits/match_report.csv`。
 *
 * 口径：
 *  - 数值一律是**文献区间** [min,max]，缺失为 null（不是 0）；
 *  - 体重 g；体长 / 嘴峰 / 翅长 / 尾长 / 跗蹠长 mm；卵 mm；
 *  - 6 项测量值原表**分雌雄**，故用 [Measured] 装；其余为单值；
 *  - 巢型 / 巢址 / 集群 / 居留型在原表是**编码**，译名见下方 MAP（不改动原值语义）。
 */
data class SpeciesTraits(
    val order: String? = null,
    val family: String? = null,
    val endemic: Boolean = false,
    val mass: Measured? = null,
    val length: Measured? = null,
    val bill: Measured? = null,
    val wing: Measured? = null,
    val tail: Measured? = null,
    val tarsus: Measured? = null,
    val diet: String? = null,
    val clutch: List<Double>? = null,
    val egg: Pair<List<Double>, List<Double>>? = null,
    val nestType: List<String> = emptyList(),
    val nestSite: List<String> = emptyList(),
    val flock: List<String> = emptyList(),
    val migrant: List<String> = emptyList(),
    val provinces: Int? = null,
    /**
     * 地理分布区划代码，如 `["VIIB","VA"]`（括号里的居留型 R/S/W/P/V 已剥离）。
     * 一级区 I–VII 的中文名与口径见 [ZooRegions]（两个独立来源逐字一致）。
     * 数据集 1445 种**全部有值**（该字段完整度 100%）。
     */
    val distRange: List<String> = emptyList(),
)

/** 分雌雄的区间值；原表某一侧缺失时为 null。 */
data class Measured(val male: List<Double>?, val female: List<Double>?)

/** 原表编码 → 中文（照数据集的图例，不自创）。 */
internal val NEST_TYPE_NAME = mapOf("1" to "洞巢", "2" to "开放巢", "3" to "寄生")
internal val NEST_SITE_NAME =
    mapOf("1" to "地面", "2" to "水面", "3" to "灌丛", "4" to "树冠", "5" to "岩壁")
internal val FLOCK_NAME = mapOf("0" to "不集群", "1" to "小群", "2" to "大群")
internal val MIGRANT_NAME =
    mapOf("R" to "留鸟", "S" to "夏候鸟", "W" to "冬候鸟", "P" to "旅鸟", "V" to "迷鸟")

/**
 * 性状资产加载器。
 *
 * ⚠️ 与 `buildCatalog()` 同一约束：调用点在 IO 线程（GuideViewModel 的 `flowOn(Dispatchers.IO)`）。
 * 资产 ~184 KB，解析约数十毫秒，**不要**放主线程。
 * 资产缺失/损坏时返回空表，图鉴整体照常工作，卡片降级为「暂缺」。
 */
object TraitsCatalog {
    private const val ASSET = "guide/traits.json"

    fun load(ctx: Context): Map<String, SpeciesTraits> = try {
        val root = JSONObject(ctx.assets.open(ASSET).bufferedReader().readText())
        val sp = root.optJSONObject("species")
        if (sp == null) {
            emptyMap()
        } else {
            val out = HashMap<String, SpeciesTraits>(sp.length() * 2)
            for (sci in sp.keys()) {
                sp.optJSONObject(sci)?.let { out[sci] = parse(it) }
            }
            out
        }
    } catch (e: Exception) {
        android.util.Log.w("TraitsCatalog", "性状资产不可用，图鉴降级为无档案卡片", e)
        emptyMap()
    }

    private fun parse(o: JSONObject): SpeciesTraits = SpeciesTraits(
        order = o.optString("order").takeIf { it.isNotBlank() && it != "null" },
        family = o.optString("family").takeIf { it.isNotBlank() && it != "null" },
        endemic = o.optBoolean("endemic", false),
        mass = measured(o, "mass"),
        length = measured(o, "length"),
        bill = measured(o, "bill"),
        wing = measured(o, "wing"),
        tail = measured(o, "tail"),
        tarsus = measured(o, "tarsus"),
        diet = o.optString("diet").takeIf { it.isNotBlank() && it != "null" },
        clutch = range(o.optJSONArray("clutch")),
        egg = egg(o.optJSONArray("egg")),
        nestType = codes(o, "nestType"),
        nestSite = codes(o, "nestSite"),
        flock = codes(o, "flock"),
        migrant = codes(o, "migrant"),
        provinces = if (o.has("provinces") && !o.isNull("provinces")) o.optInt("provinces") else null,
        distRange = codes(o, "distRange"),
    )

    private fun range(a: JSONArray?): List<Double>? {
        if (a == null || a.length() == 0) return null
        val v = ArrayList<Double>(2)
        for (i in 0 until a.length()) {
            val d = a.optDouble(i, Double.NaN)
            if (!d.isNaN()) v.add(d)
        }
        return v.takeIf { it.isNotEmpty() }
    }

    private fun measured(o: JSONObject, key: String): Measured? {
        val arr = o.optJSONArray(key) ?: return null
        val m = range(arr.optJSONArray(0))
        val f = range(arr.optJSONArray(1))
        return if (m == null && f == null) null else Measured(m, f)
    }

    private fun egg(a: JSONArray?): Pair<List<Double>, List<Double>>? {
        if (a == null || a.length() < 2) return null
        val l = range(a.optJSONArray(0)) ?: return null
        val b = range(a.optJSONArray(1)) ?: return null
        return l to b
    }

    private fun codes(o: JSONObject, key: String): List<String> {
        val a = o.optJSONArray(key) ?: return emptyList()
        val out = ArrayList<String>(a.length())
        for (i in 0 until a.length()) {
            val s = a.optString(i).trim()
            if (s.isNotEmpty() && s != "null") out.add(s)
        }
        return out
    }
}

// ------------------------------------------------------------------ 展示格式化

/** 数值 → 文本：整数不带小数点，其余最多两位小数并去尾随 0。 */
internal fun fmtNum(d: Double): String {
    val l = d.toLong()
    if (d == l.toDouble()) return l.toString()
    return String.format(Locale.US, "%.2f", d).trimEnd('0').trimEnd('.')
}

/** [min,max] → "325–430"；单点 → "340"。 */
internal fun fmtRange(v: List<Double>?, unit: String = ""): String? {
    if (v.isNullOrEmpty()) return null
    val lo = v[0]
    val hi = v.getOrElse(1) { lo }
    val s = if (lo == hi) fmtNum(lo) else "${fmtNum(lo)}–${fmtNum(hi)}"
    return if (unit.isEmpty()) s else "$s $unit"
}

/** 分雌雄的区间 → "12–15 g（雄）· 11–14 g（雌）"；两侧相同则只写一次。 */
internal fun fmtMeasured(m: Measured?, unit: String = ""): String? {
    if (m == null) return null
    val a = fmtRange(m.male, unit)
    val b = fmtRange(m.female, unit)
    return when {
        a != null && b != null && a == b -> a
        a != null && b != null -> "$a（雄）· $b（雌）"
        a != null -> "$a（雄）"
        b != null -> "$b（雌）"
        else -> null
    }
}

/** 卵大小 → "18.5 × 14.2 mm"，带区间时 "28–43 × 31–34 mm"。 */
internal fun fmtEgg(e: Pair<List<Double>, List<Double>>?): String? {
    val l = fmtRange(e?.first) ?: return null
    val b = fmtRange(e?.second) ?: return null
    return "$l × $b mm"
}

/** 编码表 → "洞巢"、"地面、树冠"；不在图例里的码原样保留，便于回查原表。 */
internal fun fmtCodes(v: List<String>, names: Map<String, String>): String? {
    if (v.isEmpty()) return null
    return v.joinToString("、") { names[it] ?: it }
}

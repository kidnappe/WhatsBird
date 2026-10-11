package com.whatsbird.vision

import android.content.Context
import android.location.Location
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.Calendar
import java.util.concurrent.ConcurrentHashMap
import javax.inject.Inject
import javax.inject.Singleton
import org.json.JSONObject

/**
 * 分区×月份 geomodel 先验查表 + 重排（docs/handoff/root/RERANK-DEVICE.md 方案 A）。
 *
 * 表：assets/prior/<key>.json（tools/distill_prior_table.py 产出，49 点网格 × 12 月）。
 * 华北 frozen val 实测：nearest 查表 top-1 +0.80（ONNX 真实坐标上限 +1.19）；
 * 无定位的月度均值 +0.27 保底。知识源质量实测见 docs/handoff/root/RERANK-DEVICE.md §4。
 *
 * 查询语义：
 *  · month = 设备日期（候鸟物候已在 geomodel 里，随季节自动变化）
 *  · 位置 = LocationProvider.lastKnown()（被动缓存位置）→ 最近采样点；
 *    无定位 → 全点月度均值。**定位只在本地当查表索引，永不上传。**
 *  · 物种对齐按**学名**映射：表与模型类表独立演进，表里没有的种权重 = 1
 *    （与 PC 端 eval_report 同口径）；只重排不删候选（软掩码已否决）。
 */
@Singleton
class PriorReranker @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    /** points: [npts][lat,lon]；byMonth: [12][npts][ncls]（月→采样点→类）。 */
    private class Table(
        val points: Array<DoubleArray>,
        val species: Array<String>,
        val byMonth: Array<Array<FloatArray>>,
    )

    private val tables = ConcurrentHashMap<String, Table>()
    private val tableMisses = ConcurrentHashMap.newKeySet<String>()
    private val vecCache = ConcurrentHashMap<String, Map<String, Float>>()

    /** 路由器的区域 key → 先验表 key（华南/西南共用合并模型的同一张表）。 */
    private val PRIOR_KEYS = listOf("huabei", "huazhong", "huanan_sw", "huadong", "dongbei")

    private fun tableKey(region: String): String? = when (region) {
        "huabei" -> "huabei"
        "huazhong" -> "huazhong"
        "huanan", "xinan" -> "huanan_sw"
        "huadong" -> "huadong"
        "dongbei" -> "dongbei"
        else -> null // cn373：不重排
    }

    /** 最近一次查表降级原因（null = 正常）。CameraVM 心跳据此打 prior=off，降级必须可见
     *  （APP-MODELS §5.2：文件名不匹配静默失效的事故族教训）。 */
    @Volatile var degradeReason: String? = null
        private set

    fun rerank(scores: List<ClassScore>, region: String, loc: Location?): List<ClassScore> {
        if (scores.size < 2) return scores
        val key = tableKey(region)
        if (key == null) {
            degradeReason = "region=$region 无先验表"
            return scores
        }
        // ⚠️ 2026-10-10：ConcurrentHashMap 不接受 null 值，加载失败改记进 tableMisses，
        // 避免每帧重复解析 860KB JSON（原先 HashMap.getOrPut 缓存了 null）。
        if (key in tableMisses) {
            degradeReason = "$key 表加载失败"
            return scores
        }
        val t = tables[key] ?: loadTable(key)?.also { tables[key] = it } ?: run {
            tableMisses.add(key)
            degradeReason = "$key 表加载失败"
            return scores
        }
        degradeReason = null
        val month = Calendar.getInstance().get(Calendar.MONTH) + 1

        // 采样点：有定位取最近点；无定位 pt=-1（后面取全点均值）
        val pt = if (loc == null) -1 else t.points.indices.minByOrNull { i ->
            val dla = t.points[i][0] - loc.latitude
            val dlo = t.points[i][1] - loc.longitude
            dla * dla + dlo * dlo
        } ?: -1
        // 空 map 表示「该月/该点无先验向量」：查表全落 1f，权重恒等，等价于旧代码的 return scores。
        val vec = vecCache["$key/$month/$pt"] ?: run {
            val arr = t.byMonth.getOrNull(month - 1) ?: return scores
            val n = t.species.size
            // 先按查表语义取该点（或无定位时的全点均值）的原始数字
            val raw = FloatArray(n) { c ->
                if (pt >= 0) arr[pt][c]
                else arr.sumOf { it[c].toDouble() }.toFloat() / arr.size
            }
            // 再按 ModelRegistry.PRIOR_SCALE 决定用原值还是转成排名值（2026-10-09 加）
            val vals = if (ModelRegistry.PRIOR_SCALE == "rank") rankValues(raw, n) else raw
            HashMap<String, Float>(n).also { m ->
                // ⚠️ 2026-10-10 修复 P0：原先这里存原样学名（"Accipiter nisus"），
                // 下面查表却用 lowercase()，永远查不到 → 权重恒为 1，重排对五区静默失效，
                // 且 degradeReason 仍为 null（心跳照常显示 prior=on）。两侧必须同为小写。
                t.species.forEachIndexed { c, s -> m[s.lowercase()] = vals[c] }
            }.also { vecCache["$key/$month/$pt"] = it }
        }
        if (vec.isEmpty()) return scores

        // α 逐区取（2026-10-09）：华中/东北在 α=1.0 下实测为**负收益**，改 0.5。
        // 未登记的区回退 1.0（等价旧行为）。依据见 ModelRegistry.PRIOR_ALPHA_BY_REGION 注释。
        val alpha = ModelRegistry.priorAlpha(region).toDouble()
        var sum = 0f
        val rescored = scores.map { s ->
            val g = (vec[s.sci.lowercase()] ?: 1f).coerceIn(1e-4f, 1.0f)
            val p = s.prob * Math.pow(g.toDouble(), alpha).toFloat()
            sum += p
            s.copy(prob = p)
        }
        if (sum <= 0f) return scores
        return rescored.map { it.copy(prob = it.prob / sum) }
            .sortedByDescending { it.prob }
    }

    /**
     * 把一组原始数字换成「排名值」：第 1 名 → 1.0，第 k 名 → k/总数。
     *
     * 为什么需要：那张表是把大模型压到 49 个网格点压出来的，**绝对数字不可靠，
     * 而且各区尺度差很多**（实测中位数：华东 0.357 / 华中 0.244 / 东北 0.057）。
     * 直接乘数字时，数字小的区里连最该出现的那种鸟都被打上一刀。
     * 只看排名则不受影响——第 1 名永远拿 1.0。
     *
     * 并列取**平均秩**（与 scipy.stats.rankdata 的默认 'average' 一致），
     * 保证和 PC 侧 tools/eval_report.py 的 --prior-scale rank 逐位一致。
     */
    private fun rankValues(v: FloatArray, n: Int): FloatArray {
        val order = v.indices.sortedBy { v[it] }
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            var j = i
            while (j + 1 < n && v[order[j + 1]] == v[order[i]]) j++
            val avgRank = (i + j) / 2.0 + 1.0        // 1-based 平均秩
            val w = (avgRank / n).toFloat()
            for (k in i..j) out[order[k]] = w
            i = j + 1
        }
        return out
    }

    /**
     * 预热：在后台线程预先解析全部先验表。
     * 2026-10-10 加——原先首次 rerank 会在推理线程上同步解析约 860KB JSON，
     * 那几帧必然掉帧；调用方应在启动后于 Dispatchers.Default 上调一次。
     */
    fun preload() {
        for (key in PRIOR_KEYS) {
            if (key in tableMisses) continue
            tables[key] ?: loadTable(key)?.also { tables[key] = it } ?: tableMisses.add(key)
        }
    }

    private fun loadTable(key: String): Table? = try {
        // ⚠️ 资产文件名带蒸馏参数后缀（<key>_geomodel_g49.json）——2026-10-07 修复的 P0：
        // 此前打开 <key>.json 永远 FileNotFoundException，重排静默失效收益为 0（APP-MODELS §5.1）。
        // 建议改代码而非改资产名：g49→g64 换代时资产名会变，逻辑跟文件名走更脆。
        val o = JSONObject(
            context.assets.open("${ModelRegistry.PRIOR_DIR}/${key}_geomodel_g49.json")
                .bufferedReader().use { it.readText() },
        )
        val ptsA = o.getJSONArray("points")
        val spA = o.getJSONArray("species")
        val pr = o.getJSONObject("prior")
        Table(
            Array(ptsA.length()) { i ->
                val p = ptsA.getJSONArray(i)
                doubleArrayOf(p.getDouble(0), p.getDouble(1))
            },
            Array(spA.length()) { i -> spA.getString(i) },
            Array(12) { mi ->
                val ma = pr.getJSONArray((mi + 1).toString())
                Array(ma.length()) { p ->
                    val ca = ma.getJSONArray(p)
                    FloatArray(ca.length()) { c -> ca.getDouble(c).toFloat() }
                }
            },
        )
    } catch (t: Throwable) {
        // 表缺失/损坏只降级为不重排，绝不拖垮分类
        android.util.Log.w("PriorReranker", "先验表加载失败: $key", t)
        null
    }
}

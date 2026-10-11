package com.whatsbird.vision

import android.content.Context
import android.os.Looper
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 离线行政区划反查（点→"省 市 区县"）：docs/guides/OFFLINE-ADMIN-GEO.md。
 *
 * 数据：assets/geo/admin_county.ndjson（tools/admin_geo/generate_admin_geo.py 生成，
 * DataV.GeoAtlas 边界，县级叶子 + bbox，坐标 GCJ-02、度×1e4 量化 ≈11m）。
 * 查询：① 查询点 WGS-84→GCJ-02（eviltransform 公开前变换——边界是火星坐标，GPS 是
 * 地心坐标，不对齐会在县界附近 300~700m 判错县）；② bbox 线性预筛（3400 个矩形，
 * 微秒级）；③ 射线法偶奇判内（外环命中、洞环翻转，环独立无嵌套所以奇偶性成立）。
 *
 * 懒加载在 IO 线程（CameraViewModel 启动即预热）；「最近命中」缓存放最前
 * ——观鸟一次 outing 不挪窝，连续记录几乎零扫描。
 */
@Singleton
class AdminGeoIndex @Inject constructor(
    @ApplicationContext private val ctx: Context,
) {

    data class County(
        val adcode: Int,
        val province: String,
        val city: String,
        val district: String?,
        val minLon: Int, val minLat: Int, val maxLon: Int, val maxLat: Int,
        val rings: List<IntArray>, // 每个 IntArray = [lon0,lat0,lon1,lat1,...]（度×SCALE）
    )

    @Volatile private var counties: List<County>? = null
    @Volatile private var lastHit: County? = null
    @Volatile private var loading = false

    /** 预热入口（IO 线程调）：解析 NDJSON 进内存，避免首次落库多等一两秒。 */
    fun preload() {
        ensure()
    }

    /** lat/lon = WGS-84（GPS 原始坐标）。返回如"湖北省 武汉市 洪山区"；境外/未覆盖返回 null。 */
    fun locate(lat: Double, lon: Double): String? {
        // ⚠️ 2026-10-10：原先直接 ensure()，若调用方忘了先 preload()，
        //   主线程首次 locate 会同步解析 8MB NDJSON（数百毫秒到数秒卡顿）。
        //   现在主线程上未就绪就返回 null 并异步补加载；IO 线程调用行为不变（同步等待）。
        if (counties == null && isMainThread()) {
            android.util.Log.w(
                "AdminGeoIndex",
                "locate 在主线程且行政区索引未就绪，本帧返回 null；请在 IO 线程先调 preload()",
            )
            preloadAsync()
            return null
        }
        val (glon, glat) = wgs84ToGcj02(lon, lat)
        val x = (glon * SCALE).roundToInt()
        val y = (glat * SCALE).roundToInt()
        lastHit?.let { if (pointIn(it, x, y)) return nameOf(it) }
        for (c in ensure()) {
            if (x in c.minLon..c.maxLon && y in c.minLat..c.maxLat && pointIn(c, x, y)) {
                lastHit = c
                return nameOf(c)
            }
        }
        return null
    }

    private fun isMainThread(): Boolean =
        Looper.myLooper() != null && Looper.myLooper() == Looper.getMainLooper()

    /** 后台补加载（幂等，重复调用只会真正解析一次）。 */
    private fun preloadAsync() {
        if (loading) return
        loading = true
        Thread({
            try {
                ensure()
            } finally {
                loading = false
            }
        }, "geo-index-load").apply { isDaemon = true }.start()
    }

    private fun pointIn(c: County, x: Int, y: Int): Boolean {
        if (x !in c.minLon..c.maxLon || y !in c.minLat..c.maxLat) return false
        var inside = false
        for (r in c.rings) if (rayRing(r, x, y)) inside = !inside
        return inside
    }

    private fun rayRing(ring: IntArray, x: Int, y: Int): Boolean {
        var inside = false
        var j = ring.size - 2
        var i = 0
        while (i < ring.size) {
            val xi = ring[i]; val yi = ring[i + 1]
            val xj = ring[j]; val yj = ring[j + 1]
            // 交点横坐标用 double：整式比较要处理 (yj-yi) 符号翻转，float 路径免踩坑
            // （量化坐标 ~1.4e6，double 尾数 53bit 远够精确）
            if ((yi > y) != (yj > y) &&
                x < (xj - xi).toDouble() * (y - yi) / (yj - yi) + xi
            ) inside = !inside
            j = i
            i += 2
        }
        return inside
    }

    private fun nameOf(c: County): String = buildString {
        append(c.province)
        if (c.city != c.province) append(' ').append(c.city)
        c.district?.let { if (it.isNotEmpty() && it != c.city) append(' ').append(it) }
    }

    private fun ensure(): List<County> {
        counties?.let { return it }
        synchronized(this) {
            counties?.let { return it }
            val list = ArrayList<County>(3400)
            try {
                ctx.assets.open(ASSET).bufferedReader().useLines { lines ->
                    for (line in lines) {
                        if (line.isBlank()) continue
                        try {
                            val o = org.json.JSONObject(line)
                            val bb = o.getJSONArray("bb")
                            val ga = o.getJSONArray("g")
                            val rings = ArrayList<IntArray>(ga.length())
                            for (i in 0 until ga.length()) {
                                val rj = ga.getJSONArray(i)
                                val r = IntArray(rj.length())
                                for (j in 0 until rj.length()) r[j] = rj.getInt(j)
                                rings.add(r)
                            }
                            val d = if (o.isNull("d")) null else o.getString("d")
                            list.add(
                                County(
                                    o.getInt("a"), o.getString("p"), o.getString("c"), d,
                                    bb.getInt(0), bb.getInt(1), bb.getInt(2), bb.getInt(3),
                                    rings,
                                ),
                            )
                        } catch (t: Throwable) {
                            Log.w(TAG, "跳过坏行", t)
                        }
                    }
                }
                Log.i(TAG, "加载 ${list.size} 个县级区划面")
            } catch (t: Throwable) {
                Log.e(TAG, "区划资产缺失/解析失败（跑 tools/admin_geo/start_generate.bat）", t)
            }
            counties = list
            return list
        }
    }

    // ---------- WGS-84 → GCJ-02（eviltransform 公开算法） ----------

    private fun outOfChina(lon: Double, lat: Double) =
        lon < 72.004 || lon > 137.8347 || lat < 0.8293 || lat > 55.8271

    private fun wgs84ToGcj02(lon: Double, lat: Double): Pair<Double, Double> {
        if (outOfChina(lon, lat)) return lon to lat
        var dLat = transformLat(lon - 105.0, lat - 35.0)
        var dLon = transformLon(lon - 105.0, lat - 35.0)
        val radLat = lat / 180.0 * PI
        var magic = sin(radLat)
        magic = 1 - EE * magic * magic
        val sqrtMagic = sqrt(magic)
        dLat = dLat * 180.0 / (A * (1 - EE) / (magic * sqrtMagic) * PI)
        dLon = dLon * 180.0 / (A / sqrtMagic * cos(radLat) * PI)
        return (lon + dLon) to (lat + dLat)
    }

    private fun transformLat(x: Double, y: Double): Double {
        var ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y +
            0.2 * sqrt(abs(x))
        ret += (20.0 * sin(6.0 * x * PI) + 20.0 * sin(2.0 * x * PI)) * 2.0 / 3.0
        ret += (20.0 * sin(y * PI) + 40.0 * sin(y / 3.0 * PI)) * 2.0 / 3.0
        ret += (160.0 * sin(y / 12.0 * PI) + 320.0 * sin(y * PI / 30.0)) * 2.0 / 3.0
        return ret
    }

    private fun transformLon(x: Double, y: Double): Double {
        var ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y +
            0.1 * sqrt(abs(x))
        ret += (20.0 * sin(6.0 * x * PI) + 20.0 * sin(2.0 * x * PI)) * 2.0 / 3.0
        ret += (20.0 * sin(x * PI) + 40.0 * sin(x / 3.0 * PI)) * 2.0 / 3.0
        ret += (150.0 * sin(x / 12.0 * PI) + 300.0 * sin(x / 30.0 * PI)) * 2.0 / 3.0
        return ret
    }

    private companion object {
        const val TAG = "AdminGeoIndex"
        const val ASSET = "geo/admin_county.ndjson"
        const val SCALE = 10000
        const val A = 6378245.0
        const val EE = 0.006693421622962
    }
}

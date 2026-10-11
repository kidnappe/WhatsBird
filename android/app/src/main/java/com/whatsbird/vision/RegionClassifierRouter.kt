package com.whatsbird.vision

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.location.Location
import dagger.Lazy
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Named
import javax.inject.Singleton

/**
 * 分类路由（PLAN §七 5 区域策略的第一步）：
 * 自动模式 = GPS 落在区域 bbox 内 → 对应区域模型；框外或无定位 → 全国 373 类兜底。
 * 位置缓存 60 秒，避免每帧查询。
 *
 * 另支持手动指定区域（App 内「区域」选择器）：null = 自动 GPS 路由。
 * 手动值仅在进程内生效（重启回自动），用于 GPS 漂移/用户明确知道所在区域时兜底。
 */
@Singleton
class RegionClassifierRouter @Inject constructor(
    @ApplicationContext private val context: Context,
    private val locationProvider: LocationProvider,
    private val priorReranker: PriorReranker,
    // dagger.Lazy：解释器只在首次实际路由到时才创建（全量常驻会拖慢推理）
    @Named("huabei") private val huabei: Lazy<BirdClassifier>,
    @Named("huazhong") private val huazhong: Lazy<BirdClassifier>,
    @Named("huanan_sw") private val huananSw: Lazy<BirdClassifier>,
    @Named("huadong") private val huadong: Lazy<BirdClassifier>,
    @Named("dongbei") private val dongbei: Lazy<BirdClassifier>,
    @Named("reject") private val reject: Lazy<BirdClassifier>,
    @Named("cn373") private val cn373: Lazy<BirdClassifier>,
) : BirdClassifier {

    @Volatile private var cachedAt = 0L
    @Volatile private var cachedRegion = "cn373"
    @Volatile private var lastReason = "初始化"
    @Volatile private var manualRegion: String? = null

    /** 当前生效的区域 key（huabei/huazhong/huanan/xinan/cn373），记录入库用。
     *  huanan/xinan 为 UI 解耦的两个大区（PLAN §七约定），推理共用 huanan_sw 合并模型。 */
    fun currentRegion(): String = pickRegion()

    fun routingReason(): String {
        pickRegion() // 读原因即刷新缓存，无鸟帧也能显示正确路由
        return lastReason
    }

    /** 当前手动指定的区域（null = 自动 GPS 路由）。 */
    fun manualRegion(): String? = manualRegion

    /** 先验重排健康状态（CameraVM 心跳打 prior=on/off 用——降级必须可见）。 */
    fun priorDegradeReason(): String? = priorReranker.degradeReason

    /** 手动指定区域；传 null 回到自动 GPS 路由。切换即失效 GPS 缓存。 */
    fun setManualRegion(region: String?) {
        manualRegion = region
        cachedAt = 0L
        pickRegion() // 立即刷新 reason，UI 不用等下一帧
    }

    // 区域显示文案（含类数）从 labels 资产行数派生，不写死：
    // 换模型类数变了自动跟上，不再手写死数；行数读不到降级为纯区名（ModelRegistry log.w）。
    private val regionLabels: Map<String, String> by lazy {
        ModelRegistry.REGION_NAMES.mapValues { (key, _) ->
            ModelRegistry.regionDisplayName(context, key)
        }
    }

    private fun label(region: String) = regionLabels[region] ?: region

    private fun pickRegion(): String {
        manualRegion?.let { m ->
            lastReason = "手动指定 → ${label(m)}"
            return m
        }
        val now = System.currentTimeMillis()
        if (now - cachedAt > 60_000) {
            val loc = locationProvider.lastKnown()
            val (region, reason) = when {
                loc == null -> "cn373" to "无定位信息 → ${label("cn373")}"
                inBox(loc, ModelRegistry.HUABEI_BBOX) ->
                    // 华北先判：华北∩东北重叠带 39–43°N×119–120°E（蒙东一角）留华北侧
                    "huabei" to "GPS 在华北框内 → ${label("huabei")}"
                inBox(loc, ModelRegistry.DONGBEI_BBOX) ->
                    "dongbei" to "GPS 在东北框内 → ${label("dongbei")}"
                inBox(loc, ModelRegistry.HUADONG_BBOX) ->
                    "huadong" to "GPS 在华东框内 → ${label("huadong")}"
                inBox(loc, ModelRegistry.HUAZHONG_BBOX) ->
                    "huazhong" to "GPS 在华中框内 → ${label("huazhong")}"
                inBox(loc, ModelRegistry.HUANAN_SW_BBOX) -> {
                    // UI 按华南/西南两大区呈现（共模型），框内按省界质心判侧
                    val side = ModelRegistry.huananSwSide(loc.latitude, loc.longitude)
                    side to if (side == "huanan")
                        "GPS 判侧华南（粤桂琼）→ ${label("huanan")}（与西南共模型）"
                    else
                        "GPS 判侧西南（云贵川渝）→ ${label("xinan")}（与华南共模型）"
                }
                else -> "cn373" to "GPS 在区域框外（%.1f, %.1f）→ ${label("cn373")}"
                    .format(loc.latitude, loc.longitude)
            }
            cachedRegion = region
            lastReason = reason
            cachedAt = now
        }
        return cachedRegion
    }

    private fun inBox(loc: Location, b: DoubleArray) =
        loc.latitude in b[0]..b[1] && loc.longitude in b[2]..b[3]

    override fun classify(src: Bitmap): List<ClassScore> = try {
        // 非鸟拒识门（2026-10-07 桌面误检事故的根解）：检测框 crop 先过二分类拒识模型，
        // 判"非鸟"→ 返回空候选（检测框保留，只灭名字）。主卡/角标/重检全走此路自动被门。
        // 阈值 0.8 实测（tools/detector_b/eval_reject.py）：FP crop 拒识率 99.1% / 真鸟误拒 0.6%。
        val rejTop = reject.get().classify(src).firstOrNull()
        val gated = rejTop != null && rejTop.sci == "__not_bird__" &&
            rejTop.prob >= ModelRegistry.REJECT_THRESHOLD
        if (gated) {
            emptyList()
        } else run {
            val region = pickRegion()
            val raw = when (region) {
                "huabei" -> huabei.get().classify(src)
                "huazhong" -> huazhong.get().classify(src)
                "huanan", "xinan" -> huananSw.get().classify(src)
                "huadong" -> huadong.get().classify(src)
                "dongbei" -> dongbei.get().classify(src)
                else -> cn373.get().classify(src)
            }
            // geomodel 先验重排：查表×月份×最近采样点，
            // 只重排不删候选；表缺失/无定位自动降级（月度均值或原样返回）
            priorReranker.rerank(raw, region, locationProvider.lastKnown())
        }
    } catch (oom: OutOfMemoryError) {
        // ⚠️ 2026-10-10：原先 catch(Throwable) 一把兜，**把 OOM 也吞了**。
        // 内存溢出不是"降级种类"能解决的：吞掉之后界面只会莫名其妙空白，
        // 日志里也看不出是内存问题。现在单独接住并按 error 级记账。
        // 注意：这里**不**顺手释放分类器——它们是进程级 Lazy 单例，Lazy 缓存后无法重建，
        // close 掉会让该区域此后永久失效。常驻内存的根治要改注入方式（见报告遗留项）。
        android.util.Log.e("RegionClassifierRouter", "分类 OOM，本帧降级为仅检测框（区域分类器常驻约 70MB）", oom)
        emptyList()
    } catch (t: Throwable) {
        // 分类器故障只降级种类、不拖垮检测框（框是最低限度交付）
        android.util.Log.w("RegionClassifierRouter", "分类失败，本帧降级为仅检测框", t)
        emptyList()
    }
}

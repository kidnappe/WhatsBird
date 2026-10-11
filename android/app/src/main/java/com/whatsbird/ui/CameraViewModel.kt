package com.whatsbird.ui

import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.whatsbird.data.AppPrefs
import com.whatsbird.data.db.ObservationDao
import com.whatsbird.data.db.ObservationEntity
import com.whatsbird.vision.AdminGeoIndex
import com.whatsbird.vision.BirdDetector
import com.whatsbird.vision.BirdTracker
import com.whatsbird.vision.ClassScore
import com.whatsbird.vision.Detection
import com.whatsbird.vision.LocationProvider
import com.whatsbird.vision.ModelRegistry
import com.whatsbird.vision.RegionClassifierRouter
import com.whatsbird.vision.Rescan
import com.whatsbird.vision.ThreeStagePipeline
import dagger.hilt.android.lifecycle.HiltViewModel
import java.util.concurrent.atomic.AtomicBoolean
import javax.inject.Inject
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/**
 * 识别展示状态机（docs/design/DESIGN.md §3.2）：
 * IDLE 找鸟中 → TRACKING 识别中（充能）→ IDENTIFIED 已识别 → LOST 沿用上次识别。
 * 屏幕只显示聚合身份，单帧分类结果先进 EWM 窗（半衰 1.5s），显示 500ms 节流，
 * 排名滞回 1.15x：新第一名须超现第一名 15% 才换位。
 * 身份只升不降：差像素不覆盖好观测；换鸟（跟踪链断链超窗）才清空重算。
 */
enum class IdState { IDLE, TRACKING, IDENTIFIED, LOST }

data class TrackIdentity(
    val zh: String,
    val sci: String,
    val prob: Float,   // 展示置信度（聚合分封顶 1）
    val hits: Int,     // 证据帧数
)

data class ObserveUiState(
    val state: IdState = IdState.IDLE,
    val identity: TrackIdentity? = null,
    val candidates: List<TrackIdentity> = emptyList(), // EWM 排名前 5（确认弹窗数据源）
    val secondary: List<String> = emptyList(), // "苍鹭 6%" 样式的次候选
    val charging: Float = 0f,                  // TRACKING 态充能 0..1
    val detections: List<Detection> = emptyList(), // 主目标框（追焦/自动放大只认它）
    val boxes: List<BoxTag> = emptyList(),     // 多鸟：全部展示框（主目标 + 其余角标）
    val frameW: Int = 0,
    val frameH: Int = 0,
    val inferMs: Float = 0f,
    val fps: Float = 0f,                    // 分析流实际帧率（性能监控叠加层用）
)

/** 多鸟取景框：主目标之外的其他轨迹，角标 = 首分类 top1（未过
 *  MIN_CLASS_PROB 不给名字——低置信显示策略 c）。 */
data class BoxTag(
    val trackId: Int,
    val det: Detection,
    val primary: Boolean,
    val zh: String? = null,
    val prob: Float = 0f,
)

@HiltViewModel
class CameraViewModel @Inject constructor(
    private val pipeline: ThreeStagePipeline,
    private val detector: BirdDetector,
    private val observationDao: ObservationDao,
    private val router: RegionClassifierRouter,
    private val locationProvider: LocationProvider,
    private val appPrefs: AppPrefs,
    private val adminGeo: AdminGeoIndex,
    private val priorReranker: com.whatsbird.vision.PriorReranker,
    @dagger.hilt.android.qualifiers.ApplicationContext
    private val appContext: android.content.Context,
) : ViewModel() {

    private val _ui = MutableStateFlow(ObserveUiState())
    val ui: StateFlow<ObserveUiState> = _ui.asStateFlow()

    /**
     * 跟踪/身份状态专用单线程 dispatcher（2026-10-10 新增）。
     *
     * 修的问题：onFrame 在 Dispatchers.Default 上改 identity / ewm(HashMap) / tracker，
     * 而 setRegion / selectTrack / onZoomChanged / 切档回调在**主线程**改同一批对象。
     * 后果：absorb 里的 `for ((k,v) in ewm)` 与主线程的 ewm.clear() 撞上会抛
     * ConcurrentModificationException（被 onFrame 的 catch 吞掉，只表现为丢帧），
     * emitUi 的 ewm.values.sortedByDescending 同理；tracker 内部列表也会错乱。
     *
     * 现在**所有**改这批状态的入口都串到这一个线程上，物理上不可能再并发。
     * 只读入口（_ui、selectedTrackId 等）走 @Volatile，不受此约束。
     */
    private val stateDispatcher = Dispatchers.Default.limitedParallelism(1)

    private val _savedCount = MutableStateFlow(0)
    val savedCount: StateFlow<Int> = _savedCount.asStateFlow()

    /** 设置页持久化的"自动放大"默认值：进取景页的初始开关（页内芯片仍可会话级临时改）。 */
    val autoZoomDefault: StateFlow<Boolean> = appPrefs.autoZoomDefault
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /** 性能监控叠加层开关（设置页）。 */
    val perfMonitor: StateFlow<Boolean> = appPrefs.perfMonitor
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    init {
        // 进相机就取一次新坐标：重排先验的"最近采样点"与记录坐标都吃它
        // （权限弹窗由 MainActivity 负责；未授权时静默失败，先验退月度均值）
        locationProvider.requestFresh {}
        // 离线区划索引预热（IO）：首次保存不等 3400 行解析
        viewModelScope.launch(Dispatchers.IO) { adminGeo.preload() }
        // 先验表预热（IO，2026-10-10）：原先首次 rerank 会在推理线程上同步解析约 860KB JSON
        viewModelScope.launch(Dispatchers.IO) { priorReranker.preload() }
    }

    private val _routingReason = MutableStateFlow(router.routingReason())

    /** 最新框选裁剪图（记录附件用），随推理帧更新。 */
    @Volatile private var lastCrop: Bitmap? = null

    // ── 帧率统计的滑动窗口计数（性能监控叠加层）──
    private var frameCount = 0
    private var fpsWindowStart = android.os.SystemClock.elapsedRealtime()

    /**
     * 记录快照（快门语义，2026-10-08 用户要求）：点「记录此鸟」的那一刻即冻结裁剪图/身份/候选。
     * 旧行为两处硬伤：① 弹窗期间推理继续跑，落库吃"选完瞬间"的 live lastCrop——鸟飞走
     *    后再点完候选，存的是漂移后的图甚至空地帧；② confirmedSci 点选结果根本没参与落库
     *    （speciesSci 取实时 identity），点第 3 候选存的是 top-1。
     * beginRecord 深拷贝：推理线程每帧替换 lastCrop，不拷贝等于没冻结；
     * 顺带消灭旧代码在 IO 线程遍历 live ewm（非线程安全 HashMap）的隐患。
     */
    data class RecordSnapshot(
        val crop: Bitmap?,
        val identity: TrackIdentity?,
        val candidates: List<TrackIdentity>,
        val lat: Double?,
        val lon: Double?,
    )

    @Volatile private var pendingRecord: RecordSnapshot? = null

    /** 「记录此鸟」按钮按下即调用；返回快照供弹窗做冻结展示（弹窗期间候选不再漂移）。
     *  经纬度也在这一刻定格（快门即定位），地址由落库时对冻结坐标做逆地理得到。 */
    fun beginRecord(): RecordSnapshot {
        val fix = locationProvider.lastKnown()
        return RecordSnapshot(
            crop = lastCrop?.copy(Bitmap.Config.ARGB_8888, false),
            identity = identity,
            candidates = _ui.value.candidates.toList(),
            lat = fix?.latitude,
            lon = fix?.longitude,
        ).also { pendingRecord = it }
    }

    /** 弹窗取消：丢弃快照，别让下次保存误吃旧快门。 */
    fun cancelRecord() {
        pendingRecord = null
    }

    val routingReason: StateFlow<String> = _routingReason.asStateFlow()

    /** 当前手动指定的识别区域（null = 自动 GPS 路由），供区域选择器显示。 */
    private val _manualRegion = MutableStateFlow(router.manualRegion())
    val manualRegion: StateFlow<String?> = _manualRegion.asStateFlow()

    /** 切换自动/手动区域；路由原因立即刷新，不等下一帧。 */

    /** 丢帧保命：上一帧没算完就直接跳过，保证预览不堆积。 */
    private val busy = AtomicBoolean(false)
    private var lastLogAt = 0L

    // 跟踪状态（B §5 P2：IoU 跟踪器接管记忆——平滑/速度预测/变焦重投影替代旧 EMA memoryDet）
    private val tracker = BirdTracker()
    @Volatile private var lastSeenAt = 0L
    private var scanPhase = 0 // 渐进扫描轮转相位：中心 → 2×2 宫格
    @Volatile private var pendingRedet: Detection? = null // 首次局部检出候选，待二次确认
    @Volatile private var zoomJoltUntil = 0L // 变焦后目标位置/尺度突变期，重检走渐进扫描
    @Volatile private var lastZoomRatio = 1f // 变焦重投影的增量基准

    // 身份聚合（docs/design/DESIGN.md §3.2）
    // @Volatile：beginRecord / saveCurrent 在主线程读它，写只在 stateDispatcher 上
    @Volatile private var identity: TrackIdentity? = null
    private val ewm = HashMap<String, Pair<com.whatsbird.vision.ClassScore, Float>>()
    private var lastAbsorbAt = 0L
    @Volatile private var lastClassifyAt = 0L
    private var lastEmitAt = 0L

    /** 切换识别区域 = 要求换模型重新识别：身份与聚合窗立即清空。 */
    fun setRegion(region: String?) {
        router.setManualRegion(region)
        _routingReason.value = router.routingReason()
        _manualRegion.value = router.manualRegion()
        // 清状态必须与 onFrame 同线程，否则会在 absorb 遍历 ewm 时被 clear 掉（见 stateDispatcher）
        viewModelScope.launch(stateDispatcher) {
            identity = null
            ewm.clear()
        }
    }

    // ===== 检测器档位（2026-10-10）=====
    // 起因：真机实测发现 auto 判错——标称 12GB 的小米 15，totalMem 只有 10.86 GiB，
    // 被 12 GiB 阈值判成低内存走中档，而中档 tiny 比旗舰低 30%（10x 小个体麻雀那一档）。
    // 阈值已改 9 GiB（ModelRegistry），设置页再给用户一个手动纠偏的口子。
    //
    // ⚠️ **切档入口只有设置页一个**（用户 2026-10-10 拍板）。本 ViewModel 因此**不再暴露
    //    detTier/setDetTier**——取景页那个芯片已删（它坏了地区下拉菜单的锚点，见 CameraScreen）。
    //    这里只保留一个职责：**被动**跟着偏好走，并在检测器真换了之后清掉旧跟踪状态。
    //    ⚠️ 少了这一步的实测后果：设置页切档时 `detector.switchTier` 已先执行并返回 false，
    //    若不比对生效资产，取景页会**留着按旧尺度建立的跟踪框与候选**继续用。

    init {
        viewModelScope.launch {
            appPrefs.detTier.collect { t ->          // 存储值已在 AppPrefs 侧做白名单校验
                val before = detector.activeAsset()
                detector.switchTier(t)
                val after = detector.activeAsset()
                if (after != before) {
                    // 换检测器 = 换尺度，旧跟踪框/候选作废（同样要上单线程，见 stateDispatcher）
                    viewModelScope.launch(stateDispatcher) {
                        identity = null
                        ewm.clear()
                        selectedTrackId = null
                    }
                    Log.i("CameraVM", "档位生效 $t -> $after，跟踪状态已清")
                }
            }
        }
    }

    // 多鸟：主目标轨道的选择状态
    @Volatile private var selectedTrackId: Int? = null

    /**
     * 当前是否还有空位接下一帧（2026-10-10 新增）。
     *
     * 供 CameraScreen 的 analyzer **先问再转码**用：相机 30fps 送帧、推理约 4fps 消费，
     * 原先 analyzer 无条件先 `toBitmap()` + `rotated()`（1080p 一帧约 8MB），
     * 之后才由 onFrame 判断 busy 并把这一帧丢掉 —— 30 帧里有 26 帧的转换是纯浪费，
     * 白白给 GC 加压。先问一句，忙就直接丢帧（proxy 仍由 finally 关闭）。
     *
     * 这只是**预筛**：检查与 onFrame 里的 CAS 之间仍可能被别人抢先，
     * 真正的互斥依旧由 onFrame 的 busy CAS 兜底。
     */
    fun frameSlotFree(): Boolean = !busy.get()

    /** 点选其他框切换主目标：换鸟 = 身份与聚合窗重算；新主目标已有缓存候选则先铺底。 */
    fun selectTrack(id: Int) {
        viewModelScope.launch(stateDispatcher) {
            selectedTrackId = id
            identity = null
            ewm.clear()
            tracker.trackById(id, android.os.SystemClock.elapsedRealtime())
                ?.candidates?.let { absorb(it) }
        }
    }

    /** 主目标：用户选定 > 确认轨迹 hits×score（BirdTracker.primary 原排序）。 */
    private fun pickPrimary(now: Long): BirdTracker.Track? =
        selectedTrackId?.let { tracker.trackById(it, now) } ?: tracker.primary(now)

    /**
     * 相机变焦后调用（带新倍率）：轨迹框围绕画面中心几何重投影（位置+尺度——
     * 退倍率后记忆框仍然有效），速度作废；重置待确认候选并进入渐进扫描期。
     * 捏合/滚轮动画会以小步长连续调用，增量 k 的连乘 = 总倍率变化，天然正确。
     */
    fun onZoomChanged(ratio: Float) {
        // tracker 是 onFrame 的共享状态，必须同线程（见 stateDispatcher）
        viewModelScope.launch(stateDispatcher) {
            val k = if (lastZoomRatio > 0f) ratio / lastZoomRatio else 1f
            if (k > 0f && k != 1f) tracker.onZoom(k)
            lastZoomRatio = ratio
            pendingRedet = null
            zoomJoltUntil = android.os.SystemClock.elapsedRealtime() + ZOOM_JOLT_MS
        }
    }

    /** 渐进式数字变焦扫描（B §5 P1）：中心 2× → 2×2 宫格，每帧一个相位轮转（~4fps 下约 1.2s 覆盖全画面）。 */
    private fun progressiveScan(src: Bitmap): Rescan? {
        val (lf, tf) = SCAN_PHASES[scanPhase++ % SCAN_PHASES.size]
        return pipeline.regionRescan(src, lf, tf, 0.5f)
    }

    /** 两个归一化框的交并比。 */
    private fun iou(a: Detection, b: Detection): Float {
        val ixL = maxOf(a.left, b.left); val iyT = maxOf(a.top, b.top)
        val ixR = minOf(a.right, b.right); val iyB = minOf(a.bottom, b.bottom)
        val iw = (ixR - ixL).coerceAtLeast(0f); val ih = (iyB - iyT).coerceAtLeast(0f)
        val inter = iw * ih
        val ua = (a.right - a.left) * (a.bottom - a.top) +
            (b.right - b.left) * (b.bottom - b.top) - inter
        return if (ua <= 0f) 0f else inter / ua
    }

    fun onFrame(src: Bitmap) {
        if (!busy.compareAndSet(false, true)) return
        // ── 帧率统计（性能监控叠加层用）：滑动窗口 1 秒，无需常驻协程 ──
        val ts = android.os.SystemClock.elapsedRealtime()
        frameCount++
        if (ts - fpsWindowStart >= 1000) {
            _ui.value = _ui.value.copy(
                fps = frameCount * 1000f / (ts - fpsWindowStart).coerceAtLeast(1)
            )
            frameCount = 0
            fpsWindowStart = ts
        }
        viewModelScope.launch(stateDispatcher) {
            try {
                val now = android.os.SystemClock.elapsedRealtime()
                // 高置信后不每帧分类，但定期复查（RECHECK_MS）：鸟换了才能被发现
                val skipClassify = identity != null && identity!!.prob >= CONFIDENT_PROB &&
                    now - lastClassifyAt < RECHECK_MS
                // 多鸟：分类/记录裁剪跟随主目标轨迹（上一帧位置），而非最高分检测
                val preTrack = pickPrimary(now)
                val r = pipeline.process(src, skipClassify = skipClassify, preferBox = preTrack?.box)
                r.crop?.let { lastCrop = it }

                var bestDet = r.detections.maxByOrNull { it.score }

                if (bestDet != null) {
                    tracker.update(r.detections, now)
                    lastSeenAt = now
                    pendingRedet = null
                    // 误检防线①：只有确认轨迹（≥CONFIRM_HITS 帧同位置）的分类才进聚合窗，
                    // 漂移的单帧 FP 不再一帧定名（装机 2026-10-07 桌面误检事故）
                    if (!skipClassify && pickPrimary(now)?.confirmed == true) absorb(r.top5)
                } else {
                    // 全帧检不到 → 记忆取主目标的预测框（速度外推，非静止旧框）
                    val mem = pickPrimary(now)?.box
                    if (mem != null) {
                        // 放远/缩小倍率后鸟太小 → 数字变焦重检；每帧都试（B §5 P1，不再隔帧节流）
                        if (now - lastSeenAt < REDETECT_WINDOW_MS) {
                            // 变焦突变期：渐进扫描（中心→宫格轮转）；稳定后预测框 2.5x 外扩重检
                            val rr = if (now < zoomJoltUntil) progressiveScan(src)
                            else pipeline.redetectInRegion(src, mem, 2.5f)
                            val d = rr?.frame?.detections?.maxByOrNull { it.score }
                            if (d != null) {
                                // 连续 2 帧同位置检出才确认（单帧误检丢弃）
                                lastCrop = rr.crop
                                val p = pendingRedet
                                if (p != null && iou(p, d) > 0.3f) {
                                    pendingRedet = null
                                    bestDet = d
                                    tracker.update(listOf(d), now) // 重获：并入轨迹复关联
                                    lastSeenAt = now
                                    // 重检裁剪是放大图：分类质量优于全帧小框裁剪（同样只认确认轨迹）
                                    if ((identity == null || identity!!.prob < CONFIDENT_PROB) &&
                                        pickPrimary(now)?.confirmed == true
                                    ) {
                                        absorb(pipeline.classifyCrop(rr.crop))
                                    }
                                } else {
                                    pendingRedet = d
                                }
                            }
                        }
                    } else {
                        // 无记忆框的全帧丢失（1x 远景初始获取）→ 渐进扫描轮转（B §5 P1）
                        val rr = progressiveScan(src)
                        val d = rr?.frame?.detections?.maxByOrNull { it.score }
                        if (d != null) {
                            bestDet = d
                            tracker.update(listOf(d), now)
                            lastSeenAt = now
                            lastCrop = rr.crop
                            // 误检防线①（同主路径）：只认确认轨迹
                            if ((identity == null || identity!!.prob < CONFIDENT_PROB) &&
                                pickPrimary(now)?.confirmed == true
                            ) {
                                absorb(pipeline.classifyCrop(rr.crop))
                            }
                        }
                    }
                }

                // 主目标（用户选定 > 确认轨迹 hits×score）与选定失效回退
                val primaryTrack = pickPrimary(now)
                if (primaryTrack == null && now - lastSeenAt >= REDETECT_WINDOW_MS) {
                    tracker.clear()
                    if (identity != null) {
                        identity = null
                        ewm.clear()
                    }
                }
                selectedTrackId?.let { sid ->
                    if (tracker.trackById(sid, now) == null) selectedTrackId = null
                }

                // 多鸟角标两段式验证（装机 16:35 事故：桌面 FP 首分类直通角标乱飞）：
                // ① 未分类轨迹首次分类 → 缓存候选；② 下次分类同种才置 labelVerified。
                // 桌面 FP 的分类结果逐次漂移（白尾鸲→乌鸫→黄眉柳莺），两验不一致永远无标。
                // 每帧预算仍 1 次；candidates 挂在 Track 上，轨迹移除随对象回收。
                tracker.snapshot(now)
                    .filter {
                        it !== primaryTrack && it.confirmed &&
                            !(it.candidates != null && it.labelVerified)
                    }
                    .minByOrNull { if (it.candidates == null) 0 else 1 }
                    ?.let { t ->
                        val t5 = pipeline.classifyCrop(pipeline.cropFor(src, t.box))
                        val old = t.candidates
                        if (old == null) {
                            t.candidates = t5
                        } else if (t5.firstOrNull()?.sci == old.firstOrNull()?.sci) {
                            t.labelVerified = true
                        } else {
                            t.candidates = t5 // 结果漂移：换新候选重新验证
                        }
                    }

                // 显示合成：当前框，或保鲜期内的预测框（1.5s 后框消失，搜索继续到 15s）
                val keep = now - lastSeenAt < MEMORY_HOLD_MS
                val displayDet = bestDet
                    ?: primaryTrack?.box?.takeIf { keep }

                emitUi(displayDet, primaryTrack, r.frameW, r.frameH, r.inferMs, now)
                _routingReason.value = router.routingReason()
                logHeartbeat(displayDet, skipClassify)
            } catch (t: Throwable) {
                Log.e(TAG, "推理失败", t)
            } finally {
                busy.set(false)
            }
        }
    }

    /** 分类结果进 EWM 聚合窗，按滞回与只升不降规则更新身份。 */
    private fun absorb(t5: List<com.whatsbird.vision.ClassScore>) {
        if (t5.isEmpty()) return
        val now = android.os.SystemClock.elapsedRealtime()
        lastClassifyAt = now
        val dt = (now - lastAbsorbAt).coerceIn(0L, 4000L)
        lastAbsorbAt = now
        val decay = 0.5f.pow(dt.toFloat() / EWM_HALF_LIFE_MS)
        for ((k, v) in ewm) ewm[k] = v.first to v.second * decay
        for (s in t5) {
            val old = ewm[s.sci]
            val best = if (old == null || s.prob > old.first.prob) s else old.first
            ewm[s.sci] = best to (old?.second ?: 0f) + s.prob
        }
        val ranked = ewm.values.sortedByDescending { it.second }
        val (top, score) = ranked.firstOrNull() ?: return
        val display = min(1f, score)
        val cur = identity
        identity = when {
            cur == null -> TrackIdentity(top.zh, top.sci, display, 1)
            top.sci == cur.sci ->
                // 只升不降：同种取历史最优置信度
                cur.copy(prob = max(cur.prob, display), hits = cur.hits + 1)
            // 换鸟判定：挑战者的当前聚合分超现身份种类的当前聚合分（非历史最高）
            // ×滞回系数——旧种不再被投票会自然衰减，鸟换了几秒内即换名
            score > (ewm[cur.sci]?.second ?: 0f) * HYSTERESIS + 0.02f ->
                TrackIdentity(top.zh, top.sci, display, 1)
            else -> cur
        }
    }

    /** UI 发射：500ms 节流，状态跃迁/换名立即发。 */
    private fun emitUi(
        displayDet: Detection?,
        primary: BirdTracker.Track?,
        frameW: Int,
        frameH: Int,
        inferMs: Float,
        now: Long,
    ) {
        // 定名双门（prob + 证据帧）对 IDENTIFIED 与 LOST 同时生效——
        // 装机视频 16:39 事故：未达标身份经 LOST（鸟已出画）分支直通面板
        //（鹊鸭 81%/黄鹡鸰 35%/黄眉柳莺 32% 全是 hits=1 的垃圾走此泄口）
        val qualified = identity != null && identity!!.prob >= CONFIDENT_PROB &&
            identity!!.hits >= MIN_IDENTIFY_HITS
        val state = when {
            displayDet != null && qualified -> IdState.IDENTIFIED
            displayDet != null -> IdState.TRACKING
            identity != null && tracker.hasTracks() && qualified -> IdState.LOST
            else -> IdState.IDLE
        }
        val ranked = ewm.values.sortedByDescending { it.second }
        val secondary = ranked.drop(1).take(2)
            .map { "${it.first.zh} ${"%.0f%%".format(min(1f, it.second) * 100)}" }
        // 充能条如实反映双门：概率门与证据帧门取较小进度
        val charging = min(
            (identity?.prob ?: 0f) / CONFIDENT_PROB,
            (identity?.hits ?: 0) / MIN_IDENTIFY_HITS.toFloat(),
        ).coerceIn(0f, 1f)
        // 多鸟框集合（含幽灵框防线③）：全部过显示门（轨迹均分 ≥
        // DISPLAY_MIN_SCORE）才画框，副目标另需确认轨迹。相机逻辑（追焦/自动放大）吃
        // 未过门的 detections，真鸟充能期间分数上来框自然出现。
        val others = tracker.snapshot(now)
            .filter { it !== primary && now - it.lastSeenAt < MEMORY_HOLD_MS }
            .filter { it.confirmed && it.score >= ModelRegistry.DISPLAY_MIN_SCORE }
            .sortedByDescending { it.score }
            .take(CORNER_MAX - 1)
        val boxes = buildList {
            if (displayDet != null &&
                (primary == null || primary.score >= ModelRegistry.DISPLAY_MIN_SCORE)
            ) {
                add(BoxTag(primary?.id ?: -1, displayDet, primary = true))
            }
            for (t in others) {
                // 角标双条件：两验一致 + 过 MIN_CLASS_PROB（漂移 FP 两验不一致，永不亮名）
                val top = t.candidates?.firstOrNull()
                    ?.takeIf { it.prob >= ModelRegistry.MIN_CLASS_PROB && t.labelVerified }
                add(BoxTag(t.id, t.box, primary = false, zh = top?.zh, prob = top?.prob ?: 0f))
            }
        }
        val next = ObserveUiState(
            state = state,
            identity = identity,
            candidates = ranked.take(5).map {
                TrackIdentity(it.first.zh, it.first.sci, min(1f, it.second), 0)
            },
            secondary = secondary,
            charging = charging,
            detections = listOfNotNull(displayDet),
            boxes = boxes,
            frameW = frameW,
            frameH = frameH,
            inferMs = inferMs,
        )
        val old = _ui.value
        val jump = state != old.state || identity?.sci != old.identity?.sci
        if (jump || now - lastEmitAt >= EMIT_INTERVAL_MS) {
            _ui.value = next
            lastEmitAt = now
        }
    }

    /** 每秒一条检测心跳，供 logcat 监听（不上屏）。 */
    private fun logHeartbeat(displayDet: Detection?, skipClassify: Boolean) {
        val now = android.os.SystemClock.elapsedRealtime()
        if (now - lastLogAt < 1000) return
        lastLogAt = now
        val top = identity
        Log.i(
            TAG,
            "det=${if (displayDet != null) 1 else 0}${if (skipClassify) "(held)" else ""} " +
                "top1=" + (top?.let { "${it.zh} ${"%.2f".format(it.prob)}" } ?: "-") +
                " ${_ui.value.state} infer=${_ui.value.inferMs.toInt()}ms " +
                "prior=${if (router.priorDegradeReason() == null) "on" else "off"} " +
                "${router.routingReason()}",
        )
    }

    /**
     * 保存记录（确认式）：UI 从候选弹窗传入用户点选的学名。
     * 2026-10-08 快门语义改造：一切取 beginRecord() 冻结的快照——裁剪图/候选/身份
     * 定格在按钮按下那一刻，弹窗里点谁就存谁（旧版点第 3 候选存的是实时 top-1）。
     * 附带：框选裁剪图、Top-5 候选 JSON、GPS 坐标、使用的区域模型。
     */
    fun saveCurrent(confirmedSci: String) {
        val snap = pendingRecord ?: beginRecord() // 未经按钮按下路径的兜底（正常流程不会走到）
        val picked = snap.candidates.firstOrNull { it.sci == confirmedSci }
            ?: snap.identity ?: return
        viewModelScope.launch(Dispatchers.IO) {
            var photoPath: String? = null
            snap.crop?.let { bmp ->
                try {
                    val dir = java.io.File(
                        appContext.filesDir, "records").apply { mkdirs() }
                    val f = java.io.File(dir, "obs_${System.currentTimeMillis()}.jpg")
                    f.outputStream().use { out ->
                        bmp.compress(Bitmap.CompressFormat.JPEG, 88, out)
                    }
                    photoPath = f.absolutePath
                } catch (t: Throwable) {
                    Log.w(TAG, "框选图保存失败", t)
                }
            }
            val candidatesJson = org.json.JSONArray().apply {
                snap.candidates.forEach {
                    put(org.json.JSONObject().apply {
                        put("sci", it.sci); put("zh", it.zh)
                        put("prob", it.prob)
                    })
                }
            }.toString()
            // 快门即定位（v3）：经纬度取按钮按下时冻结的快照；点击瞬间无坐标（冷启动首用）
            // 才回退保存时刻的缓存位置
            val fix = if (snap.lat != null && snap.lon != null) null else locationProvider.lastKnown()
            val lat = snap.lat ?: fix?.latitude
            val lon = snap.lon ?: fix?.longitude
            val address = if (lat != null && lon != null) reverseGeocode(lat, lon) else null
            observationDao.insert(
                ObservationEntity(
                    speciesSci = picked.sci,
                    speciesZh = picked.zh,
                    confidence = picked.prob,
                    lat = lat,
                    lon = lon,
                    observedAt = System.currentTimeMillis(),
                    candidates = candidatesJson,
                    confirmed = true,
                    photoPath = photoPath,
                    region = router.currentRegion(),
                    address = address,
                ),
            )
            pendingRecord = null
            _savedCount.value = observationDao.count()
        }
    }

    /**
     * 逆地理降级链（快门地址）：① 离线区划 AdminGeoIndex（永远先试：确定、免费、无网可用，
     * 粒度到区县，docs/guides/OFFLINE-ADMIN-GEO.md）→ ② 系统 Geocoder（GMS 机型白捡
     * 街道级；国产 ROM 无后端 isPresent()=false 直接跳过，零联网）。都拿不到返回 null，
     * 经纬度照常落库。刻意不走第三方在线地理服务——"数据仅存本机"的承诺到这一层成立。
     */
    private fun reverseGeocode(lat: Double, lon: Double): String? {
        adminGeo.locate(lat, lon)?.let { return it }
        return geocoderFallback(lat, lon)
    }

    private fun geocoderFallback(lat: Double, lon: Double): String? = try {
        if (!android.location.Geocoder.isPresent()) {
            null
        } else {
            val a = android.location.Geocoder(appContext, java.util.Locale.CHINA)
                .getFromLocation(lat, lon, 1)?.firstOrNull()
            a?.let {
                it.getAddressLine(0)?.takeIf { l -> l.isNotBlank() }
                    ?: listOfNotNull(it.adminArea, it.locality, it.subLocality, it.thoroughfare)
                        .joinToString("").takeIf { s -> s.isNotEmpty() }
            }
        }
    } catch (t: Throwable) {
        Log.d(TAG, "逆地理不可用（后端缺失/异常）：${t.message}")
        null
    }

    private companion object {
        const val TAG = "CameraVM"
        const val CONFIDENT_PROB = 0.35f    // 身份概率门（与证据帧门构成定名双门）
        const val MIN_IDENTIFY_HITS = 3     // 定名证据帧门：同种 ≥3 帧才进 IDENTIFIED（误检防线②）
        const val RECHECK_MS = 1500L        // 高置信后的复查间隔（鸟换了才能被发现）
        const val EWM_HALF_LIFE_MS = 1500L  // 聚合窗半衰期
        const val EMIT_INTERVAL_MS = 500L   // 显示节流
        const val HYSTERESIS = 1.15f        // 排名滞回：新第一名须超现名 15%
        const val MEMORY_HOLD_MS = 1500L     // 检测丢失后记忆框保鲜时长（仅显示）
        const val REDETECT_WINDOW_MS = 15000L // 丢失后重检搜索窗（B §5 P1：5s→15s，治退倍率即清身份）
        const val ZOOM_JOLT_MS = 3000L       // 变焦后位置突变期，重检改走渐进扫描
        const val CORNER_MAX = 5             // 多鸟同时展示框上限（主目标 1 + 角标 ≤4）
        val SCAN_PHASES = arrayOf(           // 渐进扫描轮转相位（regionRescan，regionFrac=0.5）
            0.25f to 0.25f,                  // 中心 2×
            0f to 0f, 0.5f to 0f, 0f to 0.5f, 0.5f to 0.5f, // 2×2 宫格
        )
    }
}

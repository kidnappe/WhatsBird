package com.whatsbird.vision

import kotlin.math.max
import kotlin.math.min

/**
 * 轻量 IoU 跟踪器（ByteTrack 思路的极简版：无外观特征、无 ReID）。
 *
 * 职责（替代旧 memoryDet + smooth() 的记忆逻辑）：
 *  ① 框 EMA 平滑（消除检测抖动）
 *  ② 恒速预测：丢失期间轨迹按速度外推（800ms 后冻结——鸟不会恒速太久），
 *     重检搜索窗跟着预测位置走，不再守在最后看到的位置
 *  ③ 变焦几何重投影：倍率变化时把轨迹框围绕画面中心按 k=newZoom/oldZoom 缩放，
 *     速度作废——退倍率后记忆框的**位置和大小**都仍然有意义（尺度门的替代实现）
 *  ④ 轨迹合并（v1.1）：两条确认轨迹持续高 IoU 共存 →
 *     判同一只鸟，保留更强一条——治一鸟多框（NMS 漏网双框各自建军 /
 *     丢而复返期新旧轨迹并存）
 *
 * 多目标：天然支持（每条检测独立成轨迹）；多鸟识别 UX 后续可直接消费 tracks。
 * 线程：仅 CameraViewModel 的推理协程调用，无需同步。
 */
class BirdTracker {

    class Track(
        val id: Int,
        @Volatile var box: Detection,   // 平滑后的框（归一化坐标）
        var vx: Float = 0f,             // 中心速度，归一化坐标/秒
        var vy: Float = 0f,
        var hits: Int = 0,              // 累计命中帧数（≥CONFIRM_HITS 为确认轨迹）
        var lastSeenAt: Long = 0,
        private var scoreSum: Float = 0f,
    ) {
        internal var predictedTo: Long = 0 // 外推已推进到的时间点（primary() 每帧多次调用须幂等）
        val confirmed get() = hits >= CONFIRM_HITS
        val score get() = if (hits > 0) scoreSum / hits else 0f
        /** 首次分类缓存的候选（多鸟识别：每条确认轨迹只分类一次；轨迹移除随对象回收）。 */
        var candidates: List<ClassScore>? = null
        /** 角标两验一致标记：第二次独立分类同种才置真（桌面 FP 的分类逐次漂移，永不过验）。 */
        var labelVerified: Boolean = false
        fun addHit(d: Detection, now: Long) {
            hits++; scoreSum += d.score; lastSeenAt = now; predictedTo = 0
        }
        /** 轨迹合并（v1.1）：this 吞并 sub——同一只鸟的双框收敛成一条。
         *  帧数/分数累计并入（score 变加权平均，primary 排序吃合并证据），
         *  候选缓存与两验标记择优保留（"每鸟只分类一次"原则不破坏）。
         *  v1.2：两验标记不跨种继承——sub 的"验证"是对 sub 自己种名的，
         *  并入后若种名不同，验证不作数（防错误标签借壳上屏）。 */
        internal fun absorb(sub: Track) {
            hits += sub.hits
            scoreSum += sub.scoreSum
            lastSeenAt = maxOf(lastSeenAt, sub.lastSeenAt)
            val keepTop = candidates?.firstOrNull()?.sci
            val subTop = sub.candidates?.firstOrNull()?.sci
            if (candidates == null) candidates = sub.candidates
            if (subTop != null && subTop == keepTop) {
                labelVerified = labelVerified || sub.labelVerified
            }
        }
    }

    private val tracks = ArrayList<Track>()
    private var nextId = 1
    /** 合并判据的连续共存帧计数（键 = 有序 id 对）。 */
    private val mergeStreak = HashMap<Long, Int>()

    /** 喂一帧检测（主路径全帧结果，或重检确认后的单框）。 */
    fun update(dets: List<Detection>, now: Long) {
        tracks.removeAll { now - it.lastSeenAt > TRACK_TTL_MS }
        for (t in tracks) predictTo(t, now)

        // 贪心匹配：预测框 vs 检测框，按 IoU 降序配对（轨迹数/检测数都是个位数，O(n²) 足够）
        val pairs = ArrayList<FloatArray>(tracks.size * dets.size) // [iou, trackIdx, detIdx]
        for (ti in tracks.indices) for (di in dets.indices) {
            val v = iou(tracks[ti].box, dets[di])
            if (v >= MATCH_IOU) pairs.add(floatArrayOf(v, ti.toFloat(), di.toFloat()))
        }
        pairs.sortByDescending { it[0] }
        val usedT = BooleanArray(tracks.size)
        val usedD = BooleanArray(dets.size)
        for (p in pairs) {
            val ti = p[1].toInt(); val di = p[2].toInt()
            if (usedT[ti] || usedD[di]) continue
            usedT[ti] = true; usedD[di] = true
            val t = tracks[ti]
            val d = dets[di]
            val old = t.box
            val dt = ((now - t.lastSeenAt) / 1000f).coerceIn(0.001f, 2f)
            val nvx = ((d.left + d.right - old.left - old.right) / 2f) / dt
            val nvy = ((d.top + d.bottom - old.top - old.bottom) / 2f) / dt
            if (t.hits >= CONFIRM_HITS) { // 试探轨迹先吸收位置，速度等确认后再估（首帧配对噪声大）
                t.vx = t.vx + (nvx - t.vx) * VEL_A
                t.vy = t.vy + (nvy - t.vy) * VEL_A
            }
            t.box = Detection(
                score = d.score,
                left = old.left + (d.left - old.left) * SMOOTH_A,
                top = old.top + (d.top - old.top) * SMOOTH_A,
                right = old.right + (d.right - old.right) * SMOOTH_A,
                bottom = old.bottom + (d.bottom - old.bottom) * SMOOTH_A,
                cocoClass = d.cocoClass,
            )
            t.addHit(d, now)
        }
        for (di in dets.indices) {
            if (!usedD[di]) tracks.add(Track(nextId++, dets[di], lastSeenAt = now).apply { addHit(dets[di], now) })
        }
        mergePass()
    }

    /**
     * 轨迹间合并（v1.1 治一鸟多框）。update() 原本只有"轨迹↔检测"匹配，
     * 两条同鸟轨迹（NMS 漏网双框各自建军 / 丢而复返期新旧并存）能长期共存。
     * 判据（保守起步）：双方均已确认 + IoU ≥ MERGE_IOU 连续 MERGE_STREAK_FRAMES 帧
     * （单帧贴近的真邻居两只不合，抖动即清零）；保险：双方都已有候选缓存且
     * top1 种名不同 → 不合（不同种证据优先，宁留双框不误并两只真鸟）。
     * 保留更强一条（hits×score），弱者 hits/分数/候选/两验标记并入后移除。
     */
    private fun mergePass() {
        if (tracks.size < 2) { mergeStreak.clear(); return }
        val seen = HashSet<Long>()
        val doomed = ArrayList<Track>()
        for (i in tracks.indices) for (j in i + 1 until tracks.size) {
            val a = tracks[i]; val b = tracks[j]
            if (a in doomed || b in doomed) continue
            if (!a.confirmed || !b.confirmed) continue
            val key = pairKey(a.id, b.id)
            seen.add(key)
            val iv = iou(a.box, b.box)
            if (iv < MERGE_IOU) {
                // 一鸟多框排查诊断：贴得够近却低于合并闸的对，看是闸高还是抖动（streak 清零）
                if (iv >= MERGE_DIAG_IOU) android.util.Log.i(
                    "BirdTracker", "skip t${a.id}/t${b.id} iou=%.2f<gate %.2f a[${desc(a)}] b[${desc(b)}]"
                        .format(iv, MERGE_IOU))
                mergeStreak.remove(key); continue
            }
            val ca = a.candidates?.firstOrNull()
            val cb = b.candidates?.firstOrNull()
            // 保险（v1.2 收紧）：只有「双方两验一致 + 双方 top1
            // 都过 VETO_PROB」才够格判两只真鸟。野测反例（v6 t=36）：同一只麻雀三个框
            // 三种名（树鹨43/燕雀55/棕头鸦雀49），每个都"两验一致"——远距小鸟的 crop
            // 会稳定地错认，低置信两验没有否决权；否则合并被永久锁死，一鸟多框不治。
            val twoRealBirds = a.labelVerified && b.labelVerified &&
                (ca?.prob ?: 0f) >= VETO_PROB && (cb?.prob ?: 0f) >= VETO_PROB
            if (ca != null && cb != null && ca.sci != cb.sci && twoRealBirds) continue
            val n = (mergeStreak[key] ?: 0) + 1
            if (n < MERGE_STREAK_FRAMES) { mergeStreak[key] = n; continue }
            val (keep, sub) = if (a.hits * a.score >= b.hits * b.score) a to b else b to a
            keep.absorb(sub)
            android.util.Log.i("BirdTracker", "merge t${keep.id}<-t${sub.id} iou=" +
                "%.2f".format(iou(keep.box, sub.box)))
            doomed.add(sub)
            mergeStreak.remove(key)
        }
        if (doomed.isNotEmpty()) tracks.removeAll(doomed.toSet())
        mergeStreak.keys.retainAll(seen) // 轨迹对消失即清计数
    }

    /** 诊断用轨迹摘要：命中/均分/两验 + top1 候选（一鸟多框时看双方证据强弱）。 */
    private fun desc(t: Track): String {
        val c = t.candidates?.firstOrNull()
        val label = c?.let { " ${it.sci}=${"%.2f".format(it.prob)}" } ?: ""
        return "h${t.hits} s=${"%.2f".format(t.score)}${if (t.labelVerified) " V" else ""}$label"
    }

    private fun pairKey(x: Int, y: Int): Long =
        if (x < y) x.toLong() * 1_000_000L + y else y.toLong() * 1_000_000L + x

    /**
     * 主目标：确认轨迹按 hits×score 取最优；没有确认轨迹时允许试探轨迹顶替
     * （单鸟场景首帧即有框可显示，与旧"首帧直出"行为对齐）。
     */
    fun primary(now: Long): Track? {
        if (tracks.isEmpty()) return null
        for (t in tracks) predictTo(t, now)
        val confirmed = tracks.filter { it.confirmed }
        val pool = if (confirmed.isNotEmpty()) confirmed else tracks
        return pool.maxByOrNull { it.hits.toFloat() * it.score + it.hits * 0.01f }
    }

    /** 全部轨迹的存活情况：返回最近一条轨迹的 lastSeenAt（无轨迹返回 Long.MIN_VALUE）。 */
    fun lastAnySeenAt(): Long = tracks.maxOfOrNull { it.lastSeenAt } ?: Long.MIN_VALUE

    fun hasTracks(): Boolean = tracks.isNotEmpty()

    /** 按 id 取轨迹（多鸟：用户手动选定主目标）；外推后返回，无则 null。 */
    fun trackById(id: Int, now: Long): Track? {
        for (t in tracks) predictTo(t, now)
        return tracks.firstOrNull { it.id == id }
    }

    /** 全部轨迹快照（外推后拷贝，供 UI 展示多个框）。 */
    fun snapshot(now: Long): List<Track> {
        for (t in tracks) predictTo(t, now)
        return tracks.toList()
    }

    /** 变焦重投影：k = newZoom/oldZoom。中心偏移按 k 缩放、框尺寸按 k 缩放、速度清零。 */
    fun onZoom(k: Float) {
        if (k <= 0f || k == 1f) return
        for (t in tracks) {
            val cx = 0.5f + ((t.box.left + t.box.right) / 2f - 0.5f) * k
            val cy = 0.5f + ((t.box.top + t.box.bottom) / 2f - 0.5f) * k
            val w = ((t.box.right - t.box.left) * k).coerceAtLeast(0.004f) // ≥~8px@1080p，防缩没
            val h = ((t.box.bottom - t.box.top) * k).coerceAtLeast(0.004f)
            t.box = Detection(
                score = t.box.score,
                left = (cx - w / 2f).coerceIn(0f, 1f),
                top = (cy - h / 2f).coerceIn(0f, 1f),
                right = (cx + w / 2f).coerceIn(0f, 1f),
                bottom = (cy + h / 2f).coerceIn(0f, 1f),
                cocoClass = t.box.cocoClass,
            )
            t.vx = 0f; t.vy = 0f
        }
    }

    fun clear() { tracks.clear(); mergeStreak.clear() }

    /** 把轨迹外推到 now（丢失期恒速漂移，超过冻结窗后停住；从 predictedTo 起算保证幂等）。 */
    private fun predictTo(t: Track, now: Long) {
        val from = maxOf(t.lastSeenAt, t.predictedTo)
        if (now <= from) return
        val moveMs = min(now - from, COAST_FREEZE_MS)
        if (moveMs <= 0) {
            t.predictedTo = now
            return
        }
        val dx = t.vx * moveMs / 1000f
        val dy = t.vy * moveMs / 1000f
        t.box = t.box.copy(
            left = (t.box.left + dx).coerceIn(0f, 1f),
            top = (t.box.top + dy).coerceIn(0f, 1f),
            right = (t.box.right + dx).coerceIn(0f, 1f),
            bottom = (t.box.bottom + dy).coerceIn(0f, 1f),
        )
        t.predictedTo = from + moveMs
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix = (min(a.right, b.right) - max(a.left, b.left)).coerceAtLeast(0f)
        val iy = (min(a.bottom, b.bottom) - max(a.top, b.top)).coerceAtLeast(0f)
        val inter = ix * iy
        val ua = (a.right - a.left) * (a.bottom - a.top) +
            (b.right - b.left) * (b.bottom - b.top) - inter
        return if (ua > 0f) inter / ua else 0f
    }

    companion object {
        const val CONFIRM_HITS = 2        // 连续命中 2 帧升为确认轨迹（与旧 pendingRedet 两帧确认同口径）
        const val SMOOTH_A = 0.5f         // 框 EMA（比旧 0.35 跟手：预测已吸收运动，平滑只需消抖）
        const val VEL_A = 0.4f            // 速度 EMA
        const val MATCH_IOU = 0.2f        // 预测加持下匹配门槛低于静态 0.3
        const val MERGE_IOU = 0.35f       // 轨迹合并门槛：实测同鸟重复框 IoU 带 0.32~0.43（屏摄
                                          // 嵌套/松紧双框），0.5 时只有抖动碰巧顶过才并得上
                                          //（multibox_capture.log：红脚隼 t2/t3 三次 skip、
                                          // 翠鸟 t6/t13 靠 0.74 收敛运气合并）；真邻居两鸟
                                          // 典型 IoU<0.2，0.35 + 3 帧连击 + 异种保险仍留足保护
        const val VETO_PROB = 0.6f        // 异种否决的置信门：双方 top1 都过线才判两只真鸟
                                          //（野测 43~55% 的稳定错认不配否决，见 §九）
        const val MERGE_DIAG_IOU = 0.25f  // 诊断日志下限：贴得比这还近的轨迹对必属同鸟/误检纠缠
        const val MERGE_STREAK_FRAMES = 3 // 连续共存帧数：抖动即清零，稳定双框 3 帧内必合
        const val COAST_FREEZE_MS = 800L  // 丢失超过此时长停止外推
        const val TRACK_TTL_MS = 15_000L  // 轨迹保活，与 REDETECT_WINDOW_MS 对齐（供重检复关联）
    }
}

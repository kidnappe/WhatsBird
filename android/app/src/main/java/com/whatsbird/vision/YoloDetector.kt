package com.whatsbird.vision

import android.content.Context
import android.graphics.Bitmap
import dagger.hilt.android.qualifiers.ApplicationContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.round

/**
 * YOLOX 检测器（路线 B 单类 bird）。
 *
 * 资产 yolox_bird_640_best_int8.tflite：两步法量化（onnx2tf float32 → ai_edge_quantizer int8），
 * 权重+激活全 int8、float32 输入输出；PC 验收 @th0.45 recall -0.2pp。
 * 喂法（⚠️ 与常见 YOLO 口径不同，已实测确认）：
 *   letterbox 灰边 114（右下补齐，非居中）+ BGR 通道序 + 原始 0-255，不归一化。
 * 输出 [1,6,N]（flatbuilder 已转置，与旧 [1,85,N] 同构），通道 = [reg4(xy,wh), obj, cls1]，
 * N 按 P3(stride8)→P4(16)→P5(32) 行主序（640 → 6400+1600+400 = 8400）；
 * decode：cx=(raw0+gx)*st，cy=(raw1+gy)*st，w=exp(raw2)*st，h=exp(raw3)*st，bird = cls0。
 *
 * ## 双档 + UI 换挡（2026-10-10）
 *
 * 旗舰 640（RAM 够）/ 中档 512（其余），可在设置页三选一覆盖（见 `ModelRegistry.detectorAssetFor`）。
 * **一次只驻留一档**：所有与模型绑定的状态（interpreter、输入缓冲、锚点、输出缓冲）整体装进
 * [Loaded]，切档 = 整体换掉并 close 旧的；新档**不预载**，等下一帧 detect 时才载入，
 * 避免切档瞬间在 UI 线程上卡几百毫秒。
 */
@Singleton
class YoloDetector @Inject constructor(
    @ApplicationContext private val context: Context,
) : BirdDetector {

    /** ⚠️ `ActivityManager.totalMem` **不是物理内存**。
     * 实测小米15（标称 12GB）：/proc/meminfo MemTotal=11388316kB，两者其实相同，
     * 但都低于标称值；Android 侧再经 `shr 30` 截断成 GiB 会**丢掉小数**
     * （10.86 → 打出来是 10）。**别把它当"手机多少G内存"对外说。** */
    private val ramBytes = totalMemBytes(context)

    private fun totalMemBytes(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val mi = android.app.ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        return mi.totalMem
    }

    @Volatile private var tierPref: String = ModelRegistry.TIER_AUTO
        private set

    /** 一次载入所需的全部模型派生状态。换档 = 整个换掉，绝不两档并存。 */
    private class Loaded(
        val asset: String,
        val interpreter: org.tensorflow.lite.Interpreter,
        val inputShape: IntArray,
        val size: Int,
        val anchors: Int,
        val inputBuf: ByteBuffer,
        val inputFloats: java.nio.FloatBuffer,
        val outBox: Array<Array<FloatArray>>,
        val outputs: Map<Int, Array<Array<FloatArray>>>,
        val gridX: IntArray,
        val gridY: IntArray,
        val gridStride: FloatArray,
    )

    @Volatile private var L: Loaded? = null

    private val busy = AtomicBoolean(false)
    private var lastMaxScore = 0f
    private var framesLogged = 0

    init {
        android.util.Log.i(
            TAG,
            "检测器就绪 pref=${tierPref} totalMem=${ramBytes shr 30}GiB(${ramBytes}B) " +
                "将载入=${ModelRegistry.detectorAssetFor(tierPref, ramBytes)}",
        )
    }

    override fun tier(): String = tierPref

    override fun totalMemBytes(): Long = ramBytes

    override fun activeAsset(): String =
        L?.asset ?: ModelRegistry.detectorAssetFor(tierPref, ramBytes)

    override fun switchTier(tier: String): Boolean {
        val v = when (tier) {
            ModelRegistry.TIER_FLAG, ModelRegistry.TIER_MID -> tier
            else -> ModelRegistry.TIER_AUTO
        }
        // ⚠️ 判据只能是**偏好变没变**，绝不能看 L 有没有载入。
        //   初版写成 `if (tierPref == v && L?.asset == want) return false`，
        //   于是「L 还没载入（L==null）」会被误判成"档位不同" →
        //   ① 启动时白打一条「切档 pref=auto」的假日志；② DataStore 每次发射都触发一次
        //   真切换，实测**一次用户操作打三条日志**。已按"偏好即状态"改正。
        if (tierPref == v) return false
        val want = ModelRegistry.detectorAssetFor(v, ramBytes)
        // ⚠️ 2026-10-10 修复：原先先 CAS busy，失败就直接 return false 且**不记 tierPref**，
        //   于是「推理进行中用户切档」这一下被静默丢弃，而 DataStore 不会重发 → 永远不生效。
        //   现在先把偏好落下来（绝不丢用户意图），再等锁卸载旧模型。
        tierPref = v
        // 与 detect() 互斥：等锁而不是丢弃。单帧推理最长约几百毫秒，2s 足够。
        if (!acquireBusy(2000)) {
            // 等不到也不能让旧模型赖着不走：loaded() 会发现 L 与 tierPref 不符而自行重载。
            android.util.Log.w(TAG, "切档等锁超时 pref=$v -> $want，旧档仍在跑，本帧不卸载")
            return true
        }
        try {
            L?.interpreter?.close()
            L = null                      // 新档不预载，下一帧 detect 时按需载入
            android.util.Log.i(TAG, "切档 pref=$v -> $want（旧档已卸载，新档待载入）")
            return true
        } finally {
            busy.set(false)
        }
    }

    /** 有界等待 busy 锁。返回 false 表示超时（调用方必须自行决定降级行为）。 */
    private fun acquireBusy(timeoutMs: Long): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (!busy.compareAndSet(false, true)) {
            if (System.currentTimeMillis() >= deadline) return false
            try {
                Thread.sleep(10)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                return false
            }
        }
        return true
    }

    /** 按需载入当前档位；已载入则直接返回。 */
    private fun loaded(): Loaded {
        // ⚠️ 2026-10-10：原先只看 `L != null` 就复用，切档等锁超时后旧模型会一直赖着
        //   （tierPref 已是新档）。改为比对资产名，不符即卸载重载——自愈，且只在持有 busy 时调用。
        val want = ModelRegistry.detectorAssetFor(tierPref, ramBytes)
        L?.let { if (it.asset == want) return it }
        synchronized(this) {
            val want2 = ModelRegistry.detectorAssetFor(tierPref, ramBytes)
            L?.let { if (it.asset == want2) return it }
            L?.interpreter?.close()
            val built = load(want2)
            L = built
            return built
        }
    }

    private fun load(asset: String): Loaded {
        val it = org.tensorflow.lite.Interpreter(
            loadModel(context, asset),
            org.tensorflow.lite.Interpreter.Options().setNumThreads(4),
        )
        val inputShape = it.getInputTensor(0).shape()      // [1,S,S,3]
        val size = inputShape[2]
        val buf = ByteBuffer.allocateDirect(size * size * 3 * 4).order(ByteOrder.nativeOrder())
        val anchors = (0 until 3).sumOf { (size / ANCHOR_STRIDES[it]).let { h -> h * h } }
        val outBox = Array(1) { Array(6) { FloatArray(anchors) } }
        val gridX = IntArray(anchors)
        val gridY = IntArray(anchors)
        val gridStride = FloatArray(anchors)
        check(inputShape[1] == size && inputShape[3] == 3) { "意外输入规格 $inputShape" }
        // 输出规格校验（2026-10-10 补）：outBox 硬编码 [1][6][anchors]，此前**只校验了输入**。
        // 换回退役的 A 路线 85 通道模型（该文件仍在 assets 作回退资产）会静默错解 →
        // 类名全错且无任何报错。实读两个发货模型：640=[1,6,8400]、512=[1,6,5376]，
        // 与 ANCHOR_STRIDES=[8,16,32] 推出的 anchors 逐位相等，故本检查必然通过。
        val outShape = it.getOutputTensor(0).shape()
        check(outShape[0] == 1 && outShape[1] == 6 && outShape[2] == anchors) {
            "意外输出规格 ${outShape.toList()}（期望 [1, 6, $anchors]）：解码口径对不上，会静默错解"
        }
        var n = 0
        for (lv in ANCHOR_STRIDES.indices) {
            val st = ANCHOR_STRIDES[lv]
            val hw = size / st
            for (y in 0 until hw) for (x in 0 until hw) {
                gridX[n] = x
                gridY[n] = y
                gridStride[n] = st.toFloat()
                n++
            }
        }
        check(n == anchors)
        val l = Loaded(
            asset = asset,
            interpreter = it,
            inputShape = inputShape,
            size = size,
            anchors = anchors,
            inputBuf = buf,
            inputFloats = buf.asFloatBuffer(),
            outBox = outBox,
            outputs = mapOf(0 to outBox),
            gridX = gridX,
            gridY = gridY,
            gridStride = gridStride,
        )
        android.util.Log.i(
            TAG,
            "档位已载入 asset=$asset size=${l.size}x${l.size} anchors=${l.anchors} " +
                "pref=$tierPref",
        )
        selfTest(l)
        return l
    }

    /** 启动自检：用内置已知鸟图跑完整管线，验证设备端运行时（与 PC 基准比对）。 */
    private fun selfTest(l: Loaded) {
        try {
            val bmp = context.assets.open(SELFTEST_ASSET).use {
                android.graphics.BitmapFactory.decodeStream(it)
            }
            val ratio = fillInput(l, bmp)
            l.interpreter.runForMultipleInputsOutputs(arrayOf(l.inputBuf), l.outputs)
            val dets = decode(l, ratio, bmp.width.toFloat(), bmp.height.toFloat())
            android.util.Log.i(
                TAG,
                "selftest ${bmp.width}x${bmp.height} maxScore=$lastMaxScore " +
                    "dets=${dets.size} " + dets.take(3).joinToString(" ") {
                        "s=%.2f l=%.2f t=%.2f r=%.2f b=%.2f".format(
                            it.score, it.left, it.top, it.right, it.bottom,
                        )
                    },
            )
        } catch (e: Exception) {
            android.util.Log.e(TAG, "selftest failed", e)
        }
    }

    override fun detect(src: Bitmap): List<Detection> {
        // ⚠️ 2026-10-10：原先 busy 抢不到就 return emptyList()，而 emptyList 与「真的没检出」
        //   对调用方完全同形 → 并发重检（regionRescan/redetect）会被当成画面里没有鸟，
        //   轨迹据此错误结束。等锁优先；真等不到才降级并打日志，不再静默装作"无检出"。
        if (!acquireBusy(2000)) {
            android.util.Log.w(TAG, "detect 等锁超时（并发重检未让出），本帧按无检出降级")
            return emptyList()
        }
        try {
            val l = loaded()
            val ratio = fillInput(l, src)
            l.interpreter.runForMultipleInputsOutputs(arrayOf(l.inputBuf), l.outputs)
            val dets = decode(l, ratio, src.width.toFloat(), src.height.toFloat())
            if (framesLogged < 5 || framesLogged % SAMPLE_LOG_EVERY == 0) {
                framesLogged++
                android.util.Log.i(
                    TAG,
                    "camera frame $framesLogged ${src.width}x${src.height} " +
                        "maxScore=$lastMaxScore dets=${dets.size} tier=${l.asset}",
                )
            } else {
                framesLogged++
            }
            return dets
        } finally {
            busy.set(false)
        }
    }

    /** letterbox 填充，返回缩放比（输出框 ÷ ratio 映射回原图）。 */
    private fun fillInput(l: Loaded, src: Bitmap): Float {
        val size = l.size
        val ratio = min(size.toFloat() / src.height, size.toFloat() / src.width)
        val nw = round(src.width * ratio).toInt().coerceAtMost(size)
        val nh = round(src.height * ratio).toInt().coerceAtMost(size)
        val scaled = Bitmap.createScaledBitmap(src, nw, nh, true)
        val px = IntArray(nw * nh)
        scaled.getPixels(px, 0, nw, 0, 0, nw, nh)
        // 背景填 114（灰边，BGR 同值）
        val f = l.inputFloats
        f.rewind()
        for (i in 0 until f.capacity()) f.put(PAD_VALUE)
        for (y in 0 until nh) {
            var idx = y * size * 3
            for (x in 0 until nw) {
                val p = px[y * nw + x]
                f.put(idx, (p and 0xFF).toFloat())          // BGR 通道序
                f.put(idx + 1, ((p shr 8) and 0xFF).toFloat())
                f.put(idx + 2, ((p shr 16) and 0xFF).toFloat())
                idx += 3
            }
        }
        l.inputBuf.rewind()
        return ratio
    }

    private fun decode(l: Loaded, ratio: Float, srcW: Float, srcH: Float): List<Detection> {
        val chan = l.outBox[0]
        val nSrcW = srcW * ratio // 原图→输入 的坐标换算分母
        val nSrcH = srcH * ratio
        val dets = ArrayList<Detection>(16)
        var maxScore = 0f
        for (i in 0 until l.anchors) {
            val score = chan[4][i] * chan[BIRD_CLS_OFFSET][i]
            if (score > maxScore) maxScore = score
            if (score < ModelRegistry.MIN_DET_SCORE) continue
            val st = l.gridStride[i]
            val cx = (chan[0][i] + l.gridX[i]) * st
            val cy = (chan[1][i] + l.gridY[i]) * st
            val w = exp(chan[2][i]) * st
            val h = exp(chan[3][i]) * st
            val x0 = ((cx - w / 2f) / nSrcW).coerceIn(0f, 1f)
            val y0 = ((cy - h / 2f) / nSrcH).coerceIn(0f, 1f)
            val x1 = ((cx + w / 2f) / nSrcW).coerceIn(0f, 1f)
            val y1 = ((cy + h / 2f) / nSrcH).coerceIn(0f, 1f)
            if (x1 <= x0 || y1 <= y0) continue
            dets.add(Detection(score, x0, y0, x1, y1, ModelRegistry.BIRD_CLASS_ID))
        }
        dets.sortByDescending { it.score }
        val keep = ArrayList<Detection>(dets.size)
        for (d in dets) {
            if (keep.none { iou(it, d) >= NMS_IOU }) keep.add(d)
        }
        lastMaxScore = maxScore
        return keep
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix = (min(a.right, b.right) - maxOf(a.left, b.left)).coerceAtLeast(0f)
        val iy = (min(a.bottom, b.bottom) - maxOf(a.top, b.top)).coerceAtLeast(0f)
        val inter = ix * iy
        val ua = (a.right - a.left) * (a.bottom - a.top) +
            (b.right - b.left) * (b.bottom - b.top) - inter
        return if (ua > 0f) inter / ua else 0f
    }

    private companion object {
        const val TAG = "YoloDetector"
        const val SELFTEST_ASSET = "models/selftest_bird.jpg"
        const val PAD_VALUE = 114f
        val ANCHOR_STRIDES = intArrayOf(8, 16, 32)
        const val BIRD_CLS_OFFSET = 5 + 0 // obj 在 4，B 路线单类 bird 是下标 0
        const val NMS_IOU = 0.45f
        const val SAMPLE_LOG_EVERY = 15 // 处理帧率 ~4fps → 约 4s 一条采样日志
    }
}
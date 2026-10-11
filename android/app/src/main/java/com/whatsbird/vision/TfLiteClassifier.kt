package com.whatsbird.vision

import android.content.Context
import android.graphics.Bitmap
import dagger.hilt.android.qualifiers.ApplicationContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * TFLite 分类器（通用版）：资产/labels/类数全部参数化，
 * 全国 373 类与各区域模型共用同一实现。
 * 预处理与 poc_verify.py 逐位一致：短边 256 → 居中裁 224 → ImageNet 归一化。
 */
class TfLiteClassifier @Inject constructor(
    @ApplicationContext context: Context,
    assetPath: String,
    labelsPath: String,
    private val numSpecies: Int,
    tensorDim: Int = numSpecies, // 张量输出维（老 373 模型是 384，有效类 373）
) : BirdClassifier {

    private val interpreter = org.tensorflow.lite.Interpreter(
        loadModel(context, assetPath),
        org.tensorflow.lite.Interpreter.Options().setNumThreads(4),
    )

    private val labels: List<Pair<String, String>> = loadLabels(context, labelsPath)

    private val inBuf = ByteBuffer
        .allocateDirect(INPUT * INPUT * 3 * 4)
        .order(ByteOrder.nativeOrder())
    // TFLite 要求 Java 侧形状与张量逐维一致：[1,N] → Array(1){FloatArray(N)}
    private val logitsOut = Array(1) { FloatArray(tensorDim) }

    init {
        // 启动自检（把「华中 145」教训机制化）：labels 行数、注册表类数、
        // 张量真实输出维三者不一致 → 打红字。运行期的形状异常会被 Router 的
        // catch(Throwable) 吞成「该区域静默全废」，初始化期是唯一能喊出来的时机——
        // 降级可以，降级必须可见。
        val outDim = runCatching { interpreter.getOutputTensor(0).shape().last() }.getOrDefault(-1)
        if (labels.size != numSpecies || outDim != tensorDim) {
            android.util.Log.e(
                "TfLiteClassifier",
                "LABELS/DIM 不一致，该分类可能静默降级: $assetPath " +
                    "labels=${labels.size} 注册类数=$numSpecies " +
                    "张量输出=$outDim 注册张量维=$tensorDim",
            )
        }
    }

    override fun classify(src: Bitmap): List<ClassScore> = synchronized(this) {
        // ⚠️ 2026-10-10：crop 每帧新建（centerCrop224 里还另有 scaled 中间图），
        // 原先从不回收 → 每帧白扔 1~2 张 224px 位图，GC 压力大。
        // 这里负责回收 crop；scaled 在 centerCrop224 内部回收。
        val crop = centerCrop224(src)
        try {
            fillInput(crop)
            interpreter.run(inBuf, logitsOut)
            val logits = logitsOut[0]

            val n = numSpecies
            var max = Float.NEGATIVE_INFINITY
            for (i in 0 until n) if (logits[i] > max) max = logits[i]
            val probs = FloatArray(n)
            var sum = 0.0
            for (i in 0 until n) {
                probs[i] = exp(logits[i] - max)
                sum += probs[i]
            }

            (0 until n)
                .sortedByDescending { probs[it] }
                .take(TOP_K)
                .map { i ->
                    val (sci, zh) = labels.getOrElse(i) { "?" to "?" }
                    ClassScore(i, sci, zh, (probs[i] / sum).toFloat())
                }
        } finally {
            if (!crop.isRecycled) crop.recycle()
        }
    }

    private fun centerCrop224(src: Bitmap): Bitmap {
        val scale = 256f / min(src.width, src.height)
        val scaled = Bitmap.createScaledBitmap(
            src,
            (src.width * scale).roundToInt().coerceAtLeast(INPUT),
            (src.height * scale).roundToInt().coerceAtLeast(INPUT),
            true,
        )
        // ⚠️ 2026-10-10：createScaledBitmap 在尺寸已等于目标时可能**返回 src 本身**，
        // 那就不能 recycle（会连调用方的图一起废掉）。
        val cropped = Bitmap.createBitmap(scaled, ((scaled.width - INPUT) / 2).coerceAtLeast(0),
            ((scaled.height - INPUT) / 2).coerceAtLeast(0), INPUT, INPUT)
        if (scaled !== cropped && !scaled.isRecycled) scaled.recycle()
        return cropped
    }

    private fun fillInput(bmp: Bitmap) {
        inBuf.rewind()
        val px = IntArray(INPUT * INPUT)
        bmp.getPixels(px, 0, INPUT, 0, 0, INPUT, INPUT)
        for (p in px) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            inBuf.putFloat((r - 0.485f) / 0.229f)
            inBuf.putFloat((g - 0.456f) / 0.224f)
            inBuf.putFloat((b - 0.406f) / 0.225f)
        }
        inBuf.rewind()
    }

    /** labels 资产每行 "学名\t中文名"，顺序即类别索引。 */
    private fun loadLabels(context: Context, path: String): List<Pair<String, String>> =
        context.assets.open(path).bufferedReader().readLines()
            .filter { it.isNotBlank() }
            .map { line ->
                val parts = line.split('\t')
                (parts.getOrNull(0)?.trim() ?: "?") to (parts.getOrNull(1)?.trim() ?: "")
            }

    private companion object {
        const val INPUT = 224
        const val TOP_K = 5
    }
}

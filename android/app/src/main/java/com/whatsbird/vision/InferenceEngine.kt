package com.whatsbird.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 推理引擎抽象（PLAN §7 可插拔决策）：换模型只改 assets + ModelRegistry，
 * 不动调用方。图像模型实现走 LiteRT(TFLite)；geomodel 重排将来单独走 ONNX Runtime。
 */

/** 单个检测框：坐标为相对整帧的归一化 [0,1]，x/y 顺序。 */
data class Detection(
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val cocoClass: Int = -1,
) {
    /** 目标框外扩（每侧加 宽×ratio/2；ratio=4.0 的依据见 ModelRegistry 注释与
     *  docs/handoff/root/CROP-REGIME.md）。 */
    fun expand(ratio: Float): Detection {
        val dx = (right - left) * ratio / 2f
        val dy = (bottom - top) * ratio / 2f
        return Detection(
            score,
            (left - dx).coerceIn(0f, 1f),
            (top - dy).coerceIn(0f, 1f),
            (right + dx).coerceIn(0f, 1f),
            (bottom + dy).coerceIn(0f, 1f),
        )
    }

    fun toPixelRect(frameW: Int, frameH: Int) = RectF(
        left * frameW, top * frameH, right * frameW, bottom * frameH,
    )
}

/** 单个分类得分；prob 是前 [ModelRegistry.NUM_SPECIES] 类 softmax 后的概率。 */
data class ClassScore(
    val index: Int,
    val sci: String,
    val zh: String,
    val prob: Float,
)

/** 一帧的完整推理结果。 */
data class FrameResult(
    val frameW: Int,
    val frameH: Int,
    val detections: List<Detection>,
    val top5: List<ClassScore>,
    val inferMs: Float = 0f,
    val crop: Bitmap? = null, // 最高分框的外扩裁剪图（记录附件用）
) {
    companion object {
        val EMPTY = FrameResult(0, 0, emptyList(), emptyList())
    }
}

interface BirdDetector {
    fun detect(src: Bitmap): List<Detection>

    /** 当前档位偏好：[ModelRegistry.TIER_AUTO] / [ModelRegistry.TIER_FLAG] / [ModelRegistry.TIER_MID]。 */
    fun tier(): String

    /** 实际生效的资产名（auto 时由 RAM 判定；未载入时也返回将要载入的那个）。 */
    fun activeAsset(): String

    /**
     * 切档。返回 true = 档位变了（已卸载旧模型，下一帧按新档载入）；
     * false = 本来就是这一档，什么也没做。
     * **一次只驻留一档**，不预载、不并存（两个 tflite 各 8.9MB）。
     */
    fun switchTier(tier: String): Boolean

    /** 设备总内存（ActivityManager.totalMem，**不是物理内存**，见 ModelRegistry 注释）。 */
    fun totalMemBytes(): Long
}

interface BirdClassifier {
    fun classify(src: Bitmap): List<ClassScore>
}

fun loadModel(context: Context, assetPath: String): MappedByteBuffer =
    context.assets.openFd(assetPath).use { fd ->
        FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength,
        )
    }

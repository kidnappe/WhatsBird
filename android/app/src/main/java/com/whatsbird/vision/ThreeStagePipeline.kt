package com.whatsbird.vision

import android.graphics.Bitmap
import javax.inject.Inject
import javax.inject.Singleton

/**
 * 三段式推理（PLAN §7 已锁决策）：
 * 检测 → 框外扩 10-20% → crop → 分类 → Top-5。
 * geomodel 重排是第三层，将来在 classify 结果上叠加（ONNX Runtime，独立实现）。
 */
/** 重检/扫描结果：映射回全帧的框 + 当时的放大裁剪图（分类用，像素质量优于全帧小框）。 */
data class Rescan(val frame: FrameResult, val crop: Bitmap)

@Singleton
class ThreeStagePipeline @Inject constructor(
    private val detector: BirdDetector,
    private val classifier: BirdClassifier,
) {
    /**
     * skipClassify=true 时只跑检测（锁定态省电、防候选闪烁），top5 返回空，
     * 由调用方复用此前的稳定候选。
     * preferBox（多鸟）：主目标轨迹的框——分类/记录裁剪目标跟随它而非最高分检测；
     * 为空时维持原行为（最高分检测）。仅当本帧有检出时生效。
     */
    fun process(src: Bitmap, skipClassify: Boolean = false, preferBox: Detection? = null): FrameResult {
        val t0 = android.os.SystemClock.elapsedRealtime()

        val detections = detector.detect(src)
        val bestDet = detections.maxByOrNull { it.score }
            ?: return FrameResult(src.width, src.height, detections, emptyList(), msSince(t0))
        val best = preferBox ?: bestDet

        // 框选裁剪图始终生成（记录附件用；锁定态跳分类时也保留最新画面）
        val crop = cropRect(src, best.expand(ModelRegistry.BOX_EXPAND_RATIO))

        if (skipClassify) {
            return FrameResult(
                src.width, src.height, detections, emptyList(), msSince(t0), crop,
            )
        }

        val top5 = classifier.classify(crop)
        // 软门槛：分类器 Top-1 置信度过低 → 视为不像鸟，不给出候选
        val confident = (top5.firstOrNull()?.prob ?: 0f) >= ModelRegistry.MIN_CLASS_PROB
        return FrameResult(
            src.width, src.height, detections,
            if (confident) top5 else emptyList(), msSince(t0), crop,
        )
    }

    /**
     * 局部重检测：全帧检不到小目标时，在记忆框外扩 [expand] 倍的区域内
     * 重新跑检测（等效数字变焦），坐标映射回全帧。
     * 返回 null 表示区域内也没检出。
     */
    fun redetectInRegion(
        src: Bitmap,
        mem: Detection,
        expand: Float = 2.5f,
    ): Rescan? {
        val fw = src.width
        val fh = src.height
        // ⚠️ 2026-10-10：原式 coerceIn(0f, fw - 16f) 在 fw<16 时下界大于上界会抛
        // IllegalArgumentException（极小图/异常分辨率）。这里直接放弃重扫。
        if (fw < MIN_ROI || fh < MIN_ROI) return null
        val cx = (mem.left + mem.right) / 2f * fw
        val cy = (mem.top + mem.bottom) / 2f * fh
        val w = (mem.right - mem.left) * fw * expand
        val h = (mem.bottom - mem.top) * fh * expand
        // ⚠️ 2026-10-10 修：三个边界必须彼此自洽，否则 x+width > bitmap.width()。
        //   原式：l 钳到 fw-16、roiW 下界 16 → l+roiW 最大恰好 = fw，**勉强**合法。
        //   本轮一度把 roiW 下界提到 MIN_ROI(32) 却没同步抬 l 的上界、也没给 roiW 上界，
        //   于是 l+roiW 最大到 fw+16 → 真机压测直接炸（21:35 日志 20+ 次
        //   IllegalArgumentException: x + width must be <= bitmap.width()）。
        //   现在：l 上界 = fw - MIN_ROI，roiW 上下界都钳 → l + roiW <= fw 恒成立。
        val l = (cx - w / 2).coerceIn(0f, (fw - MIN_ROI).toFloat())
        val t = (cy - h / 2).coerceIn(0f, (fh - MIN_ROI).toFloat())
        val roiW = (minOf(cx + w / 2, fw.toFloat()) - l).toInt().coerceIn(MIN_ROI, fw - l.toInt())
        val roiH = (minOf(cy + h / 2, fh.toFloat()) - t).toInt().coerceIn(MIN_ROI, fh - t.toInt())
        val t0 = android.os.SystemClock.elapsedRealtime()

        val crop = Bitmap.createBitmap(src, l.toInt(), t.toInt(), roiW, roiH)
        // ⚠️ 2026-10-10：crop 在下面的早退分支原先从不回收，每帧白扔一张 ROI 位图。
        // 只有最终返回的那张交给 Rescan 由调用方负责，其余一律就地回收。
        val local = detector.detect(crop).maxByOrNull { it.score }
        if (local == null) {
            crop.recycle()
            return null
        }

        // 局部归一化坐标 -> 全帧归一化坐标
        val mapped = Detection(
            score = local.score,
            left = (l + local.left * roiW) / fw,
            top = (t + local.top * roiH) / fh,
            right = (l + local.right * roiW) / fw,
            bottom = (t + local.bottom * roiH) / fh,
            cocoClass = local.cocoClass,
        )

        // 一致性关卡（防追到非鸟物体）：
        // ① 局部模式分数门槛更高（真鸟在放大裁剪里分数应显著高）
        // ② 面积比约束（0.25~5 倍）：追到别的物体通常尺寸差异大
        // ③ 中心距离约束：映射后中心不能漂出记忆框对角线的 0.9 倍
        val memArea = (mem.right - mem.left) * (mem.bottom - mem.top)
        val area = (mapped.right - mapped.left) * (mapped.bottom - mapped.top)
        val ratio = area / (memArea + 1e-6f)
        val memCx = (mem.left + mem.right) / 2f
        val memCy = (mem.top + mem.bottom) / 2f
        val mCx = (mapped.left + mapped.right) / 2f
        val mCy = (mapped.top + mapped.bottom) / 2f
        val diag = kotlin.math.hypot((mem.right - mem.left), (mem.bottom - mem.top))
        val dist = kotlin.math.hypot(mCx - memCx, mCy - memCy)
        if (local.score < 0.60f || ratio < 0.25f || ratio > 5f || dist > diag * 0.9f) {
            crop.recycle() // 2026-10-10：早退也要回收
            return null
        }
        return Rescan(FrameResult(fw, fh, listOf(mapped), emptyList(), msSince(t0)), crop)
    }

    /**
     * 中心数字变焦扫描：全帧检不到远景小目标时，对画面中心 [region] 比例的区域
     * 重新检测（等效 region 倍的放大，region=0.5 ≈ 2x 变焦），坐标映射回全帧。
     * 用于 1x 广角下的初始获取——鸟通常在画面中部。返回 null 表示中心区也没检出。
     */
    fun centerRescan(src: Bitmap, region: Float = 0.5f): Rescan? =
        regionRescan(src, (1f - region) / 2f, (1f - region) / 2f, region)

    /**
     * 区域数字变焦扫描（渐进扫描 B §5 P1 的宫格相位）：对 [leftFrac, topFrac] 起、
     * 边长 [regionFrac] 的子区域重新检测，坐标映射回全帧。regionRescan(0.5,0.5,0.5)
     * 即中心扫描；2×2 宫格 = 四个角各扫一次（0/0、0.5/0、0/0.5、0.5/0.5）。
     */
    fun regionRescan(
        src: Bitmap,
        leftFrac: Float,
        topFrac: Float,
        regionFrac: Float,
    ): Rescan? {
        val fw = src.width
        val fh = src.height
        // ⚠️ 2026-10-10 修：同 regionRescan 的边界自洽问题。
        // l 钳到 fw-16 时，coerceIn(MIN_ROI=32, fw-l) 的下界可能大于上界（fw-l ∈ [16,31]）
        // → coerceIn 自己抛 IllegalArgumentException。现在 l 上界 = fw - MIN_ROI，
        // 于是 fw-l >= MIN_ROI，coerceIn 区间恒非空，且 l+roiW <= fw。
        if (fw < MIN_ROI || fh < MIN_ROI) return null
        val l = (fw * leftFrac).toInt().coerceIn(0, fw - MIN_ROI)
        val t = (fh * topFrac).toInt().coerceIn(0, fh - MIN_ROI)
        val roiW = (fw * regionFrac).toInt().coerceIn(MIN_ROI, fw - l)
        val roiH = (fh * regionFrac).toInt().coerceIn(MIN_ROI, fh - t)
        val t0 = android.os.SystemClock.elapsedRealtime()

        val crop = Bitmap.createBitmap(src, l, t, roiW, roiH)
        val local = detector.detect(crop).maxByOrNull { it.score }
        if (local == null) {
            crop.recycle()
            return null
        }

        val mapped = Detection(
            score = local.score,
            left = (l + local.left * roiW) / fw,
            top = (t + local.top * roiH) / fh,
            right = (l + local.right * roiW) / fw,
            bottom = (t + local.bottom * roiH) / fh,
            cocoClass = local.cocoClass,
        )
        return Rescan(FrameResult(fw, fh, listOf(mapped), emptyList(), msSince(t0)), crop)
    }

    /** 直接对给定裁剪图分类（重检/扫描的放大裁剪，1x 场景等效提升分类像素质量）。 */
    fun classifyCrop(crop: Bitmap): List<ClassScore> = classifier.classify(crop)

    /** 按检测框外扩 [ModelRegistry.BOX_EXPAND_RATIO] 裁出分类用图（多鸟：非主目标首次分类用）。 */
    fun cropFor(src: Bitmap, d: Detection): Bitmap =
        cropRect(src, d.expand(ModelRegistry.BOX_EXPAND_RATIO))

    private fun cropRect(src: Bitmap, d: Detection): Bitmap {
        val r = d.toPixelRect(src.width, src.height)
        // left/top 先钳进画面（preferBox 是轨迹预测框，可能整体漂出画面边缘退化成 0 宽，
        // coerceIn(1, 0) 会抛 IllegalArgumentException → 整帧推理失败，2026-10-07 装机实爆）
        val left = r.left.coerceIn(0f, src.width - 1f).toInt()
        val top = r.top.coerceIn(0f, src.height - 1f).toInt()
        val width = (r.right.toInt() - left).coerceIn(1, src.width - left)
        val height = (r.bottom.toInt() - top).coerceIn(1, src.height - top)
        return Bitmap.createBitmap(src, left, top, width, height)
    }

    private fun msSince(t0: Long) =
        (android.os.SystemClock.elapsedRealtime() - t0).toFloat()

    private companion object {
        /** 裁剪 ROI 的最小边长；小于 MIN_ROI 的图直接放弃重扫。
         *  ⚠️ 2026-10-10：不要再引入"半边长"之类的派生常量——2026-10-10 那次真机压测就是
         *  roiW 下界用 MIN_ROI(32)、而 l 上界用 MIN_ROI/2(16)，两者不自洽导致
         *  x+width 越界抛异常。**l 上界与 roiW 上下界必须用同一个 MIN_ROI。** */
        const val MIN_ROI = 32
    }
}

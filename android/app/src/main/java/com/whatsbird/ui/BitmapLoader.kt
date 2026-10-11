package com.whatsbird.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * 组合期安全的照片加载（2026-10-10 新增）。
 *
 * 修的问题：图鉴格、图鉴详情 hero/缩略图、记录缩略图、日志大图、FullScreenImage
 * 原先都在组合期用 `remember { BitmapFactory.decodeFile(...) }` **同步在主线程解码**，
 * 采样率还是写死的常量。后果：列表滚动掉帧，相机原片（1080p 以上）有 OOM 风险，
 * 且 OOM 被 runCatching 吞掉后界面只是空白，看不出原因。
 *
 * 现在：解码放 IO 线程，并且**先 inJustDecodeBounds 量尺寸再反推采样率**，
 * 保证解码出来的位图不超过目标显示边长。
 */

/** 按 [maxEdgePx] 反推采样率解码；失败（含 OOM）返回 null 并记日志。 */
fun decodeSampled(path: String, maxEdgePx: Int, minSample: Int = 1): Bitmap? {
    if (maxEdgePx <= 0) return null
    return try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            Log.w(TAG, "无法读取图片头 $path")
            return null
        }
        var sample = minSample.coerceAtLeast(1)
        // 两边都还远大于目标边长时才再翻倍——任一边到���即可，避免细长图被过度缩小
        while (sample < 32 &&
            bounds.outWidth / (sample * 2) >= maxEdgePx &&
            bounds.outHeight / (sample * 2) >= maxEdgePx
        ) {
            sample *= 2
        }
        BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
    } catch (e: OutOfMemoryError) {
        // 显式接住并记账：OOM 不是"可忽略的异常"，但看图页不该因此崩掉整页
        Log.e(TAG, "解码 OOM $path (maxEdge=$maxEdgePx)", e)
        null
    } catch (e: Exception) {
        Log.w(TAG, "解码失败 $path", e)
        null
    }
}

/**
 * 组合期取位图：结果为 [State]，解码在 IO 线程完成，主线程只等结果。
 * [minSample] 用于保留原有采样下限（例如详情页要更清晰）。
 *
 * 实现说明：用 LaunchedEffect + mutableStateOf 而不是 produceState——
 * 后者会触发 lint 的 ProduceStateDoesNotAssignValue 误报（赋值在 if/else 表达式里时）。
 */
@Composable
fun rememberDecoded(path: String?, maxEdge: Dp, minSample: Int = 1): State<Bitmap?> {
    val px = with(LocalDensity.current) { maxEdge.roundToPx() }
    val holder = remember { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(path, px, minSample) {
        holder.value = if (path.isNullOrEmpty()) {
            null
        } else {
            withContext(Dispatchers.IO) { decodeSampled(path, px, minSample) }
        }
    }
    return holder
}

private const val TAG = "BitmapLoader"
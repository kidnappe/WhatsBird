/*
 * Adapted from google/jetpack-camera-app `ZoomStateManager` (Apache License 2.0)
 * Copyright (C) 2025 The Android Open Source Project
 * https://github.com/google/jetpack-camera-app
 *
 * Modifications: single-lens use case; no JCA model/controller dependencies —
 * zoom is reported back through a plain callback instead of ZoomController.
 */
package com.whatsbird.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.AnimationSpec
import androidx.compose.animation.core.tween
import androidx.compose.foundation.MutatorMutex

/**
 * 相机变焦状态管理：
 * - MutatorMutex 保证捏合 / 挡位点击 / 自动放大三类来源互斥，后到取消先到
 * - 挡位切换走 Animatable 平滑过渡（每帧回调当前倍率），不再硬跳
 */
class ZoomAnimator(
    initialZoom: Float,
    private var zoomRange: ClosedFloatingPointRange<Float>,
    private val onZoomRatio: (Float) -> Unit,
) {
    private var functionalZoom = initialZoom.coerceIn(zoomRange)
    private val mutatorMutex = MutatorMutex()

    /** 相机变焦范围就绪后（异步）更新上下限。 */
    fun updateRange(range: ClosedFloatingPointRange<Float>) {
        zoomRange = range
        functionalZoom = functionalZoom.coerceIn(range)
    }

    suspend fun absoluteZoom(targetZoomLevel: Float) {
        mutatorMutex.mutate {
            functionalZoom = targetZoomLevel.coerceIn(zoomRange)
            onZoomRatio(functionalZoom)
        }
    }

    /** 捏合手势：在当前倍率上乘系数。 */
    suspend fun scaleZoom(scalingFactor: Float) = absoluteZoom(scalingFactor * functionalZoom)

    /** 挡位点击：平滑动画到目标倍率。 */
    suspend fun animatedZoom(
        targetZoomLevel: Float,
        animationSpec: AnimationSpec<Float> = tween(durationMillis = 400),
    ) {
        mutatorMutex.mutate {
            Animatable(functionalZoom).animateTo(
                targetValue = targetZoomLevel.coerceIn(zoomRange),
                animationSpec = animationSpec,
            ) {
                functionalZoom = value
                onZoomRatio(value)
            }
        }
    }
}

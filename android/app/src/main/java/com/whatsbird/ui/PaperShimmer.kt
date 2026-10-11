package com.whatsbird.ui

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.unit.dp
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.graphicsLayer

/**
 * ⑤ 骨架 / 微光（v6 P1-3，抄 SwiftUI_Demos/Skeleton）
 *
 * **关键不是微光本身，是"真组件喂假数据"这个套路**：一个 `SpeciesCard(species: UiState)`，
 * 加载中喂 `Species.sample`，数据到了淡入替换。零重复布局、零"骨架与真卡不一致"的风险。
 *
 * 参数抄 `SkeletonModifier.swift`：`band = w/2`、旋转 5°、`easeInOut 1500ms` 循环、
 * `BlendMode.SoftLight`。
 *
 * ⚠️ **必须偏离源码**：他们那个 `blurRadius = max(w/4, 30)` 是真模糊，
 * Android 上 30px 级别的 RenderEffect 太贵。这里改成一条**预烘焙的 5 段横向渐变**
 * 靠 `translationX` 推过去，视觉等价，GPU 成本差一个数量级。
 *
 * 仅在 `loading == true` 时画；否则整个 Modifier 不产生任何绘制开销。
 */
fun Modifier.paperShimmer(loading: Boolean, bandColor: Color): Modifier {
    if (!loading) return this
    // rememberInfiniteTransition 是 @Composable，必须包在 composed 里
    return this.composed {
        val t = rememberInfiniteTransition(label = "shimmer")
        val x by t.animateFloat(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(
                    durationMillis = 1500,
                    easing = androidx.compose.animation.core.FastOutSlowInEasing,
                ),
                repeatMode = RepeatMode.Restart,
            ),
            label = "shimmerX",
        )
        graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }
            .drawBehind {
                val w = size.width
                val h = size.height
                val bandW = w * 0.6f          // 覆盖 -w/2 .. 1.6w，保证两端不留缝
                val start = -w * 0.5f + x * (w + bandW)
                // 预烘焙的 5 段渐变（透明 → 亮 → 更亮 → 亮 → 透明）
                val stops = listOf(
                    0f to Color.Transparent,
                    0.35f to bandColor.copy(alpha = 0.35f),
                    0.5f to bandColor.copy(alpha = 0.75f),
                    0.65f to bandColor.copy(alpha = 0.35f),
                    1f to Color.Transparent,
                )
                drawRect(
                    brush = Brush.linearGradient(
                        colorStops = stops.toTypedArray(),
                        start = Offset(start, 0f),
                        end = Offset(start + bandW, h),
                    ),
                    size = androidx.compose.ui.geometry.Size(w, h),
                )
            }
    }
}

/**
 * ⑤ 骨架占位块：微光底 + 圆角，用于"图鉴墙首屏 384 条尚未落库"的空态。
 */
@Composable
fun PaperSkeletonBox(
    modifier: Modifier = Modifier,
    radius: Int = 16,
) {
    val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t
    val reduce = com.whatsbird.ui.theme.rememberReduceMotion()
    Box(
        modifier
            .background(t.silA, androidx.compose.foundation.shape.RoundedCornerShape(radius.dp))
            .paperShimmer(loading = !reduce, bandColor = t.card)
    )
}
package com.whatsbird.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Stable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import kotlin.math.hypot

/**
 * ① 换色圆形擦除（docs/design/DESIGN.md §5.2 P0-1）
 *
 * 抄 SwiftUI_Demos/DarkModeAnimation 的"圆形遮罩揭开"：原版拍旧屏/新屏两张快照，
 * 再用从按钮位置扩张的圆把新快照揭出来。
 *
 * 本实现**不做快照**，改用等价但便宜得多的做法：叠一层"新纸底色"的圆，
 * 从用户点的那枚圆盘位置扩张并同时淡出；底下 tokens 本来就在
 * `PaperTokens.animateForTheme()` 里做 250ms 逐色交叉渐变，两者叠起来
 * 读感就是"新颜色从那枚圆盘里洇开"。
 *
 * ⚠️ 时长 450ms（不是原版的 900ms）——0.9s 对"标本柜的安静"太长。
 */
@Stable
class PaletteWipeState {
    /** 圆心（root 坐标系） */
    var origin by mutableStateOf(Offset.Unspecified)
    /** 洇开的颜色 = 新色板的 paper */
    var color by mutableStateOf(Color.Transparent)
    /** 自增令牌：每次 +1 触发一次新动画（不能用 Boolean 反复触发） */
    var token by mutableIntStateOf(0)

    fun fire(origin: Offset, color: Color) {
        this.origin = origin
        this.color = color
        this.token++
    }
}

val LocalPaletteWipe = compositionLocalOf { PaletteWipeState() }

/**
 * 覆盖在整页最上层的洇开层（挂载见 MainActivity 的 root Box）。
 * 无正在进行的动画时什么都不画，零开销。
 */
@Composable
fun PaletteWipeOverlay(state: PaletteWipeState) {
    val token = state.token
    val origin = state.origin
    val color = state.color
    val reduce = com.whatsbird.ui.theme.rememberReduceMotion()
    val density = LocalDensity.current

    if (token == 0 || origin == Offset.Unspecified) return
    val p = remember(token) { Animatable(0f) }
    LaunchedEffect(token) {
        if (reduce) return@LaunchedEffect          // 减弱动态效果：不做洇开，交给 tokens 的瞬变
        p.snapTo(0f)
        p.animateTo(1f, tween(450, easing = FastOutSlowInEasing))
    }
    val v = p.value
    if (reduce || v <= 0f) return

    Box(
        Modifier
            .fillMaxSize()
            .drawBehind {
                // 半径取到画面对角，任意位置点都能铺满
                val maxR = with(density) { hypot(size.width.toFloat(), size.height.toFloat()) }
                drawCircle(
                    color = color,
                    radius = v * maxR,
                    center = origin,
                    // 边扩散边消隐：(1-v)² 让中段最实、首尾都淡
                    alpha = ((1f - v) * (1f - v)).coerceIn(0f, 1f) * 0.85f,
                )
            }
    )
}
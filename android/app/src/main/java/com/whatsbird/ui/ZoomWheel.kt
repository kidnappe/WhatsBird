/*
 * 滚轮式变焦（MIUI/OneUI 相机的弧形刻度盘形态：向上的半圆拱顶）。
 *
 * 无开源对应组件可抄（GitHub/Gitee/CSDN/掘金/开源相机均无，厂商滚轮为闭源
 * 定制 UI），状态层复用 JCA 改编的 ZoomAnimator（互斥 + 动画），本文件只做
 * 视觉与手势层。关键手感：刻度带在固定指针下随手势滑动、跨刻度 CLOCK_TICK
 * 振动、按弧长映射的灵敏度、拖拽时药丸淡出/滚轮滑入（手势不中断）。
 */
package com.whatsbird.ui

import android.view.HapticFeedbackConstants
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableFloatStateOf
import kotlin.math.abs
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.layout
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val DIAL_MIN = 0.6f
private const val DIAL_MAX = 30f
private const val SWEEP_ANGLE = 120f // 拱顶两侧各 60° 的可视跨度
private const val POINTER_ANGLE = 270f // 指针固定在拱顶（圆心在屏幕下方）
private const val RADIUS_X_WIDTH = 1.35f // 盘半径 = 1.35 × 画布宽（弧线平缓）
private val LN_SPAN = ln(DIAL_MAX / DIAL_MIN)

/** 大刻度（带数字）+ 大刻度之间 3 条小刻度，构成原生相机的密集刻度扇面 */
private val MAJORS = listOf(0.6f, 1f, 2f, 2.6f, 5f, 10f, 15f, 20f, 30f)
private val TICKS: List<Pair<Float, Boolean>> = run {
    val out = mutableListOf<Pair<Float, Boolean>>()
    for (i in MAJORS.indices) {
        out.add(MAJORS[i] to true)
        if (i < MAJORS.size - 1) {
            val a = ln(MAJORS[i] / DIAL_MIN)
            val b = ln(MAJORS[i + 1] / DIAL_MIN)
            for (j in 1..3) {
                out.add((DIAL_MIN * exp(a + (b - a) * j / 4f)) to false)
            }
        }
    }
    out
}

/** 变焦挡位（对齐小米 15 镜头组：0.6 超广 / 1x / 2x / 2.6x 长焦） */
private val ZOOM_LEVELS = listOf(0.6f, 1f, 2f, 2.6f)

private fun frac(z: Float) = (ln(z / DIAL_MIN) / LN_SPAN).coerceIn(0f, 1f)

private fun tickLabel(t: Float) =
    if (t == t.toInt().toFloat()) t.toInt().toString() else "%.1f".format(t)

/**
 * 滚轮变焦：挡位药丸居中，按住横向拖动 → 药丸淡出、向上拱起的弧形刻度盘滑入
 * 并随手势滚动，松手 1.5s 后反向收回。onDragZoom 收系数，onZoomTo 收目标倍率。
 */
@Composable
fun ZoomWheel(
    zoomRatio: Float,
    expanded: Boolean,
    onExpandChange: (Boolean) -> Unit,
    onDragZoom: (Float) -> Unit,
    onZoomTo: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val currentZoom by rememberUpdatedState(zoomRatio)
    val expandedState by rememberUpdatedState(expanded)
    val view = LocalView.current
    val scope = rememberCoroutineScope()
    var collapseJob by remember { mutableStateOf<Job?>(null) }
    var lastTick by remember { mutableIntStateOf(-1) }

    /**
     * ⑦ 点选挡位时把高亮**钉死**在目标挡（NaN = 未钉住）。
     *
     * 放在这里而不是 ZoomButtonRow 里，是因为只有本函数知道每一帧的倍率变化
     * **来自哪里**：拖拽/捏合会先清掉钉住，点选则建立钉住。这样不存在
     * "钉住值过期没人清"的问题。
     */
    var pinnedLevel by remember { mutableFloatStateOf(Float.NaN) }
    // ⚠️ 2026-10-10 二次修正：原先是「倍率接近目标 <0.05 就释放」。
    // 跨倍数切换（2.6 → 0.6）时倍率单调经过，接近度判据可能在途中先被满足，
    // 钉子一放药丸就闪回实时值。现改为**按补间时长确定释放**（见 onChangeZoom 的 delay(450)）。
    // 2026-10-10：定位此问题时加过逐帧 Log.d，已确认根因后删除。
    // 2026-10-09：盘面配色已固定（白刻度 / 红选定标），不再读取主题色。
    // 主题色仍由药丸（ZoomButtonRow）自己使用，与本控件无关。

    fun cancelCollapse() {
        collapseJob?.cancel()
    }

    fun armCollapse() {
        collapseJob?.cancel()
        collapseJob = scope.launch {
            delay(1500)
            onExpandChange(false)
        }
    }

    fun tickHaptic(predictedZoom: Float) {
        val idx = (frac(predictedZoom) * (TICKS.size - 1)).roundToInt()
        if (idx != lastTick) {
            lastTick = idx
            view.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
        }
    }

    // 拖动 → 缩放系数：按弧长换算（dx ÷ 半径 = 弧度，按对数轴比例）
    fun dragFactor(dxPx: Float, canvasWidthPx: Float): Float {
        val sweepRad = Math.toRadians(SWEEP_ANGLE.toDouble()).toFloat()
        return exp(-dxPx * LN_SPAN / (RADIUS_X_WIDTH * canvasWidthPx * sweepRad))
    }

    fun handleDrag(dx: Float, widthPx: Float) {
        val factor = dragFactor(dx, widthPx)
        // 任何拖拽都撤销「钉住」——拖拽才是当前倍率的主导来源
        pinnedLevel = Float.NaN
        onDragZoom(factor)
        tickHaptic(currentZoom * factor)
    }

    // 拖拽手势挂在两个常驻容器上（药丸区 + 刻度盘区）：
    // 药丸淡出也不中断进行中的手势，滚轮出现后原地继续拖也有效
    val dragModifier = Modifier.pointerInput(Unit) {
        detectHorizontalDragGestures(
            onDragStart = {
                cancelCollapse()
                view.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                if (!expandedState) onExpandChange(true)
            },
            onDragEnd = { armCollapse() },
            onDragCancel = { armCollapse() },
        ) { change, dx ->
            change.consume()
            handleDrag(dx, size.width.toFloat())
        }
    }

    val pillAlpha by animateFloatAsState(
        targetValue = if (expanded) 0f else 1f,
        animationSpec = tween(durationMillis = 150),
        label = "pillAlpha",
    )

    // 药丸占据的高度（ZoomButtonRow: buttonSize 36 + 上下各 6 = 48dp）。
    // 轮盘拉起时这块**完全收起**（高度归零），松手收起轮盘后再长回来成为药丸背景。
    // 取景画面因此向下多出 48dp，而不是被向上顶。
    val pillSpace by animateDpAsState(
        targetValue = if (expanded) 0.dp else PILL_H,
        animationSpec = tween(durationMillis = 200),
        label = "pillSpace",
    )

    // 高度约为屏幕的 1/5（对照小米截图的盘面占比）
    Box(
        modifier = modifier
            .fillMaxWidth()
            // 滚轮展开时：点滚轮区域自身不穿透（空白处的点击由预览层处理为收起滚轮）
            .pointerInput(expanded) {
                if (expanded) detectTapGestures(onTap = { /* 吃掉，防误触对焦 */ })
            },
    ) {
        // ── 挡位药丸：轮盘拉起时**高度一并收起**（下方注释）──
        Box(
            Modifier
                .fillMaxWidth()
                .height(pillSpace)
                .then(dragModifier),
        ) {
            // 挡位药丸：居中。拖拽时淡出但保留在组合里（手势链不断）
            Box(
                Modifier
                    .align(Alignment.Center)
                    .graphicsLayer { alpha = pillAlpha }
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null,
                        enabled = expanded, // 滚轮展开期间吃掉药丸区域的点击防误触
                    ) {},
            ) {
                ZoomButtonRow(
                    currentZoom = currentZoom,
                    zoomLevels = ZOOM_LEVELS,
                    // 钉住期间高亮不参与区间推导，补间途中不会被中间挡位抢走
                    pinnedLevel = if (pinnedLevel.isNaN()) null else pinnedLevel,
                    onChangeZoom = { target ->
                        pinnedLevel = target
                        onZoomTo(target)
                        // 按补间时长确定释放：animatedZoom 是 tween(400)，
                        // 这里等 450ms 再放。不依赖「接近度」，避免中途闪回。
                        scope.launch {
                            delay(450)
                            if (!pinnedLevel.isNaN() && abs(pinnedLevel - target) < 0.01f) {
                                pinnedLevel = Float.NaN
                            }
                        }
                    },
                )
            }
        }

        // ── 弧形刻度盘：**零高度浮层**（2026-10-09）──
        // 改造前它是 Column 的第一个子项，展开时向布局插入 170dp，把「底部控制带」
        // 顶高 → 取景画面被压缩、**底边被抬起来**。
        // 现在用 Modifier.layout 向父布局报告 0 高度、内容向上绘制：
        //   · 底部带高度恒定（只由药丸决定）→ 取景画面尺寸永远不变
        //   · 盘面浮在取景画面之上，背后保持通透
        AnimatedVisibility(
            visible = expanded,
            enter = slideInVertically { it / 2 } + fadeIn(),
            exit = slideOutVertically { it / 2 } + fadeOut(),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .layout { measurable, constraints ->
                    val place = measurable.measure(constraints)
                    val up = DIAL_H.toPx().toInt()
                    layout(place.width, 0) { place.place(0, -up) }
                },
        ) {
            Canvas(
                Modifier
                    .fillMaxWidth()
                    .height(DIAL_H)
                    .then(dragModifier),
            ) {
                // 盘面配色**固定，不再跟随主题色**（2026-10-09 定稿）：
                //   刻度线与数字 = 白；选定标（指向当前倍率的那根竖线）= 红。
                // 底色透明，背���取景画面直接透过来。
                drawZoomDial(currentZoom, DIAL_ACCENT, Color.Transparent, DIAL_INK, DIAL_TICK)
            }
        }
    }
}

/** 刻度盘画布高度；同时作为「向上浮出」的位移量。 */
private val DIAL_H = 170.dp

/** 药丸占用高度（ZoomButtonRow 36dp 按钮 + 上下各 6dp 内边距）。 */
private val PILL_H = 48.dp

// ── 盘面固定配色（2026-10-09 定稿，不再跟随主题色）──
// 刻度线/数字统一白色；选定标（指当前倍率的那根竖线）用红色。
private val DIAL_INK = Color.White
private val DIAL_TICK = Color.White
private val DIAL_ACCENT = Color(0xFFE23B3B)

/** 向上拱起的弧形刻度盘：指针固定在拱顶，整条刻度带随当前倍率滑动。 */
private fun DrawScope.drawZoomDial(
    zoom: Float,
    accent: Color,
    track: Color,
    ink: Color,
    tickMinor: Color,
) {
    val w = size.width
    val h = size.height
    val r = RADIUS_X_WIDTH * w
    val cx = w / 2f
    val apexY = 0.40f * h // 拱顶（指针位置）在画布上部
    val cy = apexY + r // 圆心在拱顶正下方 r 处（屏幕外）

    val paint = android.graphics.Paint().apply {
        color = ink.toArgb()
        textAlign = android.graphics.Paint.Align.CENTER
        isAntiAlias = true
    }

    /** 透明盘面上的白色文字：先描一圈暗边再填，保证亮/暗背景下都读得清。 */
    fun drawLabel(text: String, x: Float, y: Float, sizePx: Float) {
        paint.textSize = sizePx
        paint.style = android.graphics.Paint.Style.STROKE
        paint.strokeWidth = (sizePx * 0.16f).coerceAtLeast(2f)
        paint.color = Color.Black.copy(alpha = 0.42f).toArgb()
        drawIntoCanvas { it.nativeCanvas.drawText(text, x, y, paint) }
        paint.style = android.graphics.Paint.Style.FILL
        paint.color = ink.toArgb()
        drawIntoCanvas { it.nativeCanvas.drawText(text, x, y, paint) }
    }

    // 盘面底色：2026-10-09 起传 Color.Transparent，**不画暗区**，
    // 只保留刻度线与数字，背后取景画面通透。数字改为「浅色描边 + 实心」，
    // 否则透明底上的墨绿字在暗背景上会糊掉。
    if (track.alpha > 0.01f) {
        val fill = Path().apply {
            arcTo(
                rect = Rect(cx - r, cy - r, cx + r, cy + r),
                startAngleDegrees = POINTER_ANGLE - SWEEP_ANGLE / 2f,
                sweepAngleDegrees = SWEEP_ANGLE,
                forceMoveTo = true,
            )
            lineTo(w + 50f, h + 50f)
            lineTo(-50f, h + 50f)
            close()
        }
        drawPath(fill, track)
    }

    // 刻度带：每格刻度的角度 = 指针角 + (刻度倍率 − 当前倍率) 的对数差 × 跨度
    // → 当前倍率永远对齐拱顶指针，整条带子随手势滚动（大倍率在右侧）
    val zoomFrac = frac(zoom)
    for ((t, major) in TICKS) {
        val angDeg = POINTER_ANGLE + (frac(t) - zoomFrac) * SWEEP_ANGLE
        val angRad = Math.toRadians(angDeg.toDouble())
        val cosA = cos(angRad).toFloat()
        val sinA = sin(angRad).toFloat()
        val rIn = r - 8.dp.toPx() // 弧线内侧（暗区）
        val rOut = r + 6.dp.toPx() // 弧线外侧（亮区，刻度戳出弧线的观感）
        // 白色刻度在亮背景下会消失 → 先画一条略粗的暗色衬底，再叠白色
        drawLine(
            Color.Black.copy(alpha = 0.38f),
            start = Offset(cx + rIn * cosA, cy + rIn * sinA),
            end = Offset(cx + rOut * cosA, cy + rOut * sinA),
            strokeWidth = (if (major) 3.5.dp else 2.dp).toPx(),
        )
        drawLine(
            if (major) ink else tickMinor.copy(alpha = 0.75f),
            start = Offset(cx + rIn * cosA, cy + rIn * sinA),
            end = Offset(cx + rOut * cosA, cy + rOut * sinA),
            strokeWidth = (if (major) 2.dp else 1.dp).toPx(),
        )
        if (major) {
            val lr = r - 18.dp.toPx() // 数字在弧线内侧（对照截图）
            val x = cx + lr * cosA
            val y = cy + lr * sinA + 13.dp.toPx() / 3f
            if (x > -30f && x < w + 30f && y > 0f && y < h + 30f) {
                drawLabel(tickLabel(t), x, y, 13.dp.toPx())
            }
        }
    }

    // 固定指针（选定标）：拱顶那根红色竖线 + 当前倍率大字
    drawLine(
        Color.Black.copy(alpha = 0.40f),
        start = Offset(cx, apexY - 17.dp.toPx()),
        end = Offset(cx, apexY - 2.dp.toPx()),
        strokeWidth = 5.dp.toPx(),
    )
    drawLine(
        accent,
        start = Offset(cx, apexY - 16.dp.toPx()),
        end = Offset(cx, apexY - 2.dp.toPx()),
        strokeWidth = 3.dp.toPx(),
    )
    drawLabel("%.1fx".format(zoom), cx, apexY - 26.dp.toPx(), 24.dp.toPx())
}

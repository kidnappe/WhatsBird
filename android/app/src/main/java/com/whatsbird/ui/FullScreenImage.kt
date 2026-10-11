package com.whatsbird.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.Saver
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.hypot

/** 一次全屏查看请求：该物种的照片序列（可左右滑）+ 起始页 + 题注。 */
data class ViewerReq(
    val paths: List<String>,
    val index: Int = 0,
    val zh: String,
    val sci: String,
)

/**
 * ViewerReq 的 rememberSaveable  Saver（2026-10-10）。
 *
 * 原来详情页/日志页用 `var viewReq by remember { ... }` 存大图请求，
 * 屏幕旋转或进程被回收后直接丢失——用户看到的是"照片突然没了"。
 * ViewerReq 不是 Parcelable，所以手工编码成 4 个 Bundle 安全的字段
 * （List<String> 存成 ArrayList<String>，其余是 Int/String）。
 */
val ViewerReqSaver: Saver<ViewerReq?, Any> = Saver(
    save = { r ->
        if (r == null) null
        else arrayListOf(ArrayList(r.paths), r.index, r.zh, r.sci)
    },
    restore = { a ->
        @Suppress("UNCHECKED_CAST")
        val parts = a as List<Any?>
        ViewerReq(
            paths = parts[0] as ArrayList<String>,
            index = parts[1] as Int,
            zh = parts[2] as String,
            sci = parts[3] as String,
        )
    },
)

/**
 * 全屏独占看图（docs/design/DESIGN.md §5.2）：
 * - 开场 scale 0.92→1 + 纱渐入
 *   （③ 共享元素已否决：L2 的 ModalBottomSheet 内部本身就是一个 Dialog，L3 再套 Dialog
 *    跨不过合成边界，**升级 Compose 版本也解决不了**，不是版本问题。降级式即终稿）
 * - 多图左右滑（照片墙/收集墙同源）+ 页码
 * - ⑥ 下滑关闭：跟手 + **速度投影**（PhotosApp）——200px = 满进度，松手按 dy + vy/5 判定；
 *   题注栏随进度反向平移，永远钉住
 * - ⑥ 双指捏合：**锚点落在手指位置**（InstagramPinchZoom）+ 硬钳位 ≥1（永不禁用放大后的位移）
 * - ⑦ 半边一 页码跟手：`currentPage` 是"离吸附位最近的那一页"的**台阶**值（PagerState.kt 原文），
 *   跨页过程不连续；改用 `currentPage + currentPageOffsetFraction` 连续驱动
 */
@Composable
fun FullScreenImageView(req: ViewerReq, onDismiss: () -> Unit) {
    val paths = req.paths.filter { it.isNotBlank() }
    val reduce = com.whatsbird.ui.theme.rememberReduceMotion()
    val scope = rememberCoroutineScope()

    // 下滑跟手位移（用 Animatable：拖动中 snapTo，松手可 animateTo）
    val dragY = remember { Animatable(0f) }
    // ⑥ 捏合：缩放 / 平移 / 锚点。用 MutableFloatState 存，因为要跟手实时写，
    // 且要在 pointerInput 回调里直接读（State 是稳定引用，不会捕获旧值）。
    val zoomV = remember { mutableFloatStateOf(1f) }
    val panX = remember { mutableFloatStateOf(0f) }
    val panY = remember { mutableFloatStateOf(0f) }
    val anchorX = remember { mutableFloatStateOf(0.5f) }
    val anchorY = remember { mutableFloatStateOf(0.5f) }

    // 开场：scale 0.92→1 + 纱渐入（⚠️ 此前 scale 算了但没挂到任何 modifier 上，是死代码）
    val openScale = remember { Animatable(if (reduce) 1f else 0.92f) }
    val alphaAnim = remember { Animatable(if (reduce) 1f else 0f) }
    LaunchedEffect(Unit) {
        if (!reduce) {
            alphaAnim.animateTo(1f, tween(180))
            openScale.animateTo(1f, tween(300, easing = androidx.compose.animation.core.FastOutSlowInEasing))
        } else {
            openScale.snapTo(1f); alphaAnim.snapTo(1f)
        }
    }
    val pager = rememberPagerState(
        initialPage = req.index.coerceIn(0, (paths.size - 1).coerceAtLeast(0)),
    ) { paths.size.coerceAtLeast(1) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false),
    ) {
        Box(
            Modifier
                .fillMaxSize()
                .graphicsLayer {
                    // 下滑时整层渐隐（v5 §3）
                    alpha = alphaAnim.value * (1f - abs(dragY.value) / 1200f).coerceIn(0f, 1f)
                }
                .background(Color(0xF50A0C08).copy(alpha = (0.96f * (1f - abs(dragY.value) / 1400f)).coerceIn(0.3f, 0.96f))),
            contentAlignment = Alignment.Center,
        ) {
            // 内容层：开场 scale + 下滑跟手位移
            Box(
                Modifier.fillMaxSize().graphicsLayer {
                    scaleX = openScale.value; scaleY = openScale.value
                    translationY = dragY.value
                }
            ) {
                HorizontalPager(
                    state = pager,
                    modifier = Modifier.fillMaxSize(),
                    userScrollEnabled = zoomV.value <= 1.001f,
                ) { page ->
                    // 2026-10-10：原先在组合期主线程同步 decodeFile 且写死 inSampleSize=2，
                    // 翻页时卡 UI、相机原片有 OOM 风险。改 IO 线程 + 按屏幕边长算采样率。
                    val bmp by rememberDecoded(paths.getOrNull(page), maxEdge = 2048.dp)
                    val img = bmp
                    Box(
                        Modifier
                            .fillMaxSize()
                            .graphicsLayer {
                                val s = zoomV.value
                                scaleX = s; scaleY = s
                                // 锚点补偿：缩放前把手指所在的归一化点按回原位
                                translationX = panX.value +
                                    (anchorX.value - 0.5f) * size.width * (1f - s)
                                translationY = panY.value +
                                    (anchorY.value - 0.5f) * size.height * (1f - s)
                            },
                        contentAlignment = Alignment.Center,
                    ) {
                        if (img != null) {
                            Image(
                                img.asImageBitmap(),
                                contentDescription = req.zh,
                                contentScale = ContentScale.Fit,
                                modifier = Modifier.fillMaxWidth().padding(20.dp),
                            )
                        }
                    }
                }

                /* ── ⑥ 手势层：捏合锚点 + 下滑速度投影 ── */
                Box(
                    Modifier
                        .fillMaxSize()
                        // ⚠️ 点按关闭必须和手势检测器在**同一个节点**：
                        // detectTransformGestures 会 consume 掉 down 事件，
                        // 放在父节点上的 clickable 就再也收不到 tap，"轻点关闭"会失效。
                        .clickable(
                            enabled = dragY.value == 0f && zoomV.value <= 1.01f,
                        ) { onDismiss() }
                        .pointerInput(Unit) {
                            // 捏合：scale 钳位 [1,3]；锚点跟手指；放大后才允许平移
                            detectTransformGestures { centroid, pan, zoomChange, _ ->
                                val s = (zoomV.value * zoomChange).coerceIn(1f, 3f)
                                if (s <= 1.001f) {
                                    zoomV.value = 1f; panX.value = 0f; panY.value = 0f
                                    anchorX.value = 0.5f; anchorY.value = 0.5f
                                } else {
                                    zoomV.value = s
                                    if (size.width > 0 && size.height > 0) {
                                        anchorX.value = (centroid.x / size.width).coerceIn(0f, 1f)
                                        anchorY.value = (centroid.y / size.height).coerceIn(0f, 1f)
                                    }
                                    panX.value += pan.x
                                    panY.value += pan.y
                                }
                            }
                        }
                        .pointerInput(Unit) {
                            // 下滑关闭：1x 时才生效（放大后不抢平移）；松手按速度投影判定
                            var vy = 0f
                            var lastT = 0L
                            detectVerticalDragGestures(
                                onDragStart = { vy = 0f; lastT = 0L },
                                onDragEnd = {
                                    if (zoomV.value > 1.001f) return@detectVerticalDragGestures
                                    val predicted = dragY.value + (if (reduce) 0f else vy / 5f)
                                    if (predicted > size.height * 0.5f) {
                                        scope.launch {
                                            dragY.animateTo(
                                                size.height.toFloat(),
                                                tween(220, easing = androidx.compose.animation.core.FastOutSlowInEasing),
                                            )
                                            onDismiss()
                                        }
                                    } else {
                                        scope.launch { dragY.animateTo(0f, tween(200)) }
                                    }
                                },
                                onDragCancel = {
                                    if (zoomV.value <= 1.001f) scope.launch { dragY.animateTo(0f, tween(200)) }
                                },
                            ) { change, dy ->
                                if (zoomV.value > 1.001f) return@detectVerticalDragGestures
                                change.consume()
                                val now = android.os.SystemClock.uptimeMillis()
                                if (lastT != 0L) {
                                    val dt = (now - lastT).coerceAtLeast(1L)
                                    vy = vy * 0.5f + (dy / dt * 1000f) * 0.5f   // px/s，指数平滑
                                }
                                lastT = now
                                scope.launch { dragY.snapTo((dragY.value + dy).coerceAtLeast(0f)) }
                            }
                        }
                )

                /* ── 题注栏 ── */
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 44.dp)
                        // ⑥ 导航/题注随进度反向平移，永远钉住不被拖走
                        .graphicsLayer {
                            val p = (dragY.value / 200f).coerceIn(0f, 1f)
                            translationY = -120f * p
                            alpha = 1f - p
                        },
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    if (req.zh.isNotBlank()) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(req.zh, fontSize = 17.sp, fontWeight = FontWeight.Bold, color = Color.White)
                            // 保护等级在前、危险等级在后（2026-10-11 用户定序）
                            ProtectionPills(req.sci, Modifier.padding(start = 6.dp))
                        }
                    }
                    if (req.sci.isNotBlank()) {
                        Text(
                            req.sci,
                            fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic,
                            fontSize = 13.sp, color = Color.White.copy(alpha = 0.6f),
                            modifier = Modifier.padding(top = 2.dp),
                        )
                    }
                    /* ── ⑦ 半边一：页码跟手 ──
                       现状只渲染 currentPage（Int 台阶，跨页时中点才跳一次）。
                       改成用连续量驱动，并让文字随跨页过程轻微位移+淡出。 */
                    if (paths.size > 1) {
                        val f = pager.currentPageOffsetFraction            // −0.5…0.5
                        val follow = if (reduce) 0f else kotlin.math.abs(f)  // 0…0.5
                        Text(
                            text = "${pager.currentPage + 1} / ${paths.size}",
                            fontSize = 11.sp,
                            color = Color.White.copy(alpha = (0.38f + 0.47f * (1f - follow * 2f)).coerceIn(0.38f, 0.85f)),
                            modifier = Modifier
                                .padding(top = 12.dp)
                                .graphicsLayer { translationX = if (reduce) 0f else f * 52f },
                        )
                    }
                    if (paths.size <= 1) {
                        Text(
                            "下滑或轻点关闭",
                            fontSize = 11.sp, color = Color.White.copy(alpha = 0.38f),
                            modifier = Modifier.padding(top = 12.dp),
                        )
                    }
                }
            }
        }
    }
}

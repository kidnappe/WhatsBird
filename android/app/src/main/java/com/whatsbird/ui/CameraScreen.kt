package com.whatsbird.ui

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.util.Log
import android.view.HapticFeedbackConstants
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.withFrameMillis
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.res.painterResource
import com.whatsbird.R
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.whatsbird.vision.Detection
import com.whatsbird.ui.theme.rememberReduceMotion
import com.whatsbird.vision.ModelRegistry
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.max
import kotlin.math.min
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine

@Composable
fun CameraScreen(
    modifier: Modifier = Modifier,
    viewModel: CameraViewModel = hiltViewModel(),
) {
    val ui by viewModel.ui.collectAsState()
    val savedCount by viewModel.savedCount.collectAsState()
    val manualRegion by viewModel.manualRegion.collectAsState()
    var regionMenu by remember { mutableStateOf(false) }

    // 记录成功的确认闪态（feedback.md：重要动作要确认完成）。
    // prev 从 -1 起：跳过首帧载入的历史条数，只对"本次会话新增"闪烁
    val view = LocalView.current
    var showConfirmSheet by remember { mutableStateOf(false) }
    // 快门快照：点「记录此鸟」那一刻冻结的裁剪图/候选/身份（CameraViewModel.beginRecord）
    var recordSnap by remember { mutableStateOf<CameraViewModel.RecordSnapshot?>(null) }
    var savedFlash by remember { mutableStateOf(false) }
    var prevSaved by remember { mutableStateOf(-1) }
    LaunchedEffect(savedCount) {
        if (prevSaved >= 0 && savedCount > prevSaved) {
            savedFlash = true
            delay(1800)
            savedFlash = false
        }
        prevSaved = savedCount
    }

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    // ⚠️ 2026-10-10：必须显式设 COMPATIBLE（内部走 TextureView）。
    // 默认 PERFORMANCE 走 SurfaceView，而 SurfaceView **不受 Compose 的
    // clip(RoundedCornerShape) 与 graphicsLayer 影响** —— 取景罩的圆角裁切失效、
    // 快门 punch 缩放只作用在叠加框上不作用在画面上，二者脱节。
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }
    var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var maxZoom by remember { mutableFloatStateOf(1f) }
    var minZoom by remember { mutableFloatStateOf(0.6f) }
    var zoomRatio by remember { mutableFloatStateOf(1f) }
    var autoZoom by remember { mutableStateOf(false) }
    // 进取景页的初始态跟设置页持久默认（页内芯片可临时覆盖，不回写设置）
    val autoZoomPref by viewModel.autoZoomDefault.collectAsState()
    LaunchedEffect(autoZoomPref) { autoZoom = autoZoomPref }
    var dialExpanded by remember { mutableStateOf(false) }

    // 性能监控叠加层（设置页开关，默认关）：电池温度每 2 秒读一次
    val ctx = LocalContext.current
    val perfOn by viewModel.perfMonitor.collectAsState()
    var batteryTemp by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(perfOn) {
        while (perfOn) {
            batteryTemp = readBatteryTempC(ctx)
            delay(2000)
        }
    }

    // docs/design/DESIGN.md §5.2：检测框"入锁"淡入时刻表 + 快门白闪/罩框脉冲
    val appearAt = remember { mutableStateMapOf<Int, Long>() }
    // ⚠️ 2026-10-10 新增：驱动淡入的逐帧时钟。
    // 原先在 Canvas 的 draw lambda 里直接读 System.currentTimeMillis()，那不是 State，
    // 不会触发重绘；而 ui 每 500ms 才发射一次 → 新框首帧 age=0 被钳到 0.3，
    // 要等下一次 ui 发射才跳到 1，表现为"暗框闪一下再突然变亮"，而不是 150ms 淡入。
    // 现在 Canvas 读 frameTick（快照依赖），由下面的 LaunchedEffect 在淡入期内逐帧推进。
    var frameTick by remember { mutableLongStateOf(0L) }
    LaunchedEffect(ui.boxes) {
        val nowT = System.currentTimeMillis()
        for (b in ui.boxes) appearAt.putIfAbsent(b.trackId, nowT)
        appearAt.keys.retainAll(ui.boxes.map { it.trackId }.toSet())
        frameTick = nowT
        // 只在确有框处于 150ms 淡入期内时逐帧推进，结束后自动停，不空转烧电
        var guard = 0
        while (guard++ < 32) {
            val t = System.currentTimeMillis()
            if (ui.boxes.none { t - (appearAt[it.trackId] ?: t) < 150L }) break
            frameTick = t
            withFrameMillis { }
        }
    }
    val reduceMotion = rememberReduceMotion()
    var flashA by remember { mutableFloatStateOf(0f) }
    val punch = remember { Animatable(1f) }
    LaunchedEffect(savedFlash) {
        if (savedFlash) {
            flashA = 0.85f; delay(70); flashA = 0f
            if (!reduceMotion) {
                punch.snapTo(1f)
                punch.animateTo(0.985f, tween(60))
                punch.animateTo(1f, spring(dampingRatio = Spring.DampingRatioMediumBouncy))
            }
        }
    }
    var focusLocked by remember { mutableStateOf(false) }
    var focusPoint by remember { mutableStateOf(Offset.Zero) }
    var lastMetered by remember { mutableStateOf(Offset.Zero) }

    fun focusAt(viewX: Float, viewY: Float) {
        val point = previewView.meteringPointFactory.createPoint(viewX, viewY)
        camera?.cameraControl?.startFocusAndMetering(
            FocusMeteringAction.Builder(
                point,
                FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE,
            ).disableAutoCancel().build(),
        )
        focusLocked = true
        focusPoint = Offset(viewX, viewY)
        lastMetered = Offset(viewX, viewY)
    }

    // 变焦：逻辑多摄内部自动切换超广/主摄/2.6x 长焦，超出光学范围走 ISP 数字变焦
    fun applyZoom(target: Float) {
        val limit = maxOf(min(maxZoom, 30f), minZoom + 0.1f)
        val clamped = target.coerceIn(minZoom, limit)
        if (clamped != zoomRatio) viewModel.onZoomChanged(clamped) // 位置/尺度突变 → 轨迹重投影+渐进扫描
        zoomRatio = clamped
        camera?.cameraControl?.setZoomRatio(zoomRatio)
    }

    val scope = rememberCoroutineScope()
    // JCA ZoomStateManager 改编版：互斥锁 + 挡位动画（见 ZoomAnimator.kt）
    val zoomAnimator = remember {
        ZoomAnimator(initialZoom = 1f, zoomRange = 0.6f..30f, onZoomRatio = { applyZoom(it) })
    }
    LaunchedEffect(maxZoom, minZoom) {
        zoomAnimator.updateRange(minZoom..min(maxZoom, 30f))
    }

    // 追焦：锁定后每 600ms 看一眼最高分检测框——鸟跑出 4% 视野且还在锁定点
    // 30% 邻域内就重新对焦，跳到别的目标上不追
    LaunchedEffect(camera, focusLocked) {
        while (focusLocked) {
            val r = viewModel.ui.value
            val viewW = previewView.width.toFloat()
            val viewH = previewView.height.toFloat()
            if (viewW > 0 && viewH > 0) {
                r.detections.maxByOrNull { it.score }?.let { best ->
                    val c = detCenterInView(best, r.frameW, r.frameH, viewW, viewH)
                    val maxSide = max(viewW, viewH)
                    val drift = (c - lastMetered).getDistance()
                    if (drift > maxSide * 0.04f &&
                        (c - focusPoint).getDistance() < maxSide * 0.30f
                    ) {
                        focusAt(c.x, c.y)
                    }
                }
            }
            delay(600)
        }
    }

    // 摄像机绑定后读取变焦范围
    LaunchedEffect(camera) {
        while (camera != null && maxZoom <= 1f) {
            camera?.cameraInfo?.zoomState?.value?.let { zs ->
                if (zs.maxZoomRatio > 1f) {
                    maxZoom = zs.maxZoomRatio
                    minZoom = zs.minZoomRatio
                }
            }
            delay(200)
        }
    }

    // 自动对鸟放大：框高小于画面 20% 就逐步拉近，大于 40% 就退回，中间死区防震荡
    LaunchedEffect(camera, autoZoom) {
        while (autoZoom) {
            viewModel.ui.value.detections.maxByOrNull { it.score }?.let { best ->
                val h = best.bottom - best.top
                // 自动放大只到 10x：再远数字变焦噪到没有识别价值，且容易跟丢
                if (h < 0.20f) {
                    scope.launch { zoomAnimator.absoluteZoom(min(zoomRatio, 10f) * 1.12f) }
                } else if (h > 0.40f) {
                    scope.launch { zoomAnimator.absoluteZoom(zoomRatio / 1.12f) }
                }
            }
            delay(250)
        }
    }

    LaunchedEffect(lifecycleOwner) {
        val provider = suspendCancellableCoroutine { cont ->
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener({
                try {
                    cont.resume(future.get())
                } catch (t: Throwable) {
                    cont.resumeWithException(t)
                }
            }, ContextCompat.getMainExecutor(context))
        }
        cameraProvider = provider

        val preview = Preview.Builder().build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val analysis = ImageAnalysis.Builder()
            // 分析流分辨率拉到 1080p（无则选最接近的可用档）：1x 广角下远处小鸟的
            // 像素数是检测下限的决定因素，默认 640x480 时小目标信息物理不存在
            .setResolutionSelector(
                androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        androidx.camera.core.resolutionselector.ResolutionStrategy(
                            android.util.Size(1920, 1080),
                            androidx.camera.core.resolutionselector.ResolutionStrategy
                                .FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                        )
                    )
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        analysis.setAnalyzer(analyzerExecutor) { proxy ->
            try {
                // ⚠️ 2026-10-10：先问有没有空位再转码。相机 30fps、推理约 4fps，
                // 原先无条件 toBitmap()+rotated()（1080p 约 8MB/帧）之后才由 onFrame
                // 判断 busy 丢弃 → 每秒白转 26 张大图，纯给 GC 加压。
                // 这里忙就直接丢帧，proxy 仍由下面的 finally 关闭。
                if (!viewModel.frameSlotFree()) return@setAnalyzer
                val bmp = proxy.toBitmap()
                val rotation = proxy.imageInfo.rotationDegrees
                val frame = if (rotation != 0) {
                    val r = bmp.rotated(rotation)
                    bmp.recycle() // 旋转出来的新图才是要用的，中间这张别留着占内存
                    r
                } else {
                    bmp
                }
                viewModel.onFrame(frame)
            } catch (t: Throwable) {
                Log.e(TAG, "帧处理失败", t)
            } finally {
                proxy.close()
            }
        }

        // ⚠️ 2026-10-10 修视野不一致：分析流是 16:9（1920x1080），Preview 没设分辨率、
        // 默认 4:3 —— 两路裁出的范围不同，UI 里的 FILL_CENTER 映射据此画框必然偏，
        // 点选命中与追焦中心也跟着偏。用 ViewPort + UseCaseGroup 让两路共享同一裁切。
        // viewPort 在 PreviewView 完成布局前为 null，此时退回旧绑定方式（行为不变）。
        val viewPort = previewView.viewPort
        cameraProvider?.let { p ->
            val group = androidx.camera.core.UseCaseGroup.Builder()
                .addUseCase(preview)
                .addUseCase(analysis)
                .apply { viewPort?.let { setViewPort(it) } }
                .build()
            p.unbindAll()
            camera = p.bindToLifecycle(
                lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, group,
            )
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            // ⚠️ 2026-10-10：原先直接 unbindAll()，分析器可能还挂着任务；
            // 先让相机卸下 use case，再关线程池，避免任务投到已关闭的 executor 上。
            cameraProvider?.unbindAll()
            analyzerExecutor.shutdown()
        }
    }

    // ══ 方案 D「取景罩框」（docs/design/DESIGN.md）══
    // 画面 = 一张夹进图鉴的圆角取景照片；顶栏/滚轮/结果面板全部坐在纸边上，
    // HUD 可读性与场景亮度彻底解耦（纯白玻璃直叠画面的塌陷问题从根上消失）。
    val t = paperT()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(t.paper)
            .padding(horizontal = 12.dp),
    ) {
        // 顶部工具行（纸边带）
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(
                    top = WindowInsets.statusBars.asPaddingValues()
                        .calculateTopPadding() + 12.dp,
                    bottom = 10.dp,
                ),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box {
                // 区域选项展示名 = 区名（新七区，来自 REGION_LABELS）+ labels 资产行数运行时派生。
                // 换模型类数变了 UI 自动跟上；行数读不到降级为纯区名（ModelRegistry log.w）。
                // 不用 regionDisplayName()：它内部取 ModelRegistry.REGION_NAMES（旧六区名）。
                val regionOptions = remember(context) {
                    listOf<Pair<String?, String>>(null to "自动（GPS 定位）") +
                        REGION_KEYS.map { key ->
                            val name = REGION_LABELS[key] ?: key
                            val n = ModelRegistry.labelsAssetOf(key)
                                ?.let { ModelRegistry.labelCount(context, it) }
                            key to (if (n != null) "$name $n 类" else name)
                        }
                }
                GlassChip(
                    label = "区域 · ${manualRegion?.let { REGION_LABELS[it] } ?: "自动"}",
                    active = manualRegion != null,
                    onClick = { regionMenu = true },
                )
                DropdownMenu(
                    expanded = regionMenu,
                    onDismissRequest = { regionMenu = false },
                ) {
                    regionOptions.forEach { (value, label) ->
                        DropdownMenuItem(
                            text = { Text(label) },
                            trailingIcon = {
                                if (value == manualRegion) {
                                    Icon(Icons.Filled.Check, contentDescription = null)
                                }
                            },
                            onClick = {
                                viewModel.setRegion(value)
                                regionMenu = false
                            },
                        )
                    }
                }
                // ⚠️ 这里**曾经**多放过一个「检测 · 当前档位」的 GlassChip（2026-10-10 换挡功能）。
                //    已移除，原因是它同时坏了两件事：
                //      ① `GlassChip` 带 `.clickable{}`，`onClick={}` 是个**能点但什么都不做的死芯片**，
                //         属于"看起来能点其实不能"的假可供性；
                //      ② 它和「区域 chip + DropdownMenu」同在这个 Box 里，Box 默认纵向堆叠 ⇒
                //         **Box 高度翻倍**，顶栏整体下移；而 `DropdownMenu` 锚在 Box 原点，
                //         高度一变**地区下拉菜单的弹出位置就错位**。
                //    档位切换现在**只允许在设置页**；当前档位查 `adb logcat -s YoloDetector:I | grep 档位`。
            }
            // 性能监控叠加层（设置页开关）：横排，夹在「区域」与「自动放大」两个芯片中间。
            // Row 是 SpaceBetween 三子项 → 左/中/右各一，正好落在中间。
            if (perfOn) {
                PerfOverlay(
                    inferMs = ui.inferMs,
                    fps = ui.fps,
                    batteryTempC = batteryTemp,
                )
            }
            GlassChip(
                label = "自动放大",
                active = autoZoom,
                onClick = { autoZoom = !autoZoom },
            )
        }

        // 取景窗口：手势全部落在窗内（点纸边不误对焦）；检测框/对焦环映射尺寸 = 窗口尺寸，自洽
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .graphicsLayer { scaleX = punch.value; scaleY = punch.value }
                .clip(RoundedCornerShape(22.dp))
                .border(1.dp, t.line, RoundedCornerShape(22.dp))
                // 点按对焦并锁定；点在非主框上 = 切换主目标（多鸟）；滚轮展开时点空白处 = 收起滚轮；长按取消锁定
                .pointerInput(camera) {
                    detectTapGestures(
                        onTap = { offset ->
                            if (dialExpanded) {
                                dialExpanded = false
                            } else {
                                // 多鸟切换：命中非主框（外扩 12dp 容差）→ 换主目标；对焦照常生效
                                val u = viewModel.ui.value
                                val viewW = previewView.width.toFloat()
                                val viewH = previewView.height.toFloat()
                                if (viewW > 0 && viewH > 0) {
                                    val hit = u.boxes.firstOrNull { b ->
                                        !b.primary && pointInBox(
                                            offset, b.det, u.frameW, u.frameH, viewW, viewH, pad = 12f,
                                        )
                                    }
                                    if (hit != null) viewModel.selectTrack(hit.trackId)
                                }
                                focusAt(offset.x, offset.y)
                            }
                        },
                        onLongPress = { focusLocked = false },
                    )
                }
                // 双指捏合变焦（手动操作会关掉自动放大）
                .pointerInput(camera) {
                    detectTransformGestures { _, _, zoom, _ ->
                        if (zoom != 1f) {
                            autoZoom = false
                            scope.launch { zoomAnimator.scaleZoom(zoom) }
                        }
                    }
                },
        ) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

            // 检测框叠加层（多鸟）：主目标带识别角标（结果面板同源身份），
            // 其余轨迹带首分类 top1 小角标。深色晕圈 + 亮色描边双描边：亮天空/干草地上黄框不隐形
            Canvas(modifier = Modifier.fillMaxSize()) {
                for (b in ui.boxes) {
                    val label = if (b.primary) {
                        // 封顶 99%：观鸟语境没有"100% 确定"（与结果面板同口径）
                        ui.identity?.let { it.zh to min(0.99f, it.prob) }
                    } else {
                        b.zh?.let { it to min(0.99f, b.prob) }
                    }
                    // 读 frameTick 建立快照依赖：它推进即重绘（原先读 currentTimeMillis 不触发重绘）
                    val tNow = if (frameTick > 0L) frameTick else System.currentTimeMillis()
                    val ageA = ((tNow - (appearAt[b.trackId] ?: tNow)) / 150f)
                        .coerceIn(0.3f, 1f)
                    drawDetection(b.det, ui.frameW, ui.frameH, label, minor = !b.primary, alpha = ageA)
                }
                if (focusLocked && focusPoint != Offset.Zero) {
                    drawCircle(
                        Color.Black.copy(alpha = 0.55f), radius = 20f, center = focusPoint,
                        style = Stroke(width = 7f),
                    )
                    drawCircle(
                        Color.White, radius = 20f, center = focusPoint,
                        style = Stroke(width = 3f),
                    )
                    drawCircle(Color.White, radius = 3f, center = focusPoint)
                }
            }
            // 快门白闪（80ms）：取景器"咔嚓"一帧
            if (flashA > 0f) {
                Box(Modifier.matchParentSize().background(Color.White.copy(alpha = flashA)))
            }
        }

        Spacer(Modifier.height(10.dp))

        // 底部控制带（纸边）：变焦滚轮 + 识别结果面板。
        // 底部避让 = 悬浮导航胶囊（高 56 + 底距 10 + 间隙 10）= 76dp
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(
                    bottom = WindowInsets.navigationBars.asPaddingValues()
                        .calculateBottomPadding() + 76.dp,
                ),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            // 滚轮变焦（向上拱形刻度盘，状态层为 JCA ZoomAnimator）
            ZoomWheel(
                zoomRatio = zoomRatio,
                expanded = dialExpanded,
                onExpandChange = { dialExpanded = it },
                onDragZoom = { factor ->
                    autoZoom = false
                    scope.launch { zoomAnimator.scaleZoom(factor) }
                },
                onZoomTo = { target ->
                    autoZoom = false
                    scope.launch { zoomAnimator.animatedZoom(target) }
                },
            )
            // 确认式记录：保存前从 Top-5 候选点选实际种类（数据质量铁律）。
            // 候选来自快门快照：取鸟/定名在按钮按下那一刻定格，弹窗期间不随推理漂移
            if (showConfirmSheet) {
                ConfirmRecordSheet(
                    candidates = recordSnap?.candidates ?: emptyList(),
                    defaultSci = recordSnap?.identity?.sci,
                    onDismiss = {
                        showConfirmSheet = false
                        recordSnap = null
                        viewModel.cancelRecord()
                    },
                    onConfirm = { sci ->
                        showConfirmSheet = false
                        recordSnap = null
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                            view.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
                        }
                        viewModel.saveCurrent(sci)
                    },
                )
            }
            // 识别结果面板（docs/design/DESIGN.md §3.3 状态机驱动，单帧结果永不直接上屏）
            ResultPanel(
                ui = ui,
                savedFlash = savedFlash,
                onSave = {
                    recordSnap = viewModel.beginRecord()
                    showConfirmSheet = true
                },
            )
        }
    }
}

/** 预览用 FILL_CENTER（等比放大、居中裁切），检测框按同一映射并钳制在屏内。
 *  主目标粗框 + 13sp 角标；其余轨迹（minor）细框 + 11sp 角标。
 *  双描边：外圈深色晕圈保证任意场景背景下的可读性（accessibility.md：不依赖单一通道）。 */
private fun DrawScope.brackets(
    l: Float, tp: Float, r: Float, b: Float,
    color: Color, w: Float, len: Float, alpha: Float,
) {
    val lx = minOf(len, (r - l) / 2.2f)
    val ly = minOf(len, (b - tp) / 2.2f)
    fun ln(a: Offset, c: Offset) = drawLine(color, a, c, strokeWidth = w, alpha = alpha)
    ln(Offset(l, tp), Offset(l + lx, tp)); ln(Offset(l, tp), Offset(l, tp + ly))
    ln(Offset(r, tp), Offset(r - lx, tp)); ln(Offset(r, tp), Offset(r, tp + ly))
    ln(Offset(l, b), Offset(l + lx, b)); ln(Offset(l, b), Offset(l, b - ly))
    ln(Offset(r, b), Offset(r - lx, b)); ln(Offset(r, b), Offset(r, b - ly))
}

private fun DrawScope.drawDetection(
    d: Detection,
    frameW: Int,
    frameH: Int,
    label: Pair<String, Float>? = null,
    minor: Boolean = false,
    alpha: Float = 1f,
) {
    if (frameW == 0 || frameH == 0) return
    val scale = max(size.width / frameW, size.height / frameH)
    val dx = (size.width - frameW * scale) / 2f
    val dy = (size.height - frameH * scale) / 2f
    // 钳制在屏幕内（描边不越界）
    val m = 8f
    val left = (dx + d.left * frameW * scale).coerceIn(m, size.width - m)
    val top = (dy + d.top * frameH * scale).coerceIn(m, size.height - m)
    val right = (dx + d.right * frameW * scale).coerceIn(m, size.width - m)
    val bottom = (dy + d.bottom * frameH * scale).coerceIn(m, size.height - m)
    if (right - left < 4f || bottom - top < 4f) return

    // 四角括号（MOTION §1"入锁"）：黑晕宽衬底 + 仪表黄括号，比全框更"瞄准"
    val wMain = (if (minor) 1.5f else 2.5f).dp.toPx()
    val len = minOf(right - left, bottom - top) * 0.26f
    brackets(left, top, right, bottom, Color.Black.copy(alpha = 0.55f), wMain + 3.5f, len, alpha)
    brackets(left, top, right, bottom, BoxYellow, wMain, len, alpha)
    // 右上角识别角标：名称 + 概率（半透明黑底，钳制在屏内，贴框外上方）
    if (label != null) {
        val lp = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = (if (minor) 11f else 13f).dp.toPx(); isAntiAlias = true
        }
        lp.alpha = (255 * alpha).toInt()
        val text = "${label.first} %.0f%%".format(label.second * 100)
        val tw = lp.measureText(text); val lh = lp.textSize; val pad = 4.dp.toPx()
        var bx = right - (tw + pad * 2)
        var by = top - (lh + pad * 2) - 2.dp.toPx()
        if (by < 8f) by = top + 2.dp.toPx()
        if (bx < 8f) bx = 8f
        drawRect(Color.Black.copy(alpha = 0.55f * alpha),
                 topLeft = Offset(bx, by), size = Size(tw + pad * 2, lh + pad * 2))
        drawIntoCanvas { c ->
            c.nativeCanvas.drawText(text, bx + pad, by + pad + lh * 0.8f, lp)
        }
    }
}

/** 检测框中心 → 预览视图坐标（与 drawDetection 的 FILL_CENTER 映射一致）。 */
private fun detCenterInView(
    d: Detection,
    frameW: Int,
    frameH: Int,
    viewW: Float,
    viewH: Float,
): Offset {
    if (frameW == 0 || frameH == 0) return Offset.Zero
    val scale = max(viewW / frameW, viewH / frameH)
    val dx = (viewW - frameW * scale) / 2f
    val dy = (viewH - frameH * scale) / 2f
    return Offset(
        dx + (d.left + d.right) / 2f * frameW * scale,
        dy + (d.top + d.bottom) / 2f * frameH * scale,
    )
}

/** 检测框 → 预览视图坐标矩形（同 drawDetection 映射），外扩 [pad]px 触控容差；
 *  点选切换主目标的命中判定用。 */
private fun pointInBox(
    p: Offset,
    d: Detection,
    frameW: Int,
    frameH: Int,
    viewW: Float,
    viewH: Float,
    pad: Float,
): Boolean {
    if (frameW == 0 || frameH == 0) return false
    val scale = max(viewW / frameW, viewH / frameH)
    val dx = (viewW - frameW * scale) / 2f
    val dy = (viewH - frameH * scale) / 2f
    val l = dx + d.left * frameW * scale - pad
    val t = dy + d.top * frameH * scale - pad
    val r = dx + d.right * frameW * scale + pad
    val b = dy + d.bottom * frameH * scale + pad
    return p.x in l..r && p.y in t..b
}

private fun Bitmap.rotated(degrees: Int): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}

private const val TAG = "CameraScreen"

/** 确认式记录弹窗：Top-5 候选点选（数据质量铁律）。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConfirmRecordSheet(
    candidates: List<TrackIdentity>,
    defaultSci: String?,
    onDismiss: () -> Unit,
    onConfirm: (String) -> Unit,
) {
    androidx.compose.material3.ModalBottomSheet(onDismissRequest = onDismiss) {
        Text(
            "确认这条记录的种类",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(horizontal = 20.dp),
        )
        Text(
            "点选你实际看到的鸟 · 系统会记下你的确认用于改进识别",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 6.dp),
        )
        if (candidates.isEmpty()) {
            Text(
                "暂无候选（识别中…）",
                modifier = Modifier.padding(20.dp),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            candidates.forEach { c ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onConfirm(c.sci) }
                        .padding(horizontal = 20.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(Modifier.size(18.dp), contentAlignment = Alignment.Center) {
                        if (c.sci == defaultSci) {
                            Image(
                                painter = painterResource(R.drawable.ic_star),
                                contentDescription = "默认候选",
                                modifier = Modifier.size(15.dp),
                                colorFilter = androidx.compose.ui.graphics.ColorFilter
                                    .tint(Color(0xFFFBBF24)),
                            )
                        }
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            c.zh,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            c.sci,
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Text(
                        "%.0f%%".format(c.prob * 100),
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
            }
        }
        androidx.compose.foundation.layout.Spacer(Modifier.height(24.dp))
    }
}

/** 区域路由选择项键序（展示名在组合时经 ModelRegistry.regionDisplayName 派生，value = null 表示自动）。
 *
 *  ⚠️ 2026-10-10 改用**七区一级制**（东北/华北/蒙新/青藏/西南/华中/华南，口径见 ZooRegions）。
 *  展示名走新体系，但**下发值仍是旧的模型 key**，所以 `manualRegion` 的持久化语义
 *  和 ModelRegistry.labelsAssetOf 的路由一行都不用改。
 *
 *  - 蒙新区 / 青藏区**没有区域模型**，按约定**不出现在下拉里**（用户在西北本就落全国兜底模型）。
 *  - 华南区 / 西南区**共用一个 huanan_sw 模型**，所以两个都列出来、各自指向同一个模型。
 *  - 「华东」保留在末尾：它在七区里没有对应的一级区（鲁苏皖浙赣闽被拆进华中/华南），
 *    但模型还在、且仍是最优选择，删掉会造成功能倒退。标成「覆盖区」以示区别。
 */
private val REGION_KEYS = listOf(
    "dongbei",   // 东北区
    "huabei",    // 华北区
    "xinan",     // 西南区 → huanan_sw
    "huazhong",  // 华中区
    "huanan",    // 华南区 → huanan_sw
    "huadong",   // 华东（覆盖区，非一级区，见上）
    "cn373",     // 全国兜底
)

/** 新体系一级区显示名（旧 key → 新名）。 */
private val REGION_LABELS = mapOf(
    "dongbei" to "东北区",
    "huabei" to "华北区",
    "xinan" to "西南区",
    "huazhong" to "华中区",
    "huanan" to "华南区",
    "huadong" to "华东",
    "cn373" to "全国兜底",
)

// ============ 取景页设计 token（v4 起并入纸感体系）============
// HUD 材质/文字取 LocalPaperPalette（五套配色 + 夜纸派生自动跟随）；
// 仅检测框黄保留为固定仪表色（框在真实画面上，不参与 UI 配色）。
private val BoxYellow = Color(0xFFFFD60A)            // 检测框（仪表色，只给框用）
@Composable private fun paperT() = com.whatsbird.ui.theme.LocalPaperPalette.current.t

/** 相机顶部工具芯片：黑玻璃底白字，激活态反白（相机 app 模式开关惯例）。
 *  含内边距高 ≈40dp，高于移动端控件 28dp 下限。 */
@Composable
private fun GlassChip(
    label: String,
    active: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val t = paperT()
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(20.dp))
            .background(if (active) t.moss else t.card.copy(alpha = 0.92f))
            .border(1.dp, if (active) t.moss else t.line, RoundedCornerShape(20.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 10.dp),
    ) {
        Text(
            label,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
            color = if (active) t.card else t.olive,
        )
    }
}

/**
 * 「找鸟中 / 识别中」指示器：一圈 800ms 匀速旋转的圆环。
 *
 * 规格出自 `docs/design/DESIGN.md` §5.2「识别中」一行——原文是**两个故意不同频**的循环
 * （球 600ms 上下 + 环 800ms 转圈，抄 SwiftUI_Demos 的 `DribbleAnimatedView.swift`，
 * 「同频会像心电图，不同频才像在干活」）。**2026-10-12 定稿只保留圆环**，球与文字都不动。
 * 理由是取景页同时跑硬件推理，少一个循环少一份叠加负担（`layer-map.html` ⑨ 的硬约束）。
 *
 * ⚠️ 顺带更正一处文档失实：项目内的动效收割清单写着
 * 「⑨ 已落成 `CameraScreen:813/883`」，但 `git log -S drawArc / repeatable( / InfiniteTransition`
 * 在本文件全历史零命中——那组动效**从未真正进过代码**，本函数是首次落地。
 * 参数（800ms / 90° 弧头 / 18dp / 2dp 描边）先做过并排对比 demo 再落地；
 * 对比页在 `docs/design/preview-paper/`，属设计预览，**不随公开树分发**
 * （`publish_public.ps1` 会整目录剔除该路径），所以这里只记结论不给链接。
 *
 * 实现约束：`drawArc` 只读一个动画角度，不写状态、不触发布局测量，
 * 因此不进主线程布局路径（同 `PaperShimmer` 的思路）。
 * 减弱动效开启时**不挂** `rememberInfiniteTransition`，只画静止的环——状态仍表达，但不转。
 */
@Composable
private fun SearchingRing(track: Color, head: Color, modifier: Modifier = Modifier) {
    val reduce = rememberReduceMotion()
    val sweepDeg = if (reduce) {
        0f
    } else {
        val loop = rememberInfiniteTransition(label = "searchRing")
        val v by loop.animateFloat(
            initialValue = 0f,
            targetValue = 360f,
            animationSpec = infiniteRepeatable(
                animation = tween(durationMillis = 800, easing = LinearEasing),
                repeatMode = RepeatMode.Restart,
            ),
            label = "sweep",
        )
        v
    }
    Canvas(modifier.size(18.dp)) {
        val stroke = 2.dp.toPx()
        val d = size.minDimension
        val topLeft = Offset(
            (size.width - d) / 2f + stroke / 2f,
            (size.height - d) / 2f + stroke / 2f,
        )
        val arc = Size(d - stroke, d - stroke)
        val cap = Stroke(width = stroke, cap = StrokeCap.Round)
        drawArc(track, 0f, 360f, false, topLeft, arc, style = cap)
        drawArc(head, sweepDeg, 90f, false, topLeft, arc, style = cap)
    }
}

/** 底部识别结果面板：深色玻璃层，状态机驱动切换内容。
 *  高度随内容自适应（不再钉死 206dp，Dynamic Type 大字号下不截字）。 */
@Composable
private fun ResultPanel(
    ui: ObserveUiState,
    savedFlash: Boolean,
    onSave: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val shape = RoundedCornerShape(20.dp)
    val t = paperT()
    Box(
        modifier = modifier
            .fillMaxWidth()
            .animateContentSize(tween(220))
            .shadow(8.dp, shape, ambientColor = t.ink.copy(alpha = 0.20f), spotColor = t.ink.copy(alpha = 0.25f))
            .background(t.card, shape)
            .border(1.dp, t.line, shape)
            .padding(horizontal = 16.dp, vertical = 14.dp),
    ) {
        when (ui.state) {
            IdState.IDLE -> Row(verticalAlignment = Alignment.CenterVertically) {
                // 2026-10-12：原先是静态 6dp 灰点，现换成 800ms 旋转环（见 SearchingRing 注释）
                SearchingRing(track = t.line, head = t.moss)
                Spacer(Modifier.width(10.dp))
                Text("找鸟中…", fontSize = 14.sp, color = t.olive)
            }
            IdState.TRACKING -> Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    SearchingRing(track = t.line, head = t.moss)
                    Spacer(Modifier.width(10.dp))
                    Text("识别中…", fontSize = 14.sp, color = t.olive)
                }
                Spacer(Modifier.height(10.dp))
                ResultBar(fraction = ui.charging, fill = t.moss)
            }
            IdState.IDENTIFIED, IdState.LOST -> {
                val id = ui.identity
                if (id != null) {
                    // P0：本分支的行此前是 Box 的兄弟子项（全部叠放同一位置），必须包进 Column
                    Column {
                        val lost = ui.state == IdState.LOST
                        // LOST：名字变暗而非消失——记忆还在，但不误导"鸟还在"
                        val nameAlpha by animateFloatAsState(
                            targetValue = if (lost) 0.6f else 1f,
                            animationSpec = tween(220),
                            label = "nameAlpha",
                        )
                        // 定名瞬间（签名动效，motion.md：短促、有目的）：淡入+8dp 上移落定，换名重放
                        val appear = remember(id.zh) { Animatable(if (lost) 1f else 0f) }
                        LaunchedEffect(id.zh) {
                            if (!lost) appear.animateTo(1f, tween(220, easing = FastOutSlowInEasing))
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    // 中文名是本产品的主角：28sp 粗体、允许两行不截字
                                    Text(
                                        id.zh,
                                        fontSize = 28.sp,
                                        lineHeight = 34.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = t.ink,
                                        maxLines = 2,
                                        overflow = TextOverflow.Ellipsis,
                                        // weight(fill=false)：无 weight 的保护徽章先量、拿足自身宽度，
                                        // 名字再吃剩余空间——名字过长时折行/省略的是名字，徽章不会被挤没。
                                        modifier = Modifier
                                            .weight(1f, fill = false)
                                            .graphicsLayer {
                                                alpha = nameAlpha * appear.value
                                                translationY = (1f - appear.value) * 8.dp.toPx()
                                            },
                                    )
                                    // 保护等级在前、危险等级在后（2026-10-11 用户定序）
                                    ProtectionPills(id.sci, Modifier.padding(start = 8.dp))
                                }
                                // 学名走鸟类图鉴的排版惯例：衬线斜体
                                Text(
                                    id.sci,
                                    fontSize = 14.sp,
                                    fontFamily = FontFamily.Serif,
                                    fontStyle = FontStyle.Italic,
                                    color = t.gray,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                )
                            }
                            Spacer(Modifier.width(12.dp))
                            // "记录此鸟" = 取景器的快门时刻：白底药丸，确认后闪绿
                            // 确认反馈：容器色 白↔绿 200ms + 落章式轻脉冲（已有 CONFIRM 触觉）
                            val saveContainer by animateColorAsState(
                                targetValue = if (savedFlash) t.mossDeep else t.moss,
                                animationSpec = tween(200),
                                label = "saveContainer",
                            )
                            val saveScale by animateFloatAsState(
                                targetValue = if (savedFlash) 1.04f else 1f,
                                animationSpec = spring(
                                    dampingRatio = Spring.DampingRatioMediumBouncy,
                                    stiffness = Spring.StiffnessMedium,
                                ),
                                label = "saveScale",
                            )
                            Button(
                                onClick = onSave,
                                enabled = !savedFlash,
                                shape = RoundedCornerShape(24.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = saveContainer,
                                    contentColor = t.card,
                                    disabledContainerColor = saveContainer,
                                    disabledContentColor = t.card,
                                ),
                                contentPadding = PaddingValues(horizontal = 18.dp),
                                modifier = Modifier
                                    .height(48.dp)
                                    .graphicsLayer { scaleX = saveScale; scaleY = saveScale },
                            ) {
                                if (savedFlash) {
                                    Image(
                                        painter = painterResource(R.drawable.ic_check),
                                        contentDescription = null,
                                        modifier = Modifier.size(15.dp),
                                        colorFilter = androidx.compose.ui.graphics.ColorFilter
                                            .tint(t.card),
                                    )
                                }
                                Text(
                                    if (savedFlash) "已记录" else "记录此鸟",
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Medium,
                                )
                            }
                        }
                        Spacer(Modifier.height(12.dp))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.weight(1f)) {
                                ResultBar(
                                    fraction = id.prob,
                                    fill = if (lost) t.moss.copy(alpha = 0.55f) else t.moss,
                                    height = 4.dp,
                                )
                            }
                            Spacer(Modifier.width(10.dp))
                            // 展示封顶 99%：观鸟语境里没有"100% 确定"
                            Text(
                                "%.0f%%".format(min(id.prob, 0.99f) * 100),
                                fontSize = 12.sp,
                                color = t.olive,
                            )
                        }
                        if (ui.secondary.isNotEmpty() || lost) {
                            Spacer(Modifier.height(8.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                            ) {
                                Text(
                                    "其他可能：${ui.secondary.joinToString(" · ")}",
                                    fontSize = 12.sp,
                                    color = t.gray,
                                    modifier = Modifier.weight(1f),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                )
                                if (lost) {
                                    Spacer(Modifier.width(8.dp))
                                    Text("鸟已出画", fontSize = 12.sp, color = t.gray)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/** 圆角进度条：手绘 Box 版（M3 LinearProgressIndicator 在 <8dp 高时视觉发虚）。 */
@Composable
private fun ResultBar(
    fraction: Float,
    fill: Color,
    height: Dp = 8.dp,
) {
    val tt = paperT()
    val animated by animateFloatAsState(
        targetValue = fraction.coerceIn(0f, 1f),
        animationSpec = tween(300),
        label = "resultBar",
    )
    Box(
        Modifier
            .fillMaxWidth()
            .height(height)
            .clip(RoundedCornerShape(height / 2))
            .background(tt.line),
    ) {
        if (animated > 0.005f) {
            Box(
                Modifier
                    .fillMaxWidth(animated)
                    .fillMaxHeight()
                    .clip(RoundedCornerShape(height / 2))
                    .background(fill),
            )
        }
    }
}


package com.whatsbird

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.Spring
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Icon
import androidx.compose.material3.Surface
import androidx.compose.material3.Text

import androidx.compose.animation.core.snap
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInParent
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntOffset
import kotlin.math.roundToInt
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.whatsbird.data.ThemePrefs
import com.whatsbird.ui.CameraScreen
import com.whatsbird.ui.RecordsScreen
import com.whatsbird.ui.SettingsScreen
import com.whatsbird.ui.theme.WhatsBirdTheme
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

/**
 * 「DataStore 还没读到」的哨兵值。
 *
 * ⚠️ 2026-10-12 白屏事故：原先 `collectAsState(initial = null)` 拿 null 当"未就绪"，
 * 而 `ThemePrefs.paperPalette` 在**用户从没设过配色**时也稳定发 null（键不存在）。
 * 两者撞在一起 → 渲染守卫 `if (paperKey == null) Box(空)` 永远成立 →
 * 全新安装进 App 是一整屏纸底色，且因为是 `else if` 短路，连相机权限弹窗都不弹、
 * 底部导航也不画，用户在 App 内无路可走。
 *
 * 现在用哨兵把"未加载"和"已加载但未设置"分开：真实色板 key 只有
 * moss/walnut/mist/indigo/cocoa（PaperPalettes），不会撞上这个值；
 * null 保留给后者，`PaperPalettes.byKey(null)` 会正常回退苔绿默认色板。
 */
private const val PALETTE_LOADING = "__loading__"

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject
    lateinit var themePrefs: ThemePrefs

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            // 外观双维：五套纸感色板 × 昼夜（夜间三态：跟随系统/强制/关闭，docs/design/DESIGN.md）
            val paperRaw by themePrefs.paperPalette.collectAsState(initial = PALETTE_LOADING)
            // 未就绪 vs 已就绪但用户没设过——见 PALETTE_LOADING 的注释。
            val paletteLoaded = paperRaw != PALETTE_LOADING
            val paperKey = if (paletteLoaded) paperRaw else null
            val nightPref by themePrefs.nightMode.collectAsState(initial = null)
            val dark = when (nightPref) {
                "on" -> true
                "off" -> false
                else -> androidx.compose.foundation.isSystemInDarkTheme() // system
            }
            WhatsBirdTheme(
                palette = com.whatsbird.ui.theme.PaperPalettes.byKey(paperKey),
                dark = dark,
                // 2026-10-10：偏好未就绪的那一帧不参与逐色动画（治冷启动主题闪烁）
                animate = paletteLoaded,
            ) {
                val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t
                // 根 Surface 显式给色：修设置页行文字"继承深色 contentColor 而发淡"的问题
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = t.paper,
                    contentColor = t.ink,
                ) {
                    // 2026-10-10：外观偏好还在读的时候**不渲染任何内容**。
                    // 原先先用默认色板把整页画出来，等 DataStore 到了再换色板重画，
                    // 用户会看到"默认配色闪一下变自己的配色"。这里只留一层底色。
                    // ⚠️ 2026-10-12：判据从 paperKey == null 改成 !paletteLoaded——
                    // 键不存在（全新安装）时 paperKey 恒为 null，用旧判据会永远卡在这层空 Box，
                    // 表现为一整屏纸底色、连相机权限弹窗都不弹，App 内无法自救。见 PALETTE_LOADING。
                    if (!paletteLoaded) {
                        Box(Modifier.fillMaxSize())
                    } else if (grantedState()) {
                        MainTabs()
                    } else {
                        PermissionDenied()
                    }
                    // ① 换色圆形洇开（v6 P0-1）：盖在最上层，从点中的圆盘中心扩散
                    com.whatsbird.ui.PaletteWipeOverlay(com.whatsbird.ui.LocalPaletteWipe.current)
                }
            }
        }
    }

    @Composable
    private fun grantedState(): Boolean {
        var granted by remember {
            mutableStateOf(
                ContextCompat.checkSelfPermission(
                    this, Manifest.permission.CAMERA,
                ) == PackageManager.PERMISSION_GRANTED,
            )
        }
        val launcher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission(),
        ) { granted = it }
        LaunchedEffect(Unit) {
            if (!granted) launcher.launch(Manifest.permission.CAMERA)
        }
        // 定位权限：区域路由需要（华北框内用区域模型）。拒绝则自动回退全国模型
        var locGranted by remember {
            mutableStateOf(
                ContextCompat.checkSelfPermission(
                    this, Manifest.permission.ACCESS_COARSE_LOCATION,
                ) == PackageManager.PERMISSION_GRANTED,
            )
        }
        val locLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission(),
        ) { locGranted = it }
        LaunchedEffect(granted) {
            if (granted && !locGranted) {
                locLauncher.launch(Manifest.permission.ACCESS_COARSE_LOCATION)
            }
        }
        return granted
    }
}

@Composable
private fun MainTabs() {
    // ⚠️ 2026-10-10 改 rememberSaveable：原先用 remember，屏幕旋转后 tab 回到识别页。
    var tab by rememberSaveable { mutableIntStateOf(0) }
    // ⚠️ 2026-10-10 新增：非识别页按返回键原先直接退出 app。
    // 现在统一回识别页；已在识别页时交回系统（正常退出）。
    BackHandler(enabled = tab != 0) { tab = 0 }
    // 取景页已并入纸感体系（v4）：导航胶囊全 app 统一"卡面玻璃 + 线描边"，随昼夜派生
    val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t
    Box(modifier = Modifier.fillMaxSize()) {
        // 动效规范 §4：tab 切换 150ms 交叉淡入（不做左右滑页）
        AnimatedContent(
            targetState = tab,
            transitionSpec = {
                fadeIn(tween(150)) togetherWith fadeOut(tween(150))
            },
            label = "tabCrossfade",
        ) { cur ->
            when (cur) {
                0 -> CameraScreen(modifier = Modifier.fillMaxSize())
                1 -> RecordsScreen(modifier = Modifier.fillMaxSize())
                2 -> com.whatsbird.ui.GuideScreen(modifier = Modifier.fillMaxSize())
                else -> SettingsScreen(modifier = Modifier.fillMaxSize())
            }
        }
        // v3 导航：56dp 悬浮图标胶囊（docs/design/DESIGN.md §2.2）——
        // 图标语言：filled 选中 / outlined 未选中（设置项与识别页同为双 filled），
        // 卡面玻璃 + line 描边、内容包裹宽度居中
        // ⑦ 半边二：Row 换成 NavTabPill，选中底色由**一块共享指示块**滑过去
        Box(
            modifier = Modifier
                .align(androidx.compose.ui.Alignment.BottomCenter)
                .padding(bottom = 10.dp)
                .clip(androidx.compose.foundation.shape.RoundedCornerShape(28.dp))
                // ⚠️ 2026-10-10 按用户要求去掉导航栏"阴影"。
                // 这里本来就没有 Modifier.shadow()（外层 Surface 也是默认 elevation 0），
                // 看着像阴影的是两样东西叠出来的：
                //   ① .background(t.card.copy(alpha = 0.92f)) —— 留 8% 透明，
                //      页面内容滚过去会从底下透出来一圈，像一层脏影/拖影；
                //   ② .border(1.dp, t.line, 28dp) —— 这圈描边在深色页面上读作暗边。
                // 现改为**完全不透明 + 无描边**。要恢复任一项说一声。
                .background(t.card),
        ) {
            // ⚠️ 2026-10-10 修正：原先是内联 `listOf(...)`，**每次重组都新建 4 个 lambda**，
            // Compose 无法跳过任何 tab 子树 → 每次切 tab 全量重组。
            // 更糟的是：指示块的测量曾依赖这次"被迫重新布局"才能拿到新位置
            // （现已改为测全部 4 个，不再依赖）。这里用 remember 按 tab 缓存。
            val tabs = remember(tab) {
                listOf<@Composable () -> Unit>(
                    {
                        NavTab(
                            selected = tab == 0, label = "识别",
                            icon = if (tab == 0) R.drawable.ic_eye_filled else R.drawable.ic_eye_outline,
                            onClick = { tab = 0 },
                        )
                    },
                    {
                        NavTab(
                            selected = tab == 1, label = "日志",
                            icon = if (tab == 1) R.drawable.ic_bookmark_filled else R.drawable.ic_bookmark_outline,
                            onClick = { tab = 1 },
                        )
                    },
                    {
                        NavTab(
                            selected = tab == 2, label = "图鉴",
                            icon = R.drawable.ic_life, onClick = { tab = 2 },
                        )
                    },
                    {
                        NavTab(
                            selected = tab == 3, label = "设置",
                            icon = R.drawable.ic_settings, onClick = { tab = 3 },
                        )
                    },
                )
            }
            NavTabPill(selectedIndex = tab, tabs = tabs)
        }
    }
}

/**
 * ⑦ 半边二 · 底部导航的**共享滑动指示块**（v5 §4 早就写了，一直没做）
 *
 * 抄 AnimatedSegmentedControl：原先每个 tab 各自 `background(indColor)`，
 * 切 tab 时四块底色同时瞬变、彼此不相关。现在只有一块 sage 底在选中 tab 底下滑动。
 * 与 `PaperChipRow` 同款手法：位置走 spring，宽度用带延迟的 tween → 拉伸感。
 */
@Composable
private fun NavTabPill(
    selectedIndex: Int,
    tabs: List<@Composable () -> Unit>,
) {
    val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t
    val reduce = com.whatsbird.ui.theme.rememberReduceMotion()
    val n = tabs.size
    val idx = if (n == 0) 0 else selectedIndex.coerceIn(0, n - 1)

    // ⚠️ 2026-10-10 修正：改成**测全部 n 个**再按下标取。
    // 原先只在 `i == selectedIndex` 时写测量值，等于依赖"每次切 tab 都会重新布局"
    // 才能拿到新位置——而那次重新布局是靠上游 `tabs` 每次重组都新建 lambda
    // （性能 bug）硬逼出来的。一旦给 tabs 加 remember 优化，指示块就会当场卡死。
    val xs = remember(n) { List(n) { mutableFloatStateOf(0f) } }
    val ws = remember(n) { List(n) { mutableFloatStateOf(0f) } }
    val hs = remember(n) { List(n) { mutableFloatStateOf(0f) } }

    val posSpec: androidx.compose.animation.core.AnimationSpec<Float> = if (reduce) {
        snap()
    } else {
        // ⚠️ 2026-10-10：原先沿用筛选标签那套 MediumBouncy + StiffnessMediumLow，
        // 摆动能到 350ms+；而页面交叉淡入淡出只有 150ms。结果是**页面早切完了，
        // 指示块还在来回摆**，读起来就是拖影。筛选标签没有页面切换所以 350ms 无妨，
        // 导航栏有，就得跟内容对齐 → 收到 ~200ms、只留一点点回弹。
        spring(
            dampingRatio = Spring.DampingRatioLowBouncy,
            stiffness = Spring.StiffnessMedium,
        )
    }
    val sizeSpec: androidx.compose.animation.core.AnimationSpec<Float> = if (reduce) {
        snap()
    } else {
        tween(durationMillis = 160, delayMillis = 30, easing = androidx.compose.animation.core.FastOutSlowInEasing)
    }
    val pillX by animateFloatAsState(xs[idx].value, posSpec, label = "navPillX")
    val pillW by animateFloatAsState(ws[idx].value, sizeSpec, label = "navPillW")
    val pillH = hs[idx].value
    // 指示块上下各内缩 5dp（与 NavTab 自己的 vertical padding 对齐，圆角才和胶囊外框齐平）。
    // ⚠️ 内缩只作用在**指示块**上：早前把它加在 tab 槽位上，把槽位从 56dp 压到 46dp，
    // 而 NavTab 内容需要 48dp → 文字被挤出并被胶囊 clip 切掉。槽位保持满高。
    val density = LocalDensity.current
    val pillInset = with(density) { 5.dp.toPx() }

    Box(
        Modifier
            .height(56.dp)
            .padding(horizontal = 6.dp)
    ) {
        if (pillW > 0f && pillH > 0f) {
            Box(
                Modifier
                    .offset { IntOffset(pillX.roundToInt(), pillInset.toInt()) }
                    .width(with(density) { pillW.toDp() })
                    .height(with(density) { (pillH - pillInset * 2).coerceAtLeast(1f).toDp() })
                    .clip(androidx.compose.foundation.shape.RoundedCornerShape(22.dp))
                    .background(t.sage)
            )
        }
        Row(
            Modifier.fillMaxHeight(),
            horizontalArrangement = Arrangement.spacedBy(2.dp),
            verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
        ) {
            tabs.forEachIndexed { i, tab ->
                Box(
                    modifier = Modifier
                        .fillMaxHeight()
                        .onGloballyPositioned { c ->
                            xs[i].value = c.positionInParent().x
                            ws[i].value = c.size.width.toFloat()
                            hs[i].value = c.size.height.toFloat()
                        },
                    contentAlignment = androidx.compose.ui.Alignment.Center,
                ) { tab() }
            }
        }
    }
}

@Composable
private fun NavTab(
    selected: Boolean,
    label: String,
    icon: Int,
    onClick: () -> Unit,
) {
    val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t
    // 动效规范 §4：选中指示块颜色渐移 + 轻微放大（胶囊形变简化版）
    // ⑦ 半边二：选中底色交给共享指示块（NavTabPill 画），这里只留文字/图标颜色与轻微放大
    val sel by animateFloatAsState(
        if (selected) 1f else 0f,
        // ⚠️ 2026-10-10 修正「切换有阴影拖影」：原先是
        // spring(DampingRatioMediumBouncy)——阻尼比 0.2 很弹、摆动能到 400ms+，
        // 而页面交叉淡入淡出和图标变色都只有 150ms。结果是**页面早就切完了，
        // 选中的那格还在膨胀收缩**，读起来像拉出一条残影。
        // 现在与页面/颜色同步到 150ms、不过冲，拖影随之消失。
        tween(durationMillis = 150, easing = androidx.compose.animation.core.FastOutSlowInEasing),
        label = "sel",
    )
    val fg by animateColorAsState(if (selected) t.moss else t.gray, tween(150), label = "fg")
    Column(
        modifier = Modifier
            .graphicsLayer { scaleX = 1f + 0.05f * sel; scaleY = 1f + 0.05f * sel }
            .clip(androidx.compose.foundation.shape.RoundedCornerShape(22.dp))
            // ⚠️ 2026-10-10 按用户要求去掉点选涟漪。
            // 这里是裸 `.clickable(onClick = ...)`，用的是默认 indication（ripple）：
            // 每点一下图标周围就泛开一圈再淡出，在这条窄窄的胶囊上读起来像"点选阴影"。
            // 选中反馈已经由共享指示块 + 变色 + 轻微放大承担，不需要再叠一层涟漪。
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = onClick,
            )
            .padding(horizontal = 18.dp, vertical = 5.dp),
        horizontalAlignment = androidx.compose.ui.Alignment.CenterHorizontally,
    ) {
        Icon(
            painter = painterResource(icon),
            contentDescription = label,
            tint = fg,
            modifier = Modifier.size(21.dp),
        )
        // 文字与图标的间距：2dp → 1dp
        Spacer(Modifier.height(1.dp))
        Text(
            label,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            color = fg,
        )
    }
}

@Composable
private fun PermissionDenied() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = androidx.compose.ui.Alignment.Center) {
        Text("需要相机权限才能识别鸟类")
    }
}

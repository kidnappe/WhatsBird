package com.whatsbird.ui.theme

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.AnimationSpec
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.positionInParent
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt
import com.whatsbird.R

/**
 * 纸感配色体系（docs/design/DESIGN.md §2.3/§2.5）：
 * 全 app（取景/图鉴/日志/设置）统一吃五套手选 PaperPalette——
 * 不吃 Material You 动态色、无种子色（2026-10-08 拍板移除）；
 * S0 苔绿默认，S1-S4 在设置页「外观 → 配色方案」圆盘选择器切换；
 * 夜间模式由 PaperThemeState.dark 控制，tokens 经 night() 派生。
 */
@Immutable
data class PaperTokens(
    val paper: Color,
    val card: Color,
    val ink: Color,
    val moss: Color,
    val mossDeep: Color,
    val moss2: Color,
    val sage: Color,
    val olive: Color,
    val amber: Color,
    val red: Color,
    val gray: Color,
    val line: Color,
    val silA: Color,
    val silB: Color,
) {
    /** 剪影/水印用灰（三套色温都协调） */
    val silInk: Color get() = Color(0xFFB9B4A3)
    /**
     * 夜间派生：同一套色板压深成"夜纸"——纸底混黑保留色相、卡面略亮于纸底、
     * 墨字转米白、主色提亮（暗底 primary）、点缀微提亮保持可读；
     * 五套配色共用此公式，夜间观感一致（不另手调五套暗色）。
     */
    fun night(): PaperTokens {
        val black = Color(0xFF0E100C)
        val white = Color(0xFFF3F1E7)
        return PaperTokens(
            paper = lerp(paper, black, 0.905f),
            card = lerp(paper, black, 0.855f),
            ink = lerp(ink, white, 0.88f),
            moss = lerp(moss, white, 0.60f),
            mossDeep = lerp(moss, white, 0.42f),
            moss2 = lerp(moss, white, 0.72f),
            sage = lerp(sage, black, 0.78f),
            olive = lerp(olive, white, 0.64f),
            amber = lerp(amber, white, 0.12f),
            red = lerp(red, white, 0.14f),
            gray = lerp(gray, white, 0.36f),
            line = lerp(line, black, 0.66f),
            silA = lerp(silA, black, 0.86f),
            silB = lerp(silB, black, 0.80f),
        )
    }

    private companion object {
        /** Color.lerp 的本地简写（compose graphics lerp 在 Color 上的成员式调用） */
        fun lerp(a: Color, b: Color, f: Float): Color = androidx.compose.ui.graphics.lerp(a, b, f)
    }
}

/** 全局主题状态：色板（日间定义）+ 夜间标志；t 解析出当前生效 tokens。 */
@Immutable
data class PaperThemeState(
    val palette: PaperPalette,
    val dark: Boolean,
    val overrideT: PaperTokens? = null,
) {
    val t: PaperTokens get() = overrideT ?: if (dark) palette.t.night() else palette.t
}

/** 命名色板（key 持久化在 ThemePrefs.paperPalette；hex 与预览页一一对应）。 */
@Immutable
data class PaperPalette(
    val key: String,
    val name: String,
    val tagline: String,
    val t: PaperTokens,
)

object PaperPalettes {
    private fun tok(
        paper: Long, ink: Long, moss: Long, mossDeep: Long, moss2: Long,
        sage: Long, olive: Long, amber: Long, red: Long, gray: Long,
        line: Long, silA: Long, silB: Long,
    ) = PaperTokens(
        paper = Color(paper), card = Color(0xFFFFFFFF), ink = Color(ink),
        moss = Color(moss), mossDeep = Color(mossDeep), moss2 = Color(moss2),
        sage = Color(sage), olive = Color(olive), amber = Color(amber),
        red = Color(red), gray = Color(gray), line = Color(line),
        silA = Color(silA), silB = Color(silB),
    )

    val MOSS = PaperPalette(
        "moss", "苔绿琥珀", "默认套：暖纸米白 × 深苔绿 × 琥珀点缀",
        tok(0xFFF7F4EC, 0xFF1F2A22, 0xFF3A5540, 0xFF2C4130, 0xFF5A7A5F,
            0xFFE9F0E6, 0xFF5B6B4F, 0xFFDDA45A, 0xFFE2483D, 0xFF8A9187,
            0xFFE4E0D4, 0xFFEFEBDF, 0xFFE4DFCF),
    )
    val WALNUT = PaperPalette(
        "walnut", "深胡桃铜", "复古博物，暖褐 × 黄铜",
        tok(0xFFF3EEE6, 0xFF2A2118, 0xFF5C4630, 0xFF42301E, 0xFF83623F,
            0xFFEFE4CE, 0xFF6E5B41, 0xFFB98A2F, 0xFFA63D2F, 0xFF97907F,
            0xFFE2D8C6, 0xFFF0E8D8, 0xFFE7DCC5),
    )
    val MIST = PaperPalette(
        "mist", "晨雾冷杉", "现代冷静，灰青 × 焦橙",
        tok(0xFFF2F4F5, 0xFF1B2429, 0xFF3E5C6B, 0xFF2C4450, 0xFF5D7F8F,
            0xFFE2ECEF, 0xFF4F6B77, 0xFFC2703D, 0xFFB23A48, 0xFF8A939A,
            0xFFDBE2E4, 0xFFEAEFED, 0xFFD9E2E4),
    )
    val INDIGO = PaperPalette(
        "indigo", "黛蓝山樱", "墨印和风，绀青 × 山樱",
        tok(0xFFF5F4F7, 0xFF23233A, 0xFF3C3F66, 0xFF2B2D4D, 0xFF5D6191,
            0xFFE7E6F2, 0xFF585A85, 0xFFD98E9E, 0xFFB33A4E, 0xFF8D8C9C,
            0xFFE1E0EA, 0xFFEEEDF4, 0xFFDDDCE8),
    )

    val COCOA = PaperPalette(
        "cocoa", "草绿可可", "打字机暖调，象牙白 × 杏可可（魔咖 K75 OG 实拍取色）",
        tok(0xFFF3EFE6, 0xFF2B2820, 0xFF6E8568, 0xFF57694F, 0xFF8FA285,
            0xFFDDE4D2, 0xFF6F7B5E, 0xFFD9A97C, 0xFFA8655A, 0xFF94896F,
            0xFFDFD6C2, 0xFFEDE7D8, 0xFFDCCFB6),
    )

    val ALL = listOf(MOSS, WALNUT, MIST, INDIGO, COCOA)
    val DEFAULT = MOSS

    fun byKey(key: String?): PaperPalette = ALL.firstOrNull { it.key == key } ?: DEFAULT
}

/** 当前主题状态（MainActivity 按 ThemePrefs 提供；未包裹时回退默认苔绿日间）。 */
val LocalPaperPalette = staticCompositionLocalOf { PaperThemeState(PaperPalettes.DEFAULT, false) }

/** tokens → M3 ColorScheme（日间 light / 夜间 dark；night primary 为亮色，onPrimary 用色板原深墨）。 */
fun PaperThemeState.scheme(): ColorScheme {
    val t = this.t
    val deepInk = palette.t.ink   // 色板日间的深墨字，做暗底 primary 上的文字
    return if (dark) darkColorScheme(
        primary = t.moss, onPrimary = deepInk,
        primaryContainer = t.sage, onPrimaryContainer = t.olive,
        secondary = t.olive, onSecondary = t.card,
        background = t.paper, onBackground = t.ink,
        surface = t.paper, onSurface = t.ink,
        surfaceContainer = t.card, surfaceContainerHigh = t.card, surfaceContainerHighest = t.card,
        surfaceVariant = t.sage, onSurfaceVariant = t.gray,
        outline = t.line, outlineVariant = t.line,
        error = t.red, onError = t.card,
    ) else lightColorScheme(
        primary = t.moss, onPrimary = t.card,
        primaryContainer = t.sage, onPrimaryContainer = deepInk,
        secondary = t.olive, onSecondary = t.card,
        background = t.paper, onBackground = t.ink,
        surface = t.paper, onSurface = t.ink,
        surfaceContainer = t.card, surfaceContainerHigh = t.card, surfaceContainerHighest = t.card,
        surfaceVariant = t.sage, onSurfaceVariant = t.gray,
        outline = t.line, outlineVariant = t.line,
        error = t.red, onError = t.card,
    )
}

/**
 * 纸感主题壳：按 LocalPaperPalette 状态（色板 + 昼夜）换 MaterialTheme。
 * 页根套一层 PaperTheme { ... } 即整页跟随；取景 HUD 直接读 LocalPaperPalette。
 */
@Composable
fun PaperTheme(content: @Composable () -> Unit) {
    val state = LocalPaperPalette.current
    MaterialTheme(colorScheme = state.scheme(), content = content)
}

// ==================== 纸感通用元件 ====================

/** 琥珀/主题点缀刻度条（区块标题前 4×15dp 小竖条，纸感收集册 section tick 口径）。 */
@Composable
fun PaperTick(modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .padding(end = 7.dp)
            .size(width = 4.dp, height = 15.dp)
            .background(LocalPaperPalette.current.t.amber, RoundedCornerShape(2.dp)),
    )
}

/** 收集册胶囊 chip（未选=sage 底 olive 字；选中=moss 底白字）。 */
@Composable
fun PaperChip(
    selected: Boolean,
    onClick: () -> Unit,
    label: String,
    modifier: Modifier = Modifier,
) {
    val t = LocalPaperPalette.current.t
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(50))
            .background(if (selected) t.moss else t.sage)
            .clickable(onClick = onClick)
            .padding(horizontal = 15.dp, vertical = 7.dp),
    ) {
        Text(
            label,
            fontSize = 12.5.sp,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
            color = if (selected) t.card else t.olive,
        )
    }
}

/**
 * ⑦ 半边二 · 带滑动指示块的 chip 行（v6 增补）
 *
 * 抄 SwiftUI_Demos/AnimatedSegmentedControl：原先每个 chip 各自瞬切底色
 * （`PaperChip`），点下去毫无反馈。改为**一块共享指示块**滑过去——
 * 指示块滑的时候宽度稍慢半拍（200ms + 40ms 延迟），读起来就是
 * "先拉伸跨过间隙、再收回一个 chip 宽"的两段感。
 *
 * - 位置用 spring（略带回弹），宽度用带延迟的 tween，两者不同步 = 拉伸感
 * - 减弱动态效果时两者都 snap（瞬移）
 * - 区域行需要横滚：包 horizontalScroll，指示块用 positionInParent 取位，
 *   天然跟随滚动内容
 *
 * ⚠️ 2026-10-10 真机反馈后修正：初版把 chip 改成固定 `height(30.dp)` + 内容居中，
 * **视觉与上一版不一致**（内边距丢了、字重位置偏）。现按用户要求
 * **动效保留、形式回到上一版**：chip 继续按 `PaperChip` 的 `padding(h15,v7)` 自然撑高，
 * 不写死高度；指示块只跟着实测尺寸走。
 */
@Composable
fun PaperChipRow(
    selectedIndex: Int,
    onSelect: (Int) -> Unit,
    labels: List<String>,
    modifier: Modifier = Modifier,
    scrollable: Boolean = false,
    contentPadding: PaddingValues = PaddingValues(horizontal = 2.dp, vertical = 2.dp),
) {
    val t = LocalPaperPalette.current.t
    val reduce = rememberReduceMotion()
    var measureX by remember { mutableFloatStateOf(0f) }
    var measureW by remember { mutableFloatStateOf(0f) }
    var measureH by remember { mutableFloatStateOf(0f) }

    val posSpec: AnimationSpec<Float> = if (reduce) {
        snap()
    } else {
        spring(dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessMediumLow)
    }
    // 宽度比位置慢半拍 → 指示块"先跨过间隙再收回"
    val sizeSpec: AnimationSpec<Float> = if (reduce) {
        snap()
    } else {
        tween(durationMillis = 200, delayMillis = 40, easing = androidx.compose.animation.core.FastOutSlowInEasing)
    }
    val pillX by animateFloatAsState(measureX, posSpec, label = "chipPillX")
    val pillW by animateFloatAsState(measureW, sizeSpec, label = "chipPillW")

    val rowModifier = modifier
        .then(
            if (scrollable) Modifier.horizontalScroll(rememberScrollState()) else Modifier
        )
        .padding(contentPadding)

    Box(rowModifier) {
        // 指示块：绝对定位在选中 chip 底下（尺寸全部来自实测，不写死）
        if (pillW > 0f && measureH > 0f) {
            Box(
                Modifier
                    .offset { IntOffset(pillX.roundToInt(), 0) }
                    .width(with(LocalDensity.current) { pillW.toDp() })
                    .height(with(LocalDensity.current) { measureH.toDp() })
                    .clip(RoundedCornerShape(50))
                    .background(t.moss)
            )
        }
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            labels.forEachIndexed { i, label ->
                val selected = i == selectedIndex
                // 与 PaperChip 完全一致的写法（含 padding），只是底色交给指示块
                Box(
                    modifier = Modifier
                        .onGloballyPositioned { c ->
                            // positionInParent 相对 Row，不受外层滚动影响
                            if (selected) {
                                measureX = c.positionInParent().x
                                measureW = c.size.width.toFloat()
                                measureH = c.size.height.toFloat()
                            }
                        }
                        .clip(RoundedCornerShape(50))
                        // 选中态底色交给指示块，这里只负责未选中的浅底
                        .background(if (selected) Color.Transparent else t.sage)
                        .clickable { onSelect(i) }
                        .padding(horizontal = 15.dp, vertical = 7.dp),
                ) {
                    Text(
                        label,
                        fontSize = 12.5.sp,
                        fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
                        color = if (selected) t.card else t.olive,
                        maxLines = 1,
                    )
                }
            }
        }
    }
}


/** 学名（衬线斜体，全 app 分类学排版口径）。 */
@Composable
fun PaperSciName(sci: String, modifier: Modifier = Modifier, sizePx: Int = 11) {
    val t = LocalPaperPalette.current.t
    Text(
        sci,
        modifier = modifier,
        fontFamily = FontFamily.Serif,
        fontStyle = androidx.compose.ui.text.font.FontStyle.Italic,
        fontSize = sizePx.sp,
        color = t.gray,
    )
}

/** 搜索框（白底胶囊 + 放大镜矢量图标；图鉴/日志共用）。 */
@Composable
fun PaperSearchField(
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    modifier: Modifier = Modifier,
) {
    val t = LocalPaperPalette.current.t
    androidx.compose.material3.OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier.fillMaxWidth(),
        singleLine = true,
        shape = RoundedCornerShape(24.dp),
        placeholder = { Text(placeholder, color = t.gray) },
        leadingIcon = {
            androidx.compose.foundation.Image(
                painter = androidx.compose.ui.res.painterResource(R.drawable.ic_search),
                contentDescription = null,
                modifier = Modifier.padding(start = 4.dp).size(20.dp),
                colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.gray),
            )
        },
        colors = androidx.compose.material3.OutlinedTextFieldDefaults.colors(
            focusedBorderColor = t.moss,
            unfocusedBorderColor = t.line,
            focusedContainerColor = t.card,
            unfocusedContainerColor = t.card,
            cursorColor = t.moss,
            focusedTextColor = t.ink,
            unfocusedTextColor = t.ink,
        ),
    )
}

/** 白卡（拍立得相纸/进度卡通用底：卡面=白，圆角 16-20）。 */
@Composable
fun PaperCardSurface(
    modifier: Modifier = Modifier,
    radius: Int = 18,
    content: @Composable () -> Unit,
) {
    val t = LocalPaperPalette.current.t
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(radius.dp))
            .background(t.card),
    ) { content() }
}

/**
 * IUCN 徽标胶囊（语义色固定，不随纸感主题漂移）。
 *
 * CR/EN/VU/NT 走**暖色告警梯度**（越高越深）；LC / NR 走**中性灰**——
 * 它们是"没事"和"没评过"，不能跟"快灭绝"抢同一套暖色。
 * ⚠️ 也**不能**用绿色：绿色会和 [CnProtectPill] 二级的 teal 撞车（行政级别那套是冷色）。
 *
 * ⚠️ 2026-10-11 起 LC/NR **会显示**（此前一律不画）。原因：徽章组要求
 * 「保护等级 + 危险等级」两枚一起出现，而 77 个受保护种里 **56 个 IUCN 是无危**，
 * 只画行政那枚会让人误以为"受保护的鸟 IUCN 那栏一定是红的"。
 * 空码/未知码仍不画。
 */
@Composable
fun IucnPill(code: String, modifier: Modifier = Modifier) {
    val (label, bg) = when (code) {
        "CR" -> "极危 CR" to Color(0xFFB91C1C)
        "EN" -> "濒危 EN" to Color(0xFFEA580C)
        "VU" -> "易危 VU" to Color(0xFFD97706)
        "NT" -> "近危 NT" to Color(0xFFB45309)
        "LC" -> "无危 LC" to Color(0xFF6B7280) // gray-500：中性，"没事"
        "NR" -> "未收录 NR" to Color(0xFF78716C) // stone-500：IUCN 未收录该种
        else -> return
    }
    Pill(label, bg, modifier)
}

/**
 * 国家重点保护级别徽标胶囊 —— **第二套颜色体系**，与 [IucnPill] 的暖色（红橙黄）刻意分开。
 *
 * 为什么不能共用一套：
 *  - IUCN 是**科学风险评估**，红色 = "快灭绝了"，天然要走红黄这套告警色；
 *  - 国家一/二级是**行政保护级别**，红黄色在中文语境里读作"危险 / 喜庆红"，
 *    拿来标"受保护"会误导 —— 实测 77 个受保护种里有 **59 个 IUCN 是无危 LC**，
 *    若和中华秋沙鸭（EN + 国家一级）一起画成红色，读者根本分不出差在哪。
 *  - 所以行政级别改走**冷色（靛蓝 / 青）**，读作"官方认定 / 身份标记"而非"危险"。
 * 一级比二级更深更饱和，与 [IucnPill] 内部"等级越高越深"的梯度保持一致直觉。
 * 「三有」是**更弱的一档**（《有重要生态、科学、社会价值的陆生野生动物名录》2023 版，
 * 鸟纲 1028 种，绝大多数常见鸟都在里面），所以走冷色系里更浅更蓝的一档。
 * ⚠️ 两名录**互斥**（三有名录明确排除重点保护种），一个种最多只有一个行政级别。
 *
 * 未列入名录（空字符串）不画徽章——同 [IucnPill]，满屏标签等于没标签。
 */
@Composable
fun CnProtectPill(level: String, modifier: Modifier = Modifier) {
    val (label, bg) = when (level) {
        "一级" -> "国家一级" to Color(0xFF3730A3) // indigo-800
        "二级" -> "国家二级" to Color(0xFF0F766E) // teal-700
        "三有" -> "三有" to Color(0xFF0891B2)     // cyan-600
        else -> return
    }
    Pill(label, bg, modifier)
}

/** 两套体系共用的胶囊外壳（只管形状与字号，语义色由调用方给）。 */
@Composable
private fun Pill(label: String, bg: Color, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(5.dp))
            .background(bg)
            .padding(horizontal = 5.dp, vertical = 1.dp),
    ) {
        Text(
            label,
            fontSize = 9.sp,
            fontWeight = FontWeight.Bold,
            color = if (bg.luminance() > 0.6f) Color(0xFF1F2A22) else Color.White,
            textAlign = TextAlign.Center,
        )
    }
}


// ==================== 动效基础设施（docs/design/DESIGN.md §5） ====================

/** 系统「减弱动态效果」（动画时长缩放=0）→ 全部动效降级为瞬变。 */
@Composable
fun rememberReduceMotion(): Boolean {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    return androidx.compose.runtime.remember(ctx) {
        android.provider.Settings.Global.getFloat(
            ctx.contentResolver, android.provider.Settings.Global.ANIMATOR_DURATION_SCALE, 1f,
        ) == 0f
    }
}

/** MOTION §0.2 时长档位：micro 150ms / component 250ms；减弱动效时 0。 */
@Composable
fun motionDur(micro: Boolean = false): Int = if (rememberReduceMotion()) 0 else if (micro) 150 else 250

/**
 * tokens 逐色动画（MOTION §4：换配色/昼夜 = 全 app 交叉渐变，治瞬变白闪）。
 * 只在根主题 WhatsBirdTheme 调用一次，各页经 LocalPaperPalette 拿到的即渐变值。
 */
@Composable
fun PaperTokens.animateForTheme(): PaperTokens {
    val spec: AnimationSpec<Color> = tween(motionDur())
    val vPaper by animateColorAsState(paper, spec, label = "tp")
    val vCard by animateColorAsState(card, spec, label = "tc")
    val vInk by animateColorAsState(ink, spec, label = "ti")
    val vMoss by animateColorAsState(moss, spec, label = "tm")
    val vMossDeep by animateColorAsState(mossDeep, spec, label = "tmd")
    val vMoss2 by animateColorAsState(moss2, spec, label = "tm2")
    val vSage by animateColorAsState(sage, spec, label = "ts")
    val vOlive by animateColorAsState(olive, spec, label = "to")
    val vAmber by animateColorAsState(amber, spec, label = "ta")
    val vRed by animateColorAsState(red, spec, label = "tr")
    val vGray by animateColorAsState(gray, spec, label = "tg")
    val vLine by animateColorAsState(line, spec, label = "tl")
    val vSilA by animateColorAsState(silA, spec, label = "tsa")
    val vSilB by animateColorAsState(silB, spec, label = "tsb")
    return PaperTokens(
        vPaper, vCard, vInk, vMoss, vMossDeep, vMoss2, vSage, vOlive,
        vAmber, vRed, vGray, vLine, vSilA, vSilB,
    )
}

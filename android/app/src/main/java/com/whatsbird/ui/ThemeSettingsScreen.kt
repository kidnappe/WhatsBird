/*
 * 外观设置（docs/design/DESIGN.md）：
 * - 夜间模式：三态（跟随系统 / 强制夜 / 关），开关行 + "跟随系统"联动
 * - 配色方案：五套纸感色板圆盘选择器（点击弹出，即时生效并持久化）
 * 历史"主题种子色/跟随壁纸"（Material You 动态取色）已于 2026-10-08 移除。
 */
package com.whatsbird.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.boundsInRoot
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.whatsbird.R
import com.whatsbird.data.ThemePrefs
import com.whatsbird.ui.theme.LocalPaperPalette
import com.whatsbird.ui.theme.PaperPalette
import com.whatsbird.ui.theme.PaperPalettes
import android.view.HapticFeedbackConstants
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.launch

/** 夜间三态值（与 ThemePrefs.nightMode 字符串一致）。 */
const val NIGHT_SYSTEM = "system"
const val NIGHT_ON = "on"
const val NIGHT_OFF = "off"

@HiltViewModel
class ThemeSettingsViewModel @Inject constructor(
    private val prefs: ThemePrefs,
) : ViewModel() {
    val paperPalette = prefs.paperPalette
    val nightMode = prefs.nightMode

    fun pickPaper(key: String) {
        viewModelScope.launch { prefs.setPaperPalette(key) }
    }

    fun setNight(mode: String) {
        viewModelScope.launch { prefs.setNightMode(mode) }
    }
}

/**
 * 外观分组内容（嵌进设置页「外观」卡片）：外层滚动与标题由 SettingsScreen 负责。
 */
@Composable
fun ThemeSettingsContent(
    modifier: Modifier = Modifier,
    viewModel: ThemeSettingsViewModel = hiltViewModel(),
) {
    val t = LocalPaperPalette.current.t
    val paperKey by viewModel.paperPalette.collectAsState(initial = null)
    val nightPref by viewModel.nightMode.collectAsState(initial = null)
    val wipe = LocalPaletteWipe.current
    val systemDark = androidx.compose.foundation.isSystemInDarkTheme()
    val mode = nightPref ?: NIGHT_SYSTEM
    // "夜"生效值：手动开 → true；跟随系统 → 系统标志；手动关 → false
    val nightActive = when (mode) {
        NIGHT_ON -> true
        NIGHT_OFF -> false
        else -> systemDark
    }
    val currentPalette = PaperPalettes.byKey(paperKey)
    var showPicker by rememberSaveable { mutableStateOf(false) }
    val view = LocalView.current

    fun haptic() = view.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)

    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(bottom = 6.dp),
    ) {
        // ===== 夜间模式（三态两开关）=====
        Row(
            Modifier
                .fillMaxWidth()
                .clickable {
                    haptic()
                    // 点击=手动指定与当前生效值相反的状态（覆盖"跟随系统"）
                    viewModel.setNight(if (nightActive) NIGHT_OFF else NIGHT_ON)
                }
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_moon),
                contentDescription = null,
                tint = t.olive,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("夜间模式", style = MaterialTheme.typography.bodyLarge, color = t.ink)
                Text(
                    "五套配色共用夜纸派生（压深底色、提亮主色）",
                    style = MaterialTheme.typography.bodySmall,
                    color = t.gray,
                )
            }
            Switch(
                checked = nightActive,
                onCheckedChange = { on ->
                    viewModel.setNight(if (on) NIGHT_ON else NIGHT_OFF)
                },
            )
        }
        Row(
            Modifier
                .fillMaxWidth()
                .clickable {
                    haptic()
                    if (mode == NIGHT_SYSTEM) {
                        // 关闭跟随：固化为当前生效值
                        viewModel.setNight(if (systemDark) NIGHT_ON else NIGHT_OFF)
                    } else {
                        viewModel.setNight(NIGHT_SYSTEM)
                    }
                }
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Spacer(Modifier.width(32.dp))
            Column(Modifier.weight(1f)) {
                Text("跟随系统", style = MaterialTheme.typography.bodyLarge, color = t.ink)
                Text(
                    if (mode == NIGHT_SYSTEM) "随 Android 深色设置自动切换"
                    else "当前为手动指定，不随系统变化",
                    style = MaterialTheme.typography.bodySmall,
                    color = t.gray,
                )
            }
            Switch(checked = mode == NIGHT_SYSTEM, onCheckedChange = null)
        }

        // ===== 配色方案（圆盘选择器）=====
        Row(
            Modifier
                .fillMaxWidth()
                .clickable {
                    haptic()
                    showPicker = true
                }
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_palette),
                contentDescription = null,
                tint = t.olive,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(12.dp))
            Text(
                "配色方案",
                style = MaterialTheme.typography.bodyLarge,
                color = t.ink,
                modifier = Modifier.weight(1f),
            )
            Text(
                currentPalette.name,
                style = MaterialTheme.typography.bodyMedium,
                color = t.gray,
            )
            Text(
                "  ›",
                style = MaterialTheme.typography.titleLarge,
                color = t.gray,
            )
        }
    }

    if (showPicker) {
        PalettePickerDialog(
            currentKey = currentPalette.key,
            onPick = { key, originInRoot ->
                haptic()
                // ① 圆形洇开：从点中的那枚圆盘中心扩散
                wipe.fire(originInRoot, PaperPalettes.byKey(key).t.paper)
                viewModel.pickPaper(key)
                showPicker = false
            },
            onDismiss = { showPicker = false },
        )
    }
}

/**
 * 配色方案选择弹层：三列圆盘格——右半圆主色、左上浅底、左下点缀，
 * 选中态在圆心压一枚深墨圆 + 白勾（视觉语言：一眼看出"这套色怎么配"）。
 *
 * ①（v6 P0-1）：选中时圆盘自身动起来——scale spring 弹到 1.06、阴影 0→8dp 走 spring、
 * 勾号 scaleIn(0.4→1)、底色 alpha 与字重走 animateColorAsState(200ms)。
 * 此前全部是瞬切，是全 app 视觉冲击最大的操作里最廉价的一处。
 * 配套的整屏圆形洇开见 `PaletteWipe.kt`。
 */
@Composable
private fun PalettePickerDialog(
    currentKey: String,
    /** onPick 带上圆盘在 **root 坐标系** 的中心，供圆形洇开定位 */
    onPick: (String, Offset) -> Unit,
    onDismiss: () -> Unit,
) {
    val t = LocalPaperPalette.current.t
    Dialog(onDismissRequest = onDismiss) {
        Card(
            shape = RoundedCornerShape(28.dp),
            colors = CardDefaults.cardColors(containerColor = t.card),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        ) {
            Column(Modifier.padding(horizontal = 24.dp, vertical = 24.dp)) {
                Text(
                    "配色方案",
                    fontSize = 21.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = t.ink,
                )
                Spacer(Modifier.height(18.dp))
                PaperPalettes.ALL.chunked(3).forEach { row ->
                    Row(Modifier.fillMaxWidth()) {
                        row.forEach { pal ->
                            val boxRef = remember { mutableStateOf<Rect?>(null) }
                            PaletteDiscItem(
                                pal = pal,
                                selected = pal.key == currentKey,
                                onClick = {
                                    // ① 把圆盘中心换算到 root 坐标，供洇开层定位
                                    boxRef.value?.let { r ->
                                        onPick(pal.key, Offset(r.center.x, r.center.y))
                                    } ?: onPick(pal.key, Offset.Zero)
                                },
                                modifier = Modifier
                                    .weight(1f)
                                    .onGloballyPositioned { boxRef.value = it.boundsInRoot() },
                            )
                        }
                        // 最后一行不满 3 个时占位对齐
                        repeat(3 - row.size) { Spacer(Modifier.weight(1f)) }
                    }
                    Spacer(Modifier.height(14.dp))
                }
            }
        }
    }
}

@Composable
private fun PaletteDiscItem(
    pal: PaperPalette,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val t = LocalPaperPalette.current.t
    val reduce = com.whatsbird.ui.theme.rememberReduceMotion()

    // ① 选中态全部走动画（此前瞬切）
    val scale by animateFloatAsState(
        targetValue = if (selected && !reduce) 1.06f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessMedium,
        ),
        label = "discScale",
    )
    // 浅底由 0.35 → 1 渐变（未选中保持原来的压暗感）
    val ringAlpha by animateFloatAsState(
        targetValue = if (selected) 1f else 0.35f,
        animationSpec = tween(if (reduce) 0 else 200, easing = FastOutSlowInEasing),
        label = "discRing",
    )
    val shadowDp by animateFloatAsState(
        targetValue = if (selected) 8f else 0f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioLowBouncy, stiffness = Spring.StiffnessMedium),
        label = "discShadow",
    )
    val labelColor by animateColorAsState(
        targetValue = if (selected) t.ink else t.gray,
        animationSpec = tween(if (reduce) 0 else 200),
        label = "discLabel",
    )
    // 勾号 scaleIn(0.4 → 1)：用 animatedVisibility 式的手写比例，避免依赖实验 API
    var checkShown by remember { mutableStateOf(false) }
    LaunchedEffect(selected) {
        if (selected && !reduce) { checkShown = true }
    }
    val checkScale = remember { Animatable(0.4f) }
    LaunchedEffect(selected, reduce) {
        if (selected && !reduce) {
            checkScale.snapTo(0.4f)
            checkScale.animateTo(1f, spring(dampingRatio = Spring.DampingRatioMediumBouncy))
        } else if (selected) {
            checkScale.snapTo(1f)
        } else {
            checkShown = false
            checkScale.snapTo(0.4f)
        }
    }

    Column(
        modifier = modifier.clickable(onClick = onClick),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            Modifier
                .size(76.dp)
                .graphicsLayer { scaleX = scale; scaleY = scale }
                .clip(RoundedCornerShape(20.dp))
                .background(t.sage.copy(alpha = ringAlpha))
                .shadow(shadowDp.dp, RoundedCornerShape(20.dp)),
            contentAlignment = Alignment.Center,
        ) {
            val p = pal.t
            Canvas(Modifier.size(52.dp)) {
                val rad = size.minDimension / 2f
                drawCircle(p.sage)
                // 右半圆：主色（起点 12 点顺时针 180°）
                drawArc(p.moss, startAngle = -90f, sweepAngle = 180f, useCenter = true)
                // 左下 1/4：点缀
                drawArc(p.amber, startAngle = 90f, sweepAngle = 90f, useCenter = true)
                // 左上 1/4：纸底（加描边防与白卡糊在一起）
                drawArc(p.paper, startAngle = 180f, sweepAngle = 90f, useCenter = true)
                drawCircle(p.line, radius = rad, style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5f))
                if (selected) {
                    drawCircle(p.mossDeep, radius = rad * 0.42f)
                }
            }
            if (checkShown) {
                Icon(
                    Icons.Filled.Check,
                    contentDescription = "已选",
                    tint = Color.White,
                    modifier = Modifier
                        .size(17.dp)
                        .graphicsLayer {
                            val s = checkScale.value
                            scaleX = s; scaleY = s
                        },
                )
            }
        }
        Spacer(Modifier.height(6.dp))
        Text(
            pal.name,
            fontSize = 12.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
            color = labelColor,
        )
    }
}

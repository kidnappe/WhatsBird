/*
 * Adapted from google/jetpack-camera-app `ZoomBarComponents` (Apache License 2.0)
 * Copyright (C) 2025 The Android Open Source Project
 * https://github.com/google/jetpack-camera-app
 *
 * Modifications:
 * - removed JCA ui-state / LensFacing / test-tag dependencies, plain parameters
 * - JCA 原版用 material3 1.5-alpha 的 ToggleButton + Expressive 尺寸 API（要求
 *   compileSdk 36+）；本项目的 material3 1.3 没有该组件，按钮壳换成官方
 *   FilterChip，视觉规格（药丸容器、选中实底色、实时倍率文本）与 JCA 一致
 */
package com.whatsbird.ui

import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * 相机变焦挡位药丸（google/jetpack-camera-app 设计）：
 * 半透明圆角药丸上一排定焦挡位；选中挡实时显示当前倍率（如 "2.5x"），
 * 其余挡显示固定值。
 *
 * ⚠️ 2026-10-10 修正三处真问题：
 *
 * 1. **高亮扫过整排**：高亮原本是从"当前倍率落在哪个区间"反推的，而倍率是
 *    `ZoomAnimator.animatedZoom` 的 400ms 补间、每帧都变。点 0.6x 时高亮会
 *    先亮 2x 再亮 1x 才落回来。现改为 `pinnedLevel` —— 点击后把高亮**钉死**
 *    在目标挡，倍率真正到位（偏差 < 0.05）后再释放。
 * 2. **亮着的格子显示的数字和标签对不上**：高亮扫到「1x」那一格的瞬间，
 *    它显示的是实时值 `%.1fx`（0.9x / 0.8x…），于是"标签写着 1、里面写着 0.7x"。
 *    钉住高亮后这个现象消失；补间途中显示爬升中的倍率反而是有用的反馈。
 * 3. **药丸宽度反向抖动**：选中显示 `"%.1fx"`（4 字符）、未选中把整数去尾零
 *    （`"1"` 只有 1 字符），切换瞬间两格一涨一缩；而 `animateContentSize()`
 *    又加在 Text 上、默认是**带过冲的 spring**，两条反向弹簧同时跑 → 药丸像在呼吸。
 *    现统一成**等宽文案**（未选中补一个空格凑成 4 字符），宽度不再变化；
 *    `animateContentSize` 挪到 FilterChip 上并换成无过冲的 tween。
 *
 * @param pinnedLevel 点击后钉住的目标挡；null = 交回给区间推导（= 原 JCA 逻辑）
 */
@Composable
fun ZoomButtonRow(
    currentZoom: Float,
    zoomLevels: List<Float>,
    onChangeZoom: (Float) -> Unit,
    modifier: Modifier = Modifier,
    buttonSize: Dp = 36.dp,
    spacing: Dp = 12.dp,
    pinnedLevel: Float? = null,
) {
    val current by rememberUpdatedState(currentZoom)

    // ⚠️ 2026-10-10 二次修正：原先写的是 `remember(zoomLevels) { derivedStateOf { ...pinnedLevel... } }`。
    // `pinnedLevel` 是**普通参数**不是 State，`derivedStateOf` 不会观察它，
    // 缓存也不会因为它变化而失效。直接算，少一层缓存少一个坑。
    val selectedOptionIndex: Int = remember(currentZoom, pinnedLevel, zoomLevels) {
        val pinned = pinnedLevel
        when {
            // 钉住期间一律高亮目标挡，补间途中不再被中间挡位抢走
            pinned != null -> zoomLevels.indexOfFirst { it == pinned }.takeIf { it >= 0 } ?: 0
            // 小于 1x 一律选中超广挡（index 0），否则取 ≤ 当前倍率的最近挡
            currentZoom >= 1f -> zoomLevels.indexOfLast { currentZoom >= it }
            else -> 0
        }
    }

    val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t
    Row(
        modifier = modifier
            .clip(CircleShape)  // 方案 D：药丸带坐在纸边上，容器透明，挡位各自成胶囊
            .padding(horizontal = spacing / 2, vertical = spacing / 2)
            .height(buttonSize),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        zoomLevels.forEachIndexed { index, value ->
            Box(
                modifier = Modifier.width(buttonSize + spacing + 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                ZoomButton(
                    buttonSize = buttonSize,
                    targetZoom = value,
                    zoomRatio = current,
                    isSelected = selectedOptionIndex == index,
                    pinnedLevel = pinnedLevel,
                    onChangeZoom = onChangeZoom,
                )
            }
        }
    }
}

@Composable
private fun ZoomButton(
    buttonSize: Dp,
    targetZoom: Float,
    zoomRatio: Float,
    isSelected: Boolean,
    onChangeZoom: (Float) -> Unit,
    /** 点选过渡期间显示的"目标倍率"；null = 显示实时倍率（捏合时用） */
    pinnedLevel: Float? = null,
) {
    val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t
    // 等宽文案：选中 "%.1fx" = 4 字符，未选中 "%.1f " = 4 字符（尾空格）。
    // 两态宽度完全一致 → 切换时药丸不做任何宽度动画，不会有"呼吸感"。
    //
    // ⚠️ 2026-10-10 二次修正：钉住期间**显示目标倍率**，不是补间途中的实时值。
    // 否则点 0.6x（当前 2.6x）时，被点亮的那格会在 400ms 里依次显示
    // "2.5x / 2.1x / 1.6x …"——从用户视角看，跟修复前一模一样：
    // 点了 0.6，却看到 2.5 亮着。到达后实时值就等于目标值，数字也不会跳。
    val shownZoom = pinnedLevel ?: zoomRatio
    val displayText = if (isSelected) {
        "%.1fx".format(shownZoom)
    } else {
        "%.1f ".format(targetZoom)
    }
    FilterChip(
        selected = isSelected,
        onClick = { onChangeZoom(targetZoom) },
        // animateContentSize 挪到 chip 上（原来加在 Text 里），
        // 并从默认的过冲 spring 换成无过冲的 tween
        modifier = Modifier
            .height(buttonSize)
            .animateContentSize(
                animationSpec = tween(durationMillis = 150, easing = FastOutSlowInEasing),
            ),
        shape = CircleShape,
        border = null,
        colors = if (isSelected) {
            FilterChipDefaults.filterChipColors(
                selectedContainerColor = MaterialTheme.colorScheme.primary,
                selectedLabelColor = MaterialTheme.colorScheme.onPrimary,
            )
        } else {
            FilterChipDefaults.filterChipColors(
                containerColor = t.sage,
                labelColor = t.olive,
            )
        },
        label = {
            Text(
                text = displayText,
                style = MaterialTheme.typography.labelSmall,
                textAlign = TextAlign.Center,
                maxLines = 1,
            )
        },
    )
}

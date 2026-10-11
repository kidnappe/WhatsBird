package com.whatsbird.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider

/**
 * 全 app 统一主题入口（docs/design/DESIGN.md）：
 * 五套纸感配色 × 昼/夜两态，由设置页「外观」控制（ThemePrefs 持久化）。
 * 历史上的 Material You 壁纸动态取色已于 2026-10-08 移除——
 * 壁纸漂移会毁掉纸感，且配色语义（一色一义）要求手选固定色板。
 */
@Composable
fun WhatsBirdTheme(
    palette: PaperPalette = PaperPalettes.DEFAULT,
    dark: Boolean = false,
    /**
     * false = 不做逐色渐变，直接落到目标色。
     *
     * 冷启动必须传 false（见 MainActivity）：外观偏好走 DataStore 异步读，首帧必然先用
     * 默认色板画一帧，等偏好到了再切。开着渐变时这次切换是 250ms 全 app 交叉淡变，
     * 用非默认配色或强制夜间的用户**每次启动都看得见这一下闪**。
     * 关掉之后 animateColorAsState 只在偏好就绪那一刻才创建，初值即目标色，不产生动画。
     */
    animate: Boolean = true,
    content: @Composable () -> Unit,
) {
    val base = PaperThemeState(palette, dark)
    // MOTION §4：主题切换（色板/昼夜）经 animated tokens 全 app 交叉渐变
    val state = if (animate) base.copy(overrideT = base.t.animateForTheme()) else base
    CompositionLocalProvider(LocalPaperPalette provides state) {
        MaterialTheme(colorScheme = state.scheme(), content = content)
    }
}

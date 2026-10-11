package com.whatsbird.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * ② 滚动发丝线 + 纸底钉层（v6 P0-2，抄 SwiftUI_Demos/iOS18MailAppTabbar）
 *
 * 原式：`progress = max(min((scrollOffset + topInset - 110) / 15, 1), 0)`
 * ——滚过 110px 才开始显现，在接下来的 15px 行程里淡入完毕。
 * 前 7 屏完全没有视觉噪音，一越过阈值就干脆立起来。
 *
 * **只提供"钉在顶部的一层"**（纸底 + 1px 发丝线），标题仍在 `content` 里随内容滚走——
 * 滚过去之后顶上只剩一条线，正好是"过了阈值就立起来"的效果。
 *
 * ⚠️ 核对过：全 app **只有设置页有页级大标题**（图鉴直接进进度卡、日志直接进汇总行），
 * 所以本组件目前只用在设置页。图鉴/日志若要加标题属于**设计变更**，不是动效修复，
 * 没有替用户决定。
 *
 * L2 详情弹层是 M3 原生 ModalBottomSheet、自带标题区，**不套这个**
 * （叠了会和拖拽关闭打架）。
 *
 * @param scrollOffsetPx 由各页提供**真实**滚动偏移（px）。不要在这里猜首项高度——
 *   日志是 LazyColumn（带 sticky 日期组头）、图鉴是 LazyVerticalGrid、设置是 verticalScroll，
 *   三者拿偏移的方式完全不同，交给调用方最准。
 */
@Composable
fun PaperStickyTop(
    scrollOffsetPx: () -> Float,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    val t = com.whatsbird.ui.theme.LocalPaperPalette.current.t

    // 只在 alpha 真正变化时重组，避免滚动时整页重组
    val alpha by remember(scrollOffsetPx) {
        derivedStateOf {
            val off = scrollOffsetPx().coerceAtLeast(0f)
            ((off - RevealAfterPx) / RevealWindowPx).coerceIn(0f, 1f)
        }
    }

    Box(modifier.fillMaxWidth()) {
        Column(modifier.fillMaxWidth(), content = content)
        // 钉在顶部的一层：纸底 + 1px 发丝线，随滚动淡入
        Box(
            Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .graphicsLayer { this.alpha = alpha }
        ) {
            Box(Modifier.fillMaxWidth().background(t.paper))
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(1.dp)
                    .background(t.ink.copy(alpha = 0.09f))
            )
        }
    }
}

/** 与原 demo 一致：滚过这么多像素才开始显现 */
private const val RevealAfterPx = 110f

/** 在这么多像素的行程里淡入完毕 */
private const val RevealWindowPx = 15f
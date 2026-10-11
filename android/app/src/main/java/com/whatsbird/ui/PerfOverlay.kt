package com.whatsbird.ui

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * 读电池温度（°C）。
 *
 * 用 `Intent.ACTION_BATTERY_CHANGED` 广播里的 `EXTRA_TEMPERATURE`（单位 0.1°C）——
 * 全公开 API、无需任何权限。
 * （曾想用 `BatteryManager.BATTERY_PROPERTY_TEMPERATURE`，但它在 SDK stub 里不是公开符号。）
 * 注意这是**电池温度**，不是 SoC 温度；Android 10+ 的
 * `PowerManager.getCurrentThermalStatus()` 只给 0~6 等级、不给度数，对本监控意义不大。
 */
fun readBatteryTempC(ctx: Context): Float = try {
    val i = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val t = i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
    if (t <= -1000 || t > 1000) Float.NaN else t / 10f
} catch (t: Throwable) {
    Float.NaN
}

private fun fmtTemp(v: Float): String = if (v.isNaN()) "--" else "%.1f°C".format(v)

/**
 * 性能监控叠加层（设置页开关打开时显示）。
 *
 * 位置：**取景页顶部工具行内，夹在「区域」与「自动放大」两个芯片之间**（横排一行）。
 * 该 Row 是 SpaceBetween 三子项，因此自然落在正中；关闭时不占位，两芯片复原为两端对齐。
 * 视觉：半透明深底 + 等宽小字，三项用竖线分隔。
 */
@Composable
fun PerfOverlay(
    inferMs: Float,
    fps: Float,
    batteryTempC: Float,
    modifier: Modifier = Modifier,
) {
    val hot = batteryTempC.isNaN() || batteryTempC < 42f
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(Color.Black.copy(alpha = 0.5f))
            .padding(horizontal = 7.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        PerfItem(fmtTemp(batteryTempC), if (hot) Color(0xFF9BE59B) else Color(0xFFFFB4A2))
        Divider()
        PerfItem("%.1ffps".format(fps), Color(0xFFBFD8FF))
        Divider()
        PerfItem("%.0fms".format(inferMs), Color(0xFFFFE0A3))
    }
}

@Composable
private fun Divider() {
    Text(
        "|",
        fontSize = 9.sp,
        color = Color.White.copy(alpha = 0.3f),
        fontFamily = FontFamily.Monospace,
    )
}

@Composable
private fun PerfItem(v: String, vc: Color) {
    Text(
        v,
        fontSize = 10.sp,
        color = vc,
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Medium,
        maxLines = 1,
    )
}
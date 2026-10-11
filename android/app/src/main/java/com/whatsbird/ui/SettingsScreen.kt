/*
 * 标准设置页（2026-10-08，替换此前直接顶在 tab3 的裸主题页）。
 * 结构 = 苹果式分组列表：外观 / 识别 / 数据 / 关于，每组一张圆角卡片。
 * 原则：设置页只放"用户真的会来改"的东西——
 *   外观嵌 ThemeSettingsContent（种子色/动态取色）；
 *   识别 = 自动放大持久默认（AppPrefs）+ 区域路由状态（写 Router 单例，识别页即时生效）；
 *   数据 = 占用统计 + CSV 导出（SAF，不申请存储权限）+ 清空（双确认）；
 *   关于 = 版本 / 模型清单（类数运行时派生，不写死）/ 许可摘要
 *          / 发布页与下载（外链 1，跳自家 GitHub Pages）/ 支持拾鸟（外链 2，跳爱发电）
 *          / 数据本地声明。
 *   ⚠️ 全 App 只有这两条外链，都排在「数据位置」之前——数据位置是两行长说明，
 *      外链压在页脚既不好找、又被底部悬浮导航胶囊挡住。
 */
package com.whatsbird.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.whatsbird.vision.BirdDetector
import com.whatsbird.data.AppPrefs
import com.whatsbird.ui.theme.LocalPaperPalette
import com.whatsbird.ui.theme.PaperTheme
import com.whatsbird.data.db.ObservationDao
import com.whatsbird.vision.ModelRegistry
import com.whatsbird.vision.RegionClassifierRouter
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/**
 * 爱发电赞助页（2026-10-11 定案：docs/legal/LICENSES.md §H 的"免费 App + 纯自愿赞助"安全区）。
 * ⚠️ App 内**只放这一条裸链接**，不展示任何档位名/价格/回报——档位文案只活在爱发电页面上，
 * 否则 App 自身就成了销售界面，性质完全变了。URL 集中在此，换域名只改一处。
 *
 * ⚠️ 2026-10-12 起这里不再是"唯一外链"：另加了一条 RELEASE_PAGE_URL（自家发布页）。
 * 赞助页那条的限制仍然有效——**赞助相关的限制只约束赞助链接这一条**，
 * 新的发布页外链不受"不写档位/价格"约束，但也别把赞助内容塞进发布页入口。
 */
private const val AFDIAN_URL = "https://afdian.com/a/zhouxuano6"

/**
 * 发布页（GitHub Pages）：版本号 / 体积 / SHA-256、下载入口、安装步骤、许可与资产授权都收在这里。
 * 与 AFDIAN_URL 分开定义——一个指向自家文档，一个指向赞助页，两者的"改动理由"完全不同，
 * 以后其中一条换地址不该连累另一条。URL 集中在此，换域名只改一处。
 */
private const val RELEASE_PAGE_URL = "https://kidnappe.github.io/WhatsBird/"

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val prefs: AppPrefs,
    private val dao: ObservationDao,
    private val router: RegionClassifierRouter,
    private val detector: BirdDetector,
    @ApplicationContext private val appContext: Context,
) : ViewModel() {

    val autoZoomDefault: StateFlow<Boolean> = prefs.autoZoomDefault
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    fun setAutoZoomDefault(v: Boolean) {
        viewModelScope.launch { prefs.setAutoZoomDefault(v) }
    }

    /** 性能监控叠加层开关（取景页顶部温度/帧率/推理延迟）。 */
    val perfMonitor: StateFlow<Boolean> = prefs.perfMonitor
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    fun setPerfMonitor(v: Boolean) {
        viewModelScope.launch { prefs.setPerfMonitor(v) }
    }

    /** 手动区域：Router 是进程单例，这里写 = 识别页下一帧生效。 */
    private val _manualRegion = MutableStateFlow(router.manualRegion())
    val manualRegion: StateFlow<String?> = _manualRegion.asStateFlow()

    fun revertRegion() {
        router.setManualRegion(null)
        _manualRegion.value = null
    }

    fun routingReason(): String = router.routingReason()

    // ===== 检测器档位（2026-10-10）=====
    // 设置页改的是同一个进程级 YoloDetector 单例，**下一帧 detect 即生效**（不需重启 App）。
    // 档位一旦生效就常驻到被再次切走——"一次只驻留一档"，切走时旧模型 close。
    //
    // ⚠️ 2026-10-10 修两处：
    // ① 初值原先写死 TIER_AUTO，从不读回实际档位 → 用户选了"旗舰"后重开设置页
    //    显示"自动"、与实际不符。现在初值取 detector.tier()，并持续跟随 prefs。
    private val _detTier = MutableStateFlow(detector.tier())
    val detTier: StateFlow<String> = _detTier.asStateFlow()

    init {
        viewModelScope.launch {
            prefs.detTier.collect { stored ->
                // prefs 是白名单校验过的合法值；只在与检测器当前档不同时才推
                if (stored != detector.tier()) {
                    viewModelScope.launch(Dispatchers.Default) {
                        detector.switchTier(stored)
                        _detTier.value = detector.tier()
                        _detAsset.value = detector.activeAsset()
                    }
                }
            }
        }
    }

    private val _detAsset = MutableStateFlow(detector.activeAsset())
    val detAsset: StateFlow<String> = _detAsset.asStateFlow()

    /** 设备 totalMem（字节）。⚠️ 不是物理内存，只是给用户看 auto 判定的依据。 */
    fun detTotalMem(): Long = detector.totalMemBytes()

    /**
     * auto 模式下这台机器会被判成哪一档，**只回分辨率数字**（640 / 512）。
     * ⚠️ 2026-10-11：原先直接回资产文件名（`yolox_bird_640_best_int8`），设置页那行读不完。
     * 数字从文件名里抽，将来资产改名也不会写歪。
     */
    fun detAutoPicks(): String =
        Regex("\\d{3}")
            .find(ModelRegistry.detectorAsset(detector.totalMemBytes()).substringAfterLast('/'))
            ?.value ?: "?"

    fun setDetTier(v: String) {
        // ⚠️ 2026-10-10：原先在主线程直接 switchTier。切档会 close 旧 Interpreter
        // 并在下一帧 load 新模型，属于 IO/重活，放后台线程，别卡住点击。
        viewModelScope.launch(Dispatchers.Default) {
            detector.switchTier(v)
            _detTier.value = detector.tier()
            _detAsset.value = detector.activeAsset()
            viewModelScope.launch { prefs.setDetTier(_detTier.value) }
        }
    }

    data class Stats(val records: Int, val bytes: Long)

    private val _stats = MutableStateFlow<Stats?>(null)
    val stats: StateFlow<Stats?> = _stats.asStateFlow()

    init {
        refresh()
    }

    private fun dirBytes(f: File): Long =
        if (!f.exists()) 0L else if (f.isFile) f.length()
        else f.listFiles()?.sumOf { dirBytes(it) } ?: 0L

    private fun refresh() {
        viewModelScope.launch(Dispatchers.IO) {
            val n = dao.count()
            val photos = dirBytes(File(appContext.filesDir, "records"))
            val dbs = dirBytes(File(appContext.filesDir.parentFile, "databases"))
            _stats.value = Stats(n, photos + dbs)
        }
    }

    // 一次性提示（导出/清空结果），页面消费后延时清除
    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message.asStateFlow()
    fun consumeMessage() {
        _message.value = null
    }

    /** UI 侧一次性提示（外链跳转失败等），复用同一条 3.5s 自动消失的机制。 */
    fun notify(text: String) {
        _message.value = text
    }

    /** 全量导出 CSV（UTF-8 带 BOM，Excel 直开不乱码）。经 SAF uri 写，无需存储权限。 */
    fun exportCsv(target: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val rows = dao.observeAll().first()
                val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                fun esc(s: String) = "\"" + s.replace("\"", "\"\"") + "\""
                val sb = StringBuilder("id,中文名,学名,观测时间,置信度,人工确认,区域,纬度,经度,地址,备注,照片文件\n")
                rows.forEach { o ->
                    sb.append(
                        listOf(
                            o.id.toString(), esc(o.speciesZh), esc(o.speciesSci),
                            fmt.format(Date(o.observedAt)), "%.2f".format(o.confidence),
                            if (o.confirmed) "是" else "否", o.region ?: "",
                            o.lat?.toString() ?: "", o.lon?.toString() ?: "",
                            esc(o.address ?: ""),
                            esc(o.note), o.photoPath?.let { File(it).name } ?: "",
                        ).joinToString(","),
                    ).append('\n')
                }
                appContext.contentResolver.openOutputStream(target)?.use { os ->
                    os.write(byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()))
                    os.write(sb.toString().toByteArray(Charsets.UTF_8))
                    _message.value = "已导出 ${rows.size} 条记录"
                } ?: run { _message.value = "导出失败：无法打开目标文件" }
            } catch (t: Throwable) {
                android.util.Log.w("Settings", "CSV 导出失败", t)
                _message.value = "导出失败：${t.message}"
            }
            refresh()
        }
    }

    fun clearAll() {
        viewModelScope.launch(Dispatchers.IO) {
            dao.deleteAll()
            File(appContext.filesDir, "records").listFiles()?.forEach { it.delete() }
            _message.value = "已清空全部记录与照片"
            refresh()
        }
    }
}

private fun fmtBytes(b: Long): String = when {
    b >= 1L shl 30 -> "%.2f GB".format(b / 1073741824.0)
    b >= 1L shl 20 -> "%.1f MB".format(b / 1048576.0)
    b >= 1024 -> "%d KB".format(b / 1024)
    else -> "$b B"
}

@Composable
fun SettingsScreen(
    modifier: Modifier = Modifier,
    viewModel: SettingsViewModel = hiltViewModel(),
) {
    val context = LocalContext.current
    val autoZoom by viewModel.autoZoomDefault.collectAsState()
    val perfMonitor by viewModel.perfMonitor.collectAsState()
    val manualRegion by viewModel.manualRegion.collectAsState()
    val detTier by viewModel.detTier.collectAsState()
    val stats by viewModel.stats.collectAsState()
    val message by viewModel.message.collectAsState()
    LaunchedEffect(message) {
        if (message != null) {
            delay(3500)
            viewModel.consumeMessage()
        }
    }

    val exporter = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv"),
    ) { uri -> uri?.let(viewModel::exportCsv) }

    var showClearDialog by rememberSaveable { mutableStateOf(false) }

    PaperTheme {
    val t = LocalPaperPalette.current.t
    // ②（v6 P0-2）滚动发丝线：滚过 110px 起步、15px 行程淡入。
    // 全 app 只有本页有页级大标题（图表鉴直接进进度卡、日志直接进汇总行），
    // 所以这个钉层只装在这里；标题本身仍在内容里、照常滚走。
    val scrollState = rememberScrollState()
    PaperStickyTop(
        scrollOffsetPx = { scrollState.value.toFloat() },
        modifier = modifier
            .fillMaxSize()
            .background(t.paper)
            // 状态栏避让（与日志/图鉴同修）
            .windowInsetsPadding(WindowInsets.statusBars),
    ) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(horizontal = 16.dp),
    ) {
        Spacer(Modifier.height(20.dp))
        Text(
            "设置",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )

        // ===== 外观（嵌既有主题内容，无嵌套卡片） =====
        SectionHeader("外观")
        SettingsCard {
            ThemeSettingsContent()
        }

        // ===== 识别 =====
        SectionHeader("识别")
        SettingsCard {
            SwitchRow(
                label = "自动放大",
                desc = "进入取景页后自动步进数码变焦找鸟；页内芯片可随时临时开关",
                checked = autoZoom,
                onChange = viewModel::setAutoZoomDefault,
            )
            RowDivider()
            SwitchRowLike(
                label = "识别区域",
                desc = manualRegion?.let { "手动指定：${ModelRegistry.REGION_NAMES[it] ?: it}（重启回自动）" }
                    ?: "自动（GPS 定位路由） · ${viewModel.routingReason()}",
                trailing = {
                    if (manualRegion != null) {
                        TextButton(onClick = viewModel::revertRegion) { Text("恢复自动") }
                    } else {
                        Text(
                            "自动",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
            )
            RowDivider()
            SwitchRow(
                label = "性能监控",
                desc = "取景页顶部显示电池温度、帧率与推理延迟（排查发热/掉帧用）",
                checked = perfMonitor,
                onChange = viewModel::setPerfMonitor,
            )
            RowDivider()
            Column {
                ValueRow(
                    label = "检测器档位",
                    // 只报分辨率（640 / 512），不报内部资产名——2026-10-11 用户要求。
                    value = when (detTier) {
                        ModelRegistry.TIER_FLAG -> "640"
                        ModelRegistry.TIER_MID -> "512"
                        else -> "自动 → ${viewModel.detAutoPicks()}"
                    },
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(start = 12.dp, end = 12.dp, bottom = 12.dp),
                ) {
                    listOf(
                        ModelRegistry.TIER_AUTO to "自动",
                        ModelRegistry.TIER_FLAG to "旗舰",
                        ModelRegistry.TIER_MID to "中档",
                    ).forEach { (k, label) ->
                        FilterChip(
                            selected = detTier == k,
                            onClick = { viewModel.setDetTier(k) },
                            label = { Text(label) },
                        )
                    }
                }
            }
        }

        // ===== 数据 =====
        SectionHeader("数据")
        SettingsCard {
            ValueRow("观测记录", stats?.let { "${it.records} 条" } ?: "…")
            RowDivider()
            ValueRow("存储占用", stats?.let { fmtBytes(it.bytes) } ?: "…")
            RowDivider()
            ActionRow(
                label = "导出 CSV",
                desc = "全量记录（含定位/确认标记/照片文件名），自选保存位置",
                onClick = {
                    exporter.launch(
                        "whatsbird_records_" +
                            SimpleDateFormat("yyyyMMdd_HHmm", Locale.getDefault())
                                .format(Date()) + ".csv",
                    )
                },
            )
            RowDivider()
            ActionRow(
                label = "清空全部记录",
                desc = "删除所有观测与照片，不可恢复",
                destructive = true,
                onClick = { showClearDialog = true },
            )
        }
        message?.let {
            Text(
                it,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(start = 6.dp, top = 8.dp),
            )
        }

        // ===== 关于 =====
        SectionHeader("关于")
        SettingsCard {
            val version = remember {
                runCatching {
                    context.packageManager.getPackageInfo(context.packageName, 0).versionName
                }.getOrNull() ?: "?"
            }
            ValueRow("版本", "拾鸟 $version")
            RowDivider()
            ExpandableRow("模型清单") {
                listOf(
                    "cn373", "huabei", "huadong", "huazhong", "huanan", "xinan", "dongbei",
                ).forEach { key ->
                    Text(
                        ModelRegistry.regionDisplayName(context, key),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 2.dp),
                    )
                }
                Text(
                    "检测器 YOLOX 640/512 双档 int8（按内存自动选择）；识别结果另经 geomodel 地理先验重排。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
            RowDivider()
            ExpandableRow("许可信息") {
                Text(
                    "LiteRT（TensorFlow Lite）、AndroidX/Compose、YOLOX：Apache-2.0\n" +
                        "geomodel 地理先验：代码 MIT、模型 Apache-2.0\n" +
                        "物种分布数据源（GBIF 等）按逐条许可过滤使用。\n" +
                        "完整清单见项目仓库 docs/legal/LICENSES.md。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            RowDivider()
            // 外链共两条，都排在「数据位置」那行**之前**——
            // 数据位置是两行长说明，外链压在页脚既不好找、又会被底部悬浮导航胶囊挡住。
            ActionRow(
                label = "发布页与下载",
                desc = "最新版、体积与 SHA-256 校验值、安装步骤、许可与资产授权都在这里（将打开浏览器）",
                onClick = {
                    // 国产 ROM 精简浏览器时真会没有可处理的 Activity，别静默失败。
                    val ok = runCatching {
                        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(RELEASE_PAGE_URL)))
                    }.isSuccess
                    if (!ok) viewModel.notify("没能打开浏览器，可手动访问 $RELEASE_PAGE_URL")
                },
            )
            RowDivider()
            // 只写"跳转爱发电"，不写档位与价格——见 AFDIAN_URL 上方注释。
            ActionRow(
                label = "支持拾鸟",
                desc = "拾鸟免费、无广告、数据只在你手机里。如果它帮到了你，可以支持拾鸟（将跳转爱发电）",
                onClick = {
                    // 国产 ROM 精简浏览器时真会没有可处理的 Activity，别静默失败。
                    val ok = runCatching {
                        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(AFDIAN_URL)))
                    }.isSuccess
                    if (!ok) viewModel.notify("没能打开浏览器，可手动访问 afdian.com/a/zhouxuano6")
                },
            )
            RowDivider()
            ValueRow("数据位置", "记录与照片仅存本机，无云端上传（地址由系统定位服务离线解析，无后端则留空）")
        }
        // 底部余量 96dp：与日志/图鉴的 contentPadding bottom 同口径，
        // 否则最后一行会缩进底部那枚悬浮导航胶囊底下（胶囊高 56dp + 10dp 边距）。
        Spacer(Modifier.height(96.dp))
    }
    }   // ← PaperStickyTop content 结束

    if (showClearDialog) {
        AlertDialog(
            onDismissRequest = { showClearDialog = false },
            title = { Text("清空全部记录？") },
            text = {
                Text(
                    "将删除 ${stats?.records ?: 0} 条观测记录及全部框选照片，" +
                        "图鉴点亮状态也会随之清空。此操作不可恢复，建议先「导出 CSV」。",
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showClearDialog = false
                        viewModel.clearAll()
                    },
                ) {
                    Text("清空", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearDialog = false }) { Text("取消") }
            },
        )
    }
    }
}

// ==================== 分组与行元件（本页私有，风格对齐 inset-grouped） ====================

@Composable
private fun SectionHeader(title: String) {
    Text(
        title,
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(start = 6.dp, top = 24.dp, bottom = 8.dp),
    )
}

@Composable
private fun SettingsCard(content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHigh),
        content = content,
    )
}

@Composable
private fun RowDivider() {
    HorizontalDivider(
        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f),
        modifier = Modifier.padding(start = 16.dp),
    )
}

@Composable
private fun SwitchRow(label: String, desc: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    SwitchRowLike(
        label = label,
        desc = desc,
        trailing = { Switch(checked = checked, onCheckedChange = onChange) },
    )
}

@Composable
private fun SwitchRowLike(label: String, desc: String, trailing: @Composable () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(label, style = MaterialTheme.typography.bodyLarge)
            Text(
                desc,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        trailing()
    }
}

@Composable
private fun ValueRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 2,
        )
    }
}

@Composable
private fun ActionRow(label: String, desc: String, destructive: Boolean = false, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(
                label,
                style = MaterialTheme.typography.bodyLarge,
                color = if (destructive) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.onSurface,
            )
            Text(
                desc,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text("›", style = MaterialTheme.typography.titleLarge, color = MaterialTheme.colorScheme.outline)
    }
}

@Composable
private fun ExpandableRow(title: String, body: @Composable () -> Unit) {
    var open by rememberSaveable { mutableStateOf(false) }
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { open = !open }
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(title, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
            Text(if (open) "▾" else "▸", color = MaterialTheme.colorScheme.outline)
        }
        if (open) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 16.dp, end = 16.dp, bottom = 14.dp),
            ) {
                body()
            }
        }
    }
}

package com.whatsbird.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.ExperimentalFoundationApi
import android.view.HapticFeedbackConstants
import androidx.compose.ui.platform.LocalView
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.rememberModalBottomSheetState

import androidx.compose.animation.core.Spring
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Composable
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.whatsbird.R
import com.whatsbird.data.db.ObservationDao
import com.whatsbird.data.db.ObservationEntity
import com.whatsbird.ui.theme.LocalPaperPalette
import com.whatsbird.ui.theme.PaperSciName
import com.whatsbird.ui.theme.PaperSearchField
import com.whatsbird.ui.theme.PaperTheme
import com.whatsbird.ui.theme.PaperTick
import com.whatsbird.vision.ModelRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import java.text.SimpleDateFormat
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class RecordsUiState(
    val items: List<ObservationEntity> = emptyList(),
    val query: String = "",
)

/**
 * 构造 FTS 查询串（2026-10-10）。
 *
 * 原先是直接 `"*$q*"`：FTS4/5 的 MATCH **不支持前导 `*`**（只有后缀可用），
 * 且用户输入的 `"` `-` `OR` 等会被当成查询语法 → 抛 SQLiteException；
 * 而 flatMapLatest 里没有 catch，异常一路冒到 stateIn，**输错一个字符整个 ViewModel 就崩**。
 *
 * 现在：按空白切词，每词用双引号包住（内部引号按 FTS 规则翻倍）再加后缀 `*` 做前缀匹配，
 * 其余字符一律当普通文本。多个词之间是 AND。
 */
private fun ftsQuery(raw: String): String =
    raw.trim().split(Regex("\\s+"))
        .filter { it.isNotEmpty() }
        .joinToString(" ") { "\"${it.replace("\"", "\"\"")}\"*" }

@OptIn(ExperimentalCoroutinesApi::class)
@HiltViewModel
class RecordsViewModel @Inject constructor(
    private val dao: ObservationDao,
) : ViewModel() {
    private val query = MutableStateFlow("")

    // 有搜索词走 FTS（observations_fts），没有就全量倒序
    val state: StateFlow<RecordsUiState> = query
        .flatMapLatest { q ->
            val base = if (q.isBlank()) {
                dao.observeAll()
            } else {
                dao.search(ftsQuery(q))
                    .catch { e ->
                        // 兜底：一条搜索词不该让整个日志页崩掉，降级成"无结果"
                        android.util.Log.w("RecordsVM", "搜索失败，已降级为空结果", e)
                        emit(emptyList())
                    }
            }
            base.combine(query) { items, query -> RecordsUiState(items, query) }
        }
        .catch { e ->
            android.util.Log.w("RecordsVM", "记录流异常", e)
            emit(RecordsUiState())
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), RecordsUiState())

    fun setQuery(q: String) {
        query.value = q
    }

    /** 详情弹层的编辑统一走整行回写（改判/备注共用）。 */
    fun save(o: ObservationEntity) {
        viewModelScope.launch { dao.update(o) }
    }

    fun delete(id: Long) {
        viewModelScope.launch { dao.delete(id) }
    }

    /** 右滑删除的撤销：整行原样回插。⚠️ 2026-10-10 更正注释：o.id **保持不变**
     *  （原注释写"新 id"是错的），依赖主键原样回插；UI 侧靠 remember(o.id) 复位滑动偏移。 */
    fun restore(o: ObservationEntity) {
        viewModelScope.launch { dao.insert(o) }
    }
}

@Composable
private fun MiniStat(label: String, value: String, modifier: Modifier = Modifier) {
    val t = LocalPaperPalette.current.t
    Column(
        modifier = modifier
            .shadow(4.dp, RoundedCornerShape(14.dp), ambientColor = t.ink.copy(alpha = 0.10f), spotColor = t.ink.copy(alpha = 0.12f))
            .clip(RoundedCornerShape(14.dp))
            .background(t.card)
            .padding(12.dp),
    ) {
        Text(
            value,
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.ExtraBold,
            color = t.moss,
        )
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = t.gray,
        )
    }
}

/** 滑动删除区宽度：行吸附展开时停在 -SwipeActionWidth，正好露出这一整块。 */
private val SwipeActionWidth = 92.dp

/** 缩略图：2026-10-10 起解码在 IO 线程，采样率按目标显示边长自动算（原为写死常量）。 */
@Composable
private fun loadThumb(path: String?, maxEdge: androidx.compose.ui.unit.Dp, minSample: Int = 1) =
    rememberDecoded(path, maxEdge, minSample)

/** 按日分组的日志头（琥珀刻度 + 日期粗体 + 条数）。 */
@Composable
private fun DayHeader(date: LocalDate, count: Int) {
    val t = LocalPaperPalette.current.t
    val title = remember { SimpleDateFormat("M月d日 EEEE", Locale.CHINA) }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(t.paper) // sticky 组头需不透明，防内容穿透
            .padding(start = 4.dp, end = 4.dp, top = 16.dp, bottom = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        PaperTick()
        Text(
            title.format(Date.from(date.atStartOfDay(ZoneId.systemDefault()).toInstant())),
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
            color = t.ink,
        )
        Text(
            "  ·  $count 条",
            style = MaterialTheme.typography.labelMedium,
            color = t.gray,
        )
    }
}

/**
 * 观测日志页（与图鉴分工，docs/design/DESIGN.md §4.4；
 * 纸感换装 docs/design/DESIGN.md）：
 * 「日志 = 流水」——按日分组的时间轴，一条 = 一次观测，可点开详情改判/加备注/删除。
 * 物种维度的聚合（点亮/次数/图鉴格）全部让位给图鉴页，这里不放鸟种数。
 */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun RecordsScreen(
    modifier: Modifier = Modifier,
    viewModel: RecordsViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val fmt = remember { SimpleDateFormat("HH:mm", Locale.getDefault()) }
    var detailId by rememberSaveable { mutableStateOf<Long?>(null) }
    // ⚠️ 2026-10-10 改 rememberSaveable：旋转/进程重建后原先大图请求直接丢失（照片"消失"）。
    var viewReq by rememberSaveable(stateSaver = ViewerReqSaver) { mutableStateOf<ViewerReq?>(null) }

    val groups = remember(state.items) {
        val zone = ZoneId.systemDefault()
        state.items
            .groupBy { Instant.ofEpochMilli(it.observedAt).atZone(zone).toLocalDate() }
            .toSortedMap(java.util.Comparator.reverseOrder())
    }

    val snackbar = remember { SnackbarHostState() }
    // ⑧（v6 P2-2）同一时刻只允许一行处于左滑打开态；开新行时旧的自动归位
    var openRowId by remember { mutableStateOf<Long?>(null) }
    val uiScope = rememberCoroutineScope()
    PaperTheme {
        val t = LocalPaperPalette.current.t
        Box(modifier = modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(t.paper)
                // 状态栏避让（取景页自绘全屏不吃这个；列表页顶卡此前被时间遮挡）
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(12.dp),
        ) {
            // 汇总卡：日志口径（总量/近7日/人工确认）；鸟种数归图鉴，不放这里
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                MiniStat("记录", "${state.items.size}", Modifier.weight(1f))
                MiniStat(
                    "近 7 日",
                    "+${state.items.count {
                        System.currentTimeMillis() - it.observedAt < 7L * 86400_000
                    }}",
                    Modifier.weight(1f),
                )
                MiniStat("人工确认", "${state.items.count { it.confirmed }}", Modifier.weight(1f))
            }
            PaperSearchField(
                value = state.query,
                onValueChange = viewModel::setQuery,
                placeholder = "搜索物种 / 笔记（FTS 全文）",
                modifier = Modifier.padding(top = 10.dp),
            )
            if (state.items.isEmpty()) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Image(
                        painter = painterResource(R.drawable.ic_bird),
                        contentDescription = null,
                        modifier = Modifier.size(52.dp),
                        colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.silInk),
                    )
                    Spacer(Modifier.height(10.dp))
                    Text(
                        if (state.query.isBlank()) "还没有观测日志\n在「识别」页拍鸟并点「记录此鸟」"
                        else "没有匹配「${state.query}」的记录",
                        style = MaterialTheme.typography.bodyMedium,
                        color = t.gray,
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize().padding(top = 4.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(
                        bottom = 96.dp,
                    ),
                ) {
                    groups.forEach { (day, recs) ->
                        stickyHeader(key = "day_$day") { DayHeader(day, recs.size) }
                        items(recs, key = { it.id }) { o ->
                            val t2 = LocalPaperPalette.current.t
                            // MOTION §4：左滑过 32% 删除，Snackbar 撤销
                            // ⚠️ 2026-10-10：key 必须带 o.id。原来是 remember { Animatable(0f) }，
                            // 撤销删除后同一 id 重新进组合时复用了删除动画留下的偏移（=-w），
                            // 整行停在列表外看不见——用户表现为"删了删不掉"。
                            // 带 key 后 id 一变即重建，偏移归零。
                            val off = remember(o.id) { Animatable(0f) }
                            val rowScope = rememberCoroutineScope()
                            val density = LocalDensity.current
                            // 删除区宽度（px）：行吸附展开时停在 -ACTION_WPx，露出这一整块
                            val ACTION_WPx = with(density) { SwipeActionWidth.toPx() }
                            // 行宽：pointerInput 里能拿到，但删除按钮的 onClick 在它外面，
                            // 所以另存一份（行是 fillMaxWidth，宽度稳定）。
                            var rowW by remember(o.id) { mutableFloatStateOf(0f) }
                            // ⚠️ 本行正在删除：阻止 LaunchedEffect(openRowId) 把它抢回去。
                            // 根因（2026-10-11 由 logcat 定位）：doDelete 先把 openRowId 置 null，
                            // 重组触发 LaunchedEffect(openRowId) 重跑，条件成立就 off.animateTo(0f)；
                            // Animatable 的 MutatorMutex 会**取消**我们正在跑的 animateTo(-rowW)，
                            // 协程被取消 → 后面的 viewModel.delete() 从未执行。
                            // 表现就是「划到底没反应」+「点删除按钮没反应」，两个症状一个根因。
                            var deleting by remember(o.id) { mutableStateOf(false) }
                            val view = LocalView.current
                            fun haptic(kind: Int) = view.performHapticFeedback(kind)

                            /** 统一的删除动作：整行飞出 → 落库删除 → Snackbar 可撤销。 */
                            fun doDelete() {
                                if (openRowId != o.id || deleting) return
                                deleting = true          // ★ 先亮"正在删"标志
                                openRowId = null
                                haptic(android.view.HapticFeedbackConstants.CONFIRM)
                                rowScope.launch {
                                    off.animateTo(
                                        -rowW,
                                        tween(300, easing = androidx.compose.animation.core.FastOutSlowInEasing),
                                    )
                                    val keep = o
                                    viewModel.delete(keep.id)
                                    uiScope.launch {
                                        if (snackbar.showSnackbar(
                                                "已删除「${keep.speciesZh}」", "撤销",
                                            ) == SnackbarResult.ActionPerformed
                                        ) viewModel.restore(keep)
                                    }
                                }
                            }

                            /** 收起本行：弹簧归位，弹完再撤删除区（否则红色中途消失）。 */
                            fun closeSelf() {
                                if (deleting) return
                                rowScope.launch {
                                    off.animateTo(
                                        0f,
                                        spring(dampingRatio = Spring.DampingRatioMediumBouncy),
                                    )
                                    if (openRowId == o.id) openRowId = null
                                }
                            }

                            // ⑧ 别的行被打开时，本行若处于偏移态则自动归位
                            // ⚠️ deleting 的行不参与：否则它会和 doDelete 抢同一个 Animatable
                            LaunchedEffect(openRowId) {
                                if (!deleting && openRowId != o.id && off.value != 0f) {
                                    off.animateTo(0f, spring(dampingRatio = Spring.DampingRatioMediumBouncy))
                                }
                            }
                            // ⚠️ 2026-10-10 按用户要求补上删除区。
                            // 原先整行直接跟手左移，行下面什么都没有：
                            // 有橡皮筋、也能删，但**看不见删的是什么**——真机反馈
                            // 「只有皮筋效果没有删除按钮」。
                            // 现在把行包进 Box，底层铺一层红色删除区，行本体在上层滑动，
                            // 红色随着左移一点点露出来。
                            //
                            // ⚠️ 2026-10-10 快速滚动掉帧修正：删除区**只在正在被拖的那一行组合**。
                            // 之前每行都常驻着 Box+Row+Image+Text 共 4 个额外节点，
                            // 而快速滚动时逐帧有行入场，这层开销被乘以「可见行数」放大。
                            // 图鉴页的格子没有这层，所以同样内容下滑更顺。
                            // 收窄到「一次只有一行处于左滑态」（openRowId 本来就有这个语义）。
                            Box(Modifier.fillMaxWidth()) {
                                if (openRowId == o.id) {
                                    // —— 底层：删除区（只露出右侧一部分）——
                                    Row(
                                        Modifier
                                            .matchParentSize()
                                            .clip(RoundedCornerShape(14.dp))
                                            .background(t2.red),
                                        horizontalArrangement = Arrangement.End,
                                        verticalAlignment = Alignment.CenterVertically,
                                    ) {
                                        Row(
                                            verticalAlignment = Alignment.CenterVertically,
                                            modifier = Modifier
                                                .padding(end = 20.dp)
                                                .graphicsLayer {
                                                    // 在 layer 里读 off.value = 延迟读取，
                                                    // 只重画不重组；放组合里会每帧重组整块。
                                                    val reveal =
                                                        (-off.value / ACTION_WPx).coerceIn(0f, 1f)
                                                    alpha = reveal
                                                    scaleX = 0.7f + 0.3f * reveal
                                                    scaleY = 0.7f + 0.3f * reveal
                                                }
                                                // 主流做法：行吸附展开后，点删除按钮直接删。
                                                // 在 onClick 里读 off.value 判断是否真展开，
                                                // 避免为了取这个值而在组合里订阅动画（每帧重组）。
                                                .clickable(
                                                    interactionSource = remember {
                                                        androidx.compose.foundation.interaction.MutableInteractionSource()
                                                    },
                                                    indication = null,
                                                ) {
                                                    if (-off.value > ACTION_WPx * 0.6f) {
                                                                                                                 doDelete()
                                                    } else {
                                                                                                             }
                                                },
                                        ) {
                                            Image(
                                                painter = painterResource(R.drawable.ic_trash),
                                                contentDescription = "删除",
                                                modifier = Modifier.size(18.dp),
                                                colorFilter = androidx.compose.ui.graphics.ColorFilter
                                                    .tint(androidx.compose.ui.graphics.Color.White),
                                            )
                                            Spacer(Modifier.width(6.dp))
                                            Text(
                                                "删除",
                                                fontSize = 13.sp,
                                                fontWeight = FontWeight.SemiBold,
                                                color = androidx.compose.ui.graphics.Color.White,
                                            )
                                        }
                                    }
                                }
                                Row(
                                    modifier = Modifier
                                    .fillMaxWidth()
                                    .graphicsLayer {
                                        translationX = off.value
                                        alpha = 1f + off.value / 500f
                                    }
                                    .shadow(4.dp, RoundedCornerShape(14.dp), ambientColor = t2.ink.copy(alpha = 0.10f), spotColor = t2.ink.copy(alpha = 0.12f))
                                    .clip(RoundedCornerShape(14.dp))
                                    .background(t2.card)
                                    .onSizeChanged { rowW = it.width.toFloat() }
                                    .pointerInput(o.id) {
                                        val w = size.width.toFloat()
                                        var vx = 0f
                                        var lastT = 0L
                                        // 已经越过"松手即删"线的档位（只震一次）
                                        var crossedDeleteZone = false
                                        val deleteZonePx = w * 0.5f
                                        detectHorizontalDragGestures(
                                            onDragStart = {
                                                crossedDeleteZone = false
                                                // 「抓住了」
                                                haptic(android.view.HapticFeedbackConstants.CLOCK_TICK)
                                                openRowId = o.id      // 开新的 → 其余行自动归位
                                                vx = 0f; lastT = 0L
                                            },
                                            onDragEnd = {
                                                                                                 if (openRowId != o.id) return@detectHorizontalDragGestures
                                                // ⑧ 速度参与判定：endTarget = vx + offset
                                                // 快速轻扫也算数。px/s 直接用，除以 5 投影。
                                                val endTarget = off.value + vx
                                                when {
                                                    // ① 大幅滑动（过一半行宽）→ 直接删，不吸附
                                                    -endTarget > w * 0.5f -> doDelete()
                                                    // ② 中等幅度（过半删除区）→ **吸附展开**，
                                                    //    露出可点的删除按钮，等用户点（主流做法）
                                                    -off.value > ACTION_WPx * 0.5f -> {
                                                        // 「停住了，可以点了」——比视觉更早给反馈
                                                        haptic(android.view.HapticFeedbackConstants.CLOCK_TICK)
                                                        rowScope.launch {
                                                            off.animateTo(
                                                                -ACTION_WPx,
                                                                spring(dampingRatio = Spring.DampingRatioMediumBouncy),
                                                            )
                                                        }
                                                    }
                                                    // ③ 幅度不够 → 归位
                                                    else -> closeSelf()
                                                }
                                            },
                                            onDragCancel = {
                                                if (!deleting) haptic(android.view.HapticFeedbackConstants.CLOCK_TICK)
                                                closeSelf()
                                            },
                                        ) { change, dx ->
                                            change.consume()
                                            val now = android.os.SystemClock.uptimeMillis()
                                            if (lastT != 0L) {
                                                val dt = (now - lastT).coerceAtLeast(1L)
                                                vx = vx * 0.5f + (dx / dt * 1000f) * 0.5f   // px/s
                                            }
                                            lastT = now
                                            val raw = off.value + dx
                                            val clamped = raw.coerceIn(-w, 0f)
                                            // ⑧ 橡皮筋：过冲量 ÷10——拖过头会变慢，但永远拖得动
                                            val rubber = ((raw - clamped) / 10f).coerceAtMost(0f)
                                            // 越过"松手即删"的线时给一次**重**震动——
                                            // 这是 iOS/Material 滑动删除的标准手法：
                                            // 用户在越过临界点的那一刻就知道"再划就删了"，
                                            // 纯视觉的红底渐变太弱，看不出来。
                                            if (!crossedDeleteZone && -clamped > deleteZonePx) {
                                                crossedDeleteZone = true
                                                haptic(android.view.HapticFeedbackConstants.LONG_PRESS)
                                            } else if (crossedDeleteZone && -clamped <= deleteZonePx) {
                                                crossedDeleteZone = false
                                            }
                                            rowScope.launch { off.snapTo(clamped + rubber) }
                                        }
                                    }
                                    .clickable {
                                                // 主流做法：本行已展开 → 点行身收起，而不是打开详情；
                                                // 若别的行展开着，点这一行会顺手把它收起。
                                                if (openRowId == o.id && off.value < -1f) {
                                                    closeSelf()
                                                } else {
                                                    if (openRowId != null) openRowId = null
                                                    detailId = o.id
                                                }
                                            }
                                    .padding(horizontal = 12.dp, vertical = 9.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                val bmp by loadThumb(o.photoPath, 100.dp)
                                val img = bmp
                                if (img != null) {
                                    Image(
                                        img.asImageBitmap(),
                                        contentDescription = o.speciesZh,
                                        contentScale = ContentScale.Crop,
                                        modifier = Modifier
                                            .size(50.dp)
                                            .clip(RoundedCornerShape(10.dp))
                                            .background(t2.sage)
                                            .clickable {
                                                viewReq = ViewerReq(listOfNotNull(o.photoPath), 0, o.speciesZh, o.speciesSci)
                                            },
                                    )
                                } else {
                                    Box(
                                        Modifier
                                            .size(50.dp)
                                            .clip(RoundedCornerShape(10.dp))
                                            .background(t2.silA),
                                        contentAlignment = Alignment.Center,
                                    ) {
                                        Image(
                                            painter = painterResource(R.drawable.ic_bird),
                                            contentDescription = null,
                                            modifier = Modifier.size(26.dp),
                                            colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t2.silInk),
                                        )
                                    }
                                }
                                Column(
                                    Modifier
                                        .padding(start = 11.dp)
                                        .weight(1f),
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Text(
                                            o.speciesZh,
                                            style = MaterialTheme.typography.titleMedium,
                                            fontSize = 15.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = t2.ink,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis,
                                            // 徽章无 weight 先量，名字吃剩余空间，过长时省略的是名字
                                            modifier = Modifier.weight(1f, fill = false),
                                        )
                                        // 保护等级在前、危险等级在后（2026-10-11 用户定序）
                                        ProtectionPills(o.speciesSci, Modifier.padding(start = 6.dp))
                                    }
                                    PaperSciName(o.speciesSci, Modifier.padding(top = 1.dp))
                                }
                                if (o.confirmed) {
                                    Image(
                                        painter = painterResource(R.drawable.ic_check),
                                        contentDescription = "已人工确认",
                                        modifier = Modifier.size(13.dp),
                                        colorFilter = androidx.compose.ui.graphics.ColorFilter
                                            .tint(t2.moss),
                                    )
                                    Spacer(Modifier.width(4.dp))
                                }
                                if (o.note.isNotBlank()) {
                                    Image(
                                        painter = painterResource(R.drawable.ic_note),
                                        contentDescription = "有备注",
                                        modifier = Modifier.size(13.dp),
                                        colorFilter = androidx.compose.ui.graphics.ColorFilter
                                            .tint(t2.gray),
                                    )
                                    Spacer(Modifier.width(4.dp))
                                }
                                Text(
                                    fmt.format(Date(o.observedAt)),
                                    style = MaterialTheme.typography.labelLarge,
                                    fontSize = 12.sp,
                                    color = t2.amber,
                                )
                            }
                        }
                                }   // ← 行本体 Row 结束
                            }       // ← 外层 Box（承载删除区）结束
                    }
                }
            }

        SnackbarHost(
            hostState = snackbar,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 108.dp),
        ) { data ->
            Snackbar(
                snackbarData = data,
                containerColor = t.ink,
                contentColor = t.paper,
                actionColor = t.moss2,
            )
        }
        }
    }

    val detail = state.items.firstOrNull { it.id == detailId }
    if (detail != null) {
        RecordDetailSheet(
            item = detail,
            onSave = { viewModel.save(it) },
            onDelete = { id -> viewModel.delete(id); detailId = null },
            onDismiss = { detailId = null },
            onView = { viewReq = it },
        )
    }
    viewReq?.let { v -> FullScreenImageView(v) { viewReq = null } }
}

private data class Cand(val sci: String, val zh: String, val prob: Float)

private fun parseCandidates(json: String?): List<Cand> = try {
    json?.let {
        val arr = org.json.JSONArray(it)
        (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            Cand(o.getString("sci"), o.optString("zh"), o.optDouble("prob", 0.0).toFloat())
        }
    }.orEmpty()
} catch (t: Throwable) {
    emptyList()
}

/** 日志详情弹层（纸感）：大图 + 全量字段 + Top-5 候选改判 + 备注编辑 + 删除。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RecordDetailSheet(
    item: ObservationEntity,
    onSave: (ObservationEntity) -> Unit,
    onDelete: (Long) -> Unit,
    onDismiss: () -> Unit,
    onView: (ViewerReq) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val fmt = remember { SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()) }
    val cands = remember(item.candidates) { parseCandidates(item.candidates) }
    // ⚠️ 2026-10-10 改 rememberSaveable(item.id)：未保存的备注在旋转后原先直接丢失。
    // 这里存的是"用户正在编辑但还没落库"的文本，丢了等于白输入。
    var note by rememberSaveable(item.id) { mutableStateOf(item.note) }
    val big by loadThumb(item.photoPath, 1080.dp)
    val bigImg = big

    PaperTheme {
        val t = LocalPaperPalette.current.t
        ModalBottomSheet(
            onDismissRequest = onDismiss,
            sheetState = sheetState,
            containerColor = t.card,
        ) {
            Column(Modifier.padding(horizontal = 20.dp, vertical = 4.dp)) {
                if (bigImg != null) {
                    Image(
                        bigImg.asImageBitmap(),
                        contentDescription = item.speciesZh,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 220.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .clickable {
                                onView(ViewerReq(listOfNotNull(item.photoPath), 0, item.speciesZh, item.speciesSci))
                            },
                    )
                    Spacer(Modifier.height(12.dp))
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        item.speciesZh,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = t.ink,
                        modifier = Modifier.weight(1f, fill = false),
                    )
                    // 保护等级在前、危险等级在后（2026-10-11 用户定序）
                    ProtectionPills(item.speciesSci, Modifier.padding(start = 6.dp))
                    if (item.confirmed) {
                        Spacer(Modifier.width(6.dp))
                        Image(
                            painter = painterResource(R.drawable.ic_check),
                            contentDescription = null,
                            modifier = Modifier.size(15.dp),
                            colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.moss),
                        )
                        Text(
                            " 人工确认",
                            style = MaterialTheme.typography.labelMedium,
                            color = t.moss,
                        )
                    }
                }
                PaperSciName(item.speciesSci, Modifier.padding(top = 2.dp), sizePx = 12)
                Spacer(Modifier.height(8.dp))
                // 快门时刻逆地理地址（无后端/无定位时为空，下面坐标行兜底）
                item.address?.let {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Image(
                            painter = painterResource(R.drawable.ic_place),
                            contentDescription = null,
                            modifier = Modifier.size(15.dp),
                            colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.olive),
                        )
                        Text(
                            " $it",
                            style = MaterialTheme.typography.bodyMedium,
                            color = t.ink,
                        )
                    }
                }
                // 元信息：时间/置信度/区域/经纬度
                val meta = buildString {
                    append(fmt.format(Date(item.observedAt)))
                    append(" · 置信 ${"%.0f%%".format(item.confidence * 100)}")
                    item.region?.let {
                        append(" · ${ModelRegistry.REGION_NAMES[it] ?: it}")
                    }
                    if (item.lat != null && item.lon != null) {
                        append(" · ${"%.4f, %.4f".format(item.lat, item.lon)}")
                    } else {
                        append(" · 无定位")
                    }
                }
                Text(
                    meta,
                    style = MaterialTheme.typography.bodySmall,
                    color = t.gray,
                )

                if (cands.isNotEmpty()) {
                    Spacer(Modifier.height(14.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        PaperTick()
                        Text(
                            "识别候选（点一条改判）",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = t.ink,
                        )
                    }
                    cands.forEach { c ->
                        val current = c.sci == item.speciesSci
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(10.dp))
                                .background(if (current) t.sage else Color.Transparent)
                                .clickable(enabled = !current) {
                                    onSave(
                                        item.copy(
                                            speciesSci = c.sci,
                                            speciesZh = c.zh,
                                            confidence = c.prob,
                                            confirmed = true,
                                        ),
                                    )
                                }
                                .padding(horizontal = 8.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                c.zh.ifBlank { c.sci },
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = if (current) FontWeight.Bold else FontWeight.Normal,
                                color = if (current) t.moss else t.ink,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                "${"%.0f%%".format(c.prob * 100)}${if (current) "  · 当前" else ""}",
                                style = MaterialTheme.typography.labelMedium,
                                color = if (current) t.olive else t.gray,
                            )
                        }
                    }
                }

                Spacer(Modifier.height(14.dp))
                OutlinedTextField(
                    value = note,
                    onValueChange = { note = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("备注", color = t.gray) },
                    minLines = 2,
                    shape = RoundedCornerShape(14.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = t.moss,
                        unfocusedBorderColor = t.line,
                        focusedLabelColor = t.moss,
                        unfocusedLabelColor = t.gray,
                        cursorColor = t.moss,
                        focusedTextColor = t.ink,
                        unfocusedTextColor = t.ink,
                    ),
                )
                Row(
                    modifier = Modifier.fillMaxWidth().padding(top = 10.dp, bottom = 24.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextButton(
                        onClick = { onDelete(item.id) },
                        colors = ButtonDefaults.textButtonColors(
                            contentColor = t.red,
                        ),
                    ) {
                        Text("删除")
                    }
                    if (note != item.note) {
                        TextButton(
                            onClick = { onSave(item.copy(note = note)) },
                            colors = ButtonDefaults.textButtonColors(
                                contentColor = t.moss,
                            ),
                        ) {
                            Text("保存备注")
                        }
                    }
                }
            }
        }
    }
}

package com.whatsbird.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed as lazyItemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState

import androidx.compose.animation.core.snap

import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.offset
import androidx.compose.ui.platform.LocalView
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
import com.whatsbird.ui.theme.PaperChipRow
import com.whatsbird.ui.theme.PaperSciName
import com.whatsbird.ui.theme.PaperSearchField
import com.whatsbird.ui.theme.PaperTheme
import com.whatsbird.ui.theme.PaperTick
import com.whatsbird.ui.theme.PaperTokens
import com.whatsbird.ui.theme.rememberReduceMotion
import com.whatsbird.vision.ModelRegistry
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/**
 * 图鉴 = 全国一册（384 种，骨架即 v2 全国模型 labels）+ 区域归属徽章，
 * 方案讨论与口径见 docs/design/DESIGN.md §4.4（2026-10-08 晚段）。
 * 要点：
 *  - 种池 = 模型能识别的全集（不存在永远点不亮的死格）；区域不再是"册"，
 *    每格标注该种归属哪些区（可多个），区域下拉降级为筛选器；
 *  - 进度一个数：点亮 N / 384；"已点亮"筛选 = life list 口径（加新序）；
 *  - 徽章与 IUCN 来源：六区 guide_*.json；记录种不在骨架内的（如东北旧底册
 *    的百步 Cuculus micropterus，v2 未收录）只进日志不进图鉴——已知行为。
 * 视觉 = 纸感收集册（docs/design/DESIGN.md）：白卡拍立得 + 三层名 +
 * 点缀刻度条；配色五选一（设置→外观→图鉴配色），取景页深色口径不受影响。
 */
data class GuideEntry(
    val sci: String,
    val zh: String,
    /**
     * IUCN 红色名录等级。**以 `guide/protection.json` 为准**（覆盖 380/384），
     * 只有该资产查不到的种才回落到六区 guide_*.json 的旧字段（旧数据仅 148/384，
     * 且含 9 条 `NE` 占位与 6 条已被重新评估修正的等级）。
     */
    val iucn: String,
    /**
     * 国家重点保护野生动物级别：`一级` / `二级` / 空（未列入）。
     * 与 [iucn] 是**两套独立体系**，颜色也分开，见 PaperTheme 的 IucnPill / CnProtectPill。
     */
    val cnLevel: String = "",
    val regions: List<String>, // **识别区** key（各区模型类别表；空 = 识别只能落全国兜底模型，不是"地理无归属"）
    /**
     * **地理分布区**（一级区中文名，来自数据集的 I–VII 区划代码，口径见 [ZooRegions]）。
     * 与 [regions] 完全解耦：识别区由 bbox+GBIF 采样决定、与区属统计无关；
     * 地理区是物种真实的分布。空 = 数据集未收该种的分布区（384 册里 50 种），显示「暂缺」。
     */
    val zooRegions: List<String> = emptyList(),
    val litAt: Long? = null,
    val count: Int = 0,
    val photos: List<String> = emptyList(), // 点亮后该种最近框选图（详情照片墙，≤6）
    val times: List<Long> = emptyList(),    // 该种观测时间（倒序，≤10）
    /** 对外展示的主图（用户在详情里长按小图指定）。null = 未指定，用 photos 里最新的一张。 */
    val cover: String? = null,
    /**
     * 物种性状（图鉴详情页「物种档案」卡片）。来源与口径见 [SpeciesTraits]。
     * null = 该种不在性状数据集覆盖范围内（384 种里 3 种），卡片显示「暂缺」。
     */
    val traits: SpeciesTraits? = null,
) {
    /** 实际用于展示的主图：指定封面优先，否则最新一张。 */
    val coverPath: String? get() = cover?.takeIf { it in photos } ?: photos.firstOrNull()
}

data class GuideUiState(
    val total: Int = 0,
    val litCount: Int = 0,
    val weekNew: Int = 0, // 近 7 日加新数（litAt 落入窗口）
    val entries: List<GuideEntry> = emptyList(),
    val query: String = "",
    val litFilter: GuideFilter = GuideFilter.ALL,
    val regionFilter: String? = null, // null = 全部区域
    /** ⑤（v6 P1-3）首次数据是否已落库。false 时图鉴墙走骨架占位，别给用户一片空白 */
    val loaded: Boolean = false,
)

/** ⑤ 首屏骨架铺几格（双列 × 三行，够撑满首屏又不至于太重） */
private const val GuideSkeletonCount = 6

/** 点亮态筛选：全部 / 已点亮（= life list 口径，按加新排序）/ 未见 */
enum class GuideFilter { ALL, LIT, UNLIT }

/** 区域徽章固定序（与 GPS 自动路由优先级一致，CameraScreen §REGION_KEYS） */
private val REGION_ORDER = listOf("huabei", "dongbei", "huadong", "huazhong", "huanan", "xinan")

@HiltViewModel
class GuideViewModel @Inject constructor(
    dao: ObservationDao,
    @ApplicationContext private val appCtx: android.content.Context,
    private val prefs: com.whatsbird.data.AppPrefs,
) : ViewModel() {

    private val query = MutableStateFlow("")
    private val litFilter = MutableStateFlow(GuideFilter.ALL)
    private val regionFilter = MutableStateFlow<String?>(null)

    /** 物种封面覆写（用户在详情里长按小图指定）。 */
    private val covers = prefs.covers

    /**
     * 全国一册底表：骨架 = v2 全国模型 labels（"sci\tzh" 384 行）；
     * 区域徽章 + IUCN 从六区 guide_*.json 联入（guide 里有、骨架没有的种不设格）。
     *
     * ⚠️ 2026-10-10：原先是 `by lazy`，首次访问发生在 stateIn 的 combine lambda 里，
     *   而 viewModelScope 默认 Main.immediate 且整条链没有 flowOn →
     *   读 7 个 assets + 解析 6 份 JSON 全在主线程，首次进图鉴必卡。
     *   改为 flow + flowOn(IO)，并在下面与记录流合并时用 flowOn(Default) 做重活。
     */
    private val catalogFlow: Flow<List<GuideEntry>> = flow { emit(loadCatalog()) }
        .flowOn(Dispatchers.IO)

    /** 解析结果缓存：保住原 `by lazy` 的"只解析一次"语义（stateIn 在空闲 5s 后会重新收集）。 */
    @Volatile private var catalogCache: List<GuideEntry>? = null

    private suspend fun loadCatalog(): List<GuideEntry> {
        catalogCache?.let { return it }
        return buildCatalog().also { catalogCache = it }
    }

    private fun buildCatalog(): List<GuideEntry> {
        val base = try {
            appCtx.assets.open(ModelRegistry.CN373_LABELS).bufferedReader().readLines()
                .filter { it.isNotBlank() }
                .map {
                    val p = it.split("\t")
                    p[0].trim() to p.getOrElse(1) { "" }.trim()
                }
        } catch (e: Exception) {
            emptyList()
        }
        val iucnBy = HashMap<String, String>()
        val regionBySci = HashMap<String, ArrayList<String>>()
        for (r in REGION_ORDER) {
            try {
                val arr = org.json.JSONArray(
                    appCtx.assets.open("guide/guide_$r.json").bufferedReader().readText())
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    val sci = o.getString("sci")
                    regionBySci.getOrPut(sci) { ArrayList() }.add(r)
                    o.optString("iucn").takeIf { it.isNotBlank() }?.let { iucnBy.putIfAbsent(sci, it) }
                }
            } catch (e: Exception) {
                // 单区 guide 缺失不阻塞整册：该区徽章为空，其余照常
            }
        }
        // 物种性状（详情页「物种档案」卡片）：384 种里覆盖 381 种，缺的 3 种卡片显示「暂缺」。
        // 与上面几个资产同一次 IO 读取。
        val traitsBySci = TraitsCatalog.load(appCtx)
        // 保护状态（详情页「保护状态」卡片 + 卡片页徽章）：380/384。
        // ⚠️ 这一份**取代**上面 iucnBy 的权威地位——guide_*.json 的 iucn 只有 148/384，
        //   且其中 9 条是 NE 占位、6 条等级已被新评估修正（详见 tools/bird_protection 的构建报告）。
        val protectionBySci = ProtectionCatalog.load(appCtx)
        return base.map { (sci, zh) ->
            val p = protectionBySci[sci]
            GuideEntry(
                sci = sci,
                zh = zh,
                iucn = p?.iucn?.takeIf { it.isNotBlank() } ?: iucnBy[sci] ?: "",
                cnLevel = p?.cn ?: "",
                regions = regionBySci[sci] ?: emptyList(),
                // 地理分布区：区划代码 → 一级区中文名。识别区之外的独立一层。
                zooRegions = ZooRegions.displayNames(traitsBySci[sci]?.distRange ?: emptyList()),
                traits = traitsBySci[sci],
            )
        }
    }

    /** 记录流 + 筛选/搜索条件 + 封面覆写（只是收集，重活留给下面合并时在 Default 线程做）。 */
    private data class GuideInputs(
        val all: List<ObservationEntity>,
        val q: String,
        val f: GuideFilter,
        val rf: String?,
        val covs: Map<String, String>,
    )

    private val inputs: Flow<GuideInputs> =
        combine(dao.observeAllAsc(), query, litFilter, regionFilter, covers) { all, q, f, rf, covs ->
            GuideInputs(all, q, f, rf, covs)
        }

    /** 记录流 × 一册底表 × 封面覆写 → 图鉴状态（点亮 + 筛选/搜索 + 主图）。 */
    val state: StateFlow<GuideUiState> =
        combine(inputs, catalogFlow) { inp, cat ->
            val bySci = HashMap<String, MutableList<ObservationEntity>>()
            for (o in inp.all) bySci.getOrPut(o.speciesSci) { ArrayList() }.add(o)
            val q = inp.q
            val f = inp.f
            val rf = inp.rf
            val entries = cat.map { e ->
                val rec = bySci[e.sci]
                if (rec.isNullOrEmpty()) {
                    e
                } else {
                    e.copy(
                        litAt = rec.first().observedAt, // observeAllAsc：头部=首见
                        count = rec.size,
                        photos = rec.asReversed().mapNotNull { it.photoPath }.take(6),
                        times = rec.asReversed().take(10).map { it.observedAt },
                        cover = inp.covs[e.sci],
                    )
                }
            }.filter { e ->
                val lit = e.litAt != null
                (f == GuideFilter.ALL || (f == GuideFilter.LIT) == lit) &&
                    (rf == null || e.zooRegions.contains(rf)) &&
                    (q.isBlank() || e.zh.contains(q, true) || e.sci.contains(q, true))
            }.sortedWith(
                compareBy<GuideEntry> { it.litAt == null }.thenBy { it.litAt ?: 0L },
            )
            // litCount = 一册中被任何记录命中的种数，与筛选/搜索解耦（进度口径稳定）
            val litCount = cat.count { bySci.containsKey(it.sci) }
            val weekAgo = System.currentTimeMillis() - 7L * 86400_000
            val weekNew = cat.count {
                val r = bySci[it.sci]?.firstOrNull()
                r != null && r.observedAt >= weekAgo
            }
            GuideUiState(cat.size, litCount, weekNew, entries, q, f, rf, loaded = true)
            // ⚠️ 2026-10-10：384 条的 map/filter/sort 原先也在主线程，这里挪到 Default。
        }.flowOn(Dispatchers.Default)
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), GuideUiState())

    fun setQuery(q: String) {
        query.value = q
    }

    fun setLitFilter(f: GuideFilter) {
        litFilter.value = f
    }

    /** null = 全部区域 */
    fun setRegionFilter(r: String?) {
        regionFilter.value = r
    }

    /** 长按小图设为对外展示的主图；再长按一次同一张 = 取消指定（回到最新一张）。 */
    fun setCover(sci: String, path: String) {
        viewModelScope.launch {
            val cur = covers.first()[sci]
            prefs.setCover(sci, if (cur == path) null else path)
        }
    }
}

/**
 * 地理分布区徽章串：来自数据集区划代码映射的一级区名，**全部写出**，
 * 放不下就缩字号（见 [regionBadgeFontSize]）。
 */
private fun regionBadges(zoo: List<String>): String = zoo.joinToString(" ")

/**
 * 徽章字号按区域个数递减。
 * 2 区 9sp（原有规格）；每多一区降 0.5sp；5 区及以上 7sp。
 * 不用 ellipsis 截断——截掉等于信息丢失，宁可字小一点。
 */
private fun regionBadgeFontSize(regionCount: Int): androidx.compose.ui.unit.TextUnit = when {
    regionCount <= 2 -> 10.sp
    regionCount == 3 -> 9.5.sp
    regionCount == 4 -> 9.sp
    regionCount == 5 -> 8.5.sp
    else -> 8.sp
}

/** 格位照片/剪影位（点亮=封面图；未点亮或无图=渐变底+鸟剪影+胶囊）。 */
@Composable
private fun PhotoOrSilhouette(
    e: GuideEntry,
    radius: Int,
    iconSize: Int,
    showPill: Boolean,
    pillLabel: String = "未点亮",
) {
    val t = LocalPaperPalette.current.t
    val lit = e.litAt != null
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(radius.dp))
            .background(Brush.linearGradient(listOf(t.silA, t.silB))),
        contentAlignment = Alignment.Center,
    ) {
        val photoPath = if (lit) e.coverPath else null
        // 2026-10-10：原先 remember{ decodeFile(inSampleSize=4) } 在主线程解码，滚动必卡
        val bmp by rememberDecoded(photoPath, maxEdge = 320.dp)
        val img = bmp
        if (img != null) {
            Image(
                img.asImageBitmap(), contentDescription = e.zh,
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            Image(
                painter = painterResource(R.drawable.ic_bird),
                contentDescription = null,
                modifier = Modifier
                    .size(iconSize.dp)
                    .graphicsLayer { alpha = if (lit) 0.9f else 0.5f },
                colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(
                    if (lit) t.moss2.copy(alpha = 0.55f) else t.silInk,
                ),
            )
            if (showPill && !lit) {
                Box(
                    Modifier
                        .align(Alignment.Center)
                        .padding(top = (iconSize * 0.9f).dp)
                        .clip(RoundedCornerShape(50))
                        .background(t.paper.copy(alpha = 0.92f))
                        .padding(horizontal = 10.dp, vertical = 3.dp),
                ) {
                    Text(
                        pillLabel,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Medium,
                        color = t.gray,
                    )
                }
            }
        }
    }
}

/** 图鉴格 = 白卡拍立得（微旋 + 柔影 + 三层名 + 角标）；动效见 动效规范 §2。 */
@Composable
private fun GuideCell(
    e: GuideEntry,
    index: Int,
    onView: (ViewerReq) -> Unit,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val t = LocalPaperPalette.current.t
    val lit = e.litAt != null
    val now = System.currentTimeMillis()
    val isNew = lit && now - (e.litAt ?: 0L) < 7L * 86400_000
    // 刚点亮（8 秒内首见）：进册仪式一次性播放——pop + 落羽 + 成功触觉
    // （全 app 唯一允许弹跳的地方，MOTION §0.1"收集仪式"）
    val justLit = lit && now - (e.litAt ?: 0L) < 8_000L
    val reduce = rememberReduceMotion()
    val pop = remember { Animatable(1f) }
    val fall = remember { Animatable(1f) } // 1=未播/隐藏，向 0 走=下落
    val viewRef = LocalView.current
    LaunchedEffect(justLit) {
        if (justLit && !reduce) {
            viewRef.performHapticFeedback(android.view.HapticFeedbackConstants.CONFIRM)
            fall.snapTo(1f)
            pop.animateTo(1.08f, tween(90))
            pop.animateTo(1f, spring(dampingRatio = Spring.DampingRatioMediumBouncy))
            fall.animateTo(0f, tween(1300))
        }
    }
    // 按下拿起（hover 的触屏翻译）：归正 + 抬 6dp + 阴影加深
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val lift by animateFloatAsState(
        if (pressed) 1f else 0f,
        spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessMediumLow,
        ),
        label = "lift",
    )
    // ×n 计数变化脉冲
    var pulse by remember { mutableStateOf(false) }
    LaunchedEffect(e.count) {
        if (e.count > 0 && !reduce) { pulse = true; delay(170); pulse = false }
    }
    val countScale by animateFloatAsState(if (pulse) 1.14f else 1f, spring(), label = "cnt")
    // 确定性微旋（收集册桌面感；≤2° 不碍阅读）
    val rot = floatArrayOf(-1.1f, 1.3f, -1.7f, 0.9f)[index % 4]
    Column(
        modifier
            .graphicsLayer {
                rotationZ = rot * (1f - lift)
                translationY = -6.dp.toPx() * lift
                scaleX = pop.value * (1f + 0.02f * lift)
                scaleY = pop.value * (1f + 0.02f * lift)
            }
            .shadow(
                (6f + 6f * lift).dp, RoundedCornerShape(16.dp),
                ambientColor = t.ink.copy(alpha = 0.25f), spotColor = t.ink.copy(alpha = 0.30f),
            )
            .clip(RoundedCornerShape(16.dp))
            .background(t.card)
            .clickable(interactionSource = interaction, indication = null, onClick = onClick)
            .padding(8.dp),
    ) {
        // 照片区不再单独拦截点击：**格内点图 = 进详情页**（与整卡点击一致）；
        // 全屏大图只在详情页里点大图才展开。
        Box(Modifier.fillMaxWidth()) {
            PhotoOrSilhouette(e, radius = 10, iconSize = 42, showPill = true)
            // 落羽粒子（justLit 期间，固定散布；与点亮同时只播一次）
            if (fall.value < 1f && e.photos.isNotEmpty()) {
                BoxWithConstraints(Modifier.matchParentSize()) {
                    val cellW = maxWidth
                    val drift = floatArrayOf(8f, 30f, 52f, 70f)
                    for (k in drift.indices) {
                        Image(
                            painter = painterResource(R.drawable.ic_feather),
                            contentDescription = null,
                            modifier = Modifier
                                .size(15.dp)
                                .offset(x = cellW * (drift[k] / 100f))
                                .graphicsLayer {
                                    translationY = (1f - fall.value) * 150.dp.toPx()
                                    rotationZ = (1f - fall.value) * 150f
                                    alpha = kotlin.math.sin(
                                        (1f - fall.value) * Math.PI.toFloat(),
                                    ) * 0.85f
                                },
                            colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.olive),
                        )
                    }
                }
            }
            if (isNew) {
                Text(
                    "new",
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(4.dp)
                        .clip(RoundedCornerShape(50))
                        .background(t.red)
                        .padding(horizontal = 8.dp, vertical = 2.dp),
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                )
            }
            if (lit && e.count > 1) {
                Text(
                    "×${e.count}",
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .padding(4.dp)
                        .graphicsLayer { scaleX = countScale; scaleY = countScale }
                        .clip(RoundedCornerShape(50))
                        // ⚠️ 2026-10-10 修夜间不可读：底色原先用 t.ink@0.72，而夜间 ink 被 mix 成
                        // 接近米白的浅色（PaperTheme.kt:73），白字压浅底几乎看不见。
                        // 改用**日间 ink**（palette.t.ink）：昼间完全不变，夜间得到深底白字。
                        .background(LocalPaperPalette.current.palette.t.ink.copy(alpha = 0.72f))
                        .padding(horizontal = 7.dp, vertical = 2.dp),
                    style = TextStyle(fontSize = 10.sp, fontWeight = FontWeight.SemiBold),
                    color = Color.White,
                )
            }
        }
        Column(Modifier.padding(start = 4.dp, end = 4.dp, top = 9.dp, bottom = 3.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    e.zh.ifBlank { e.sci },
                    style = MaterialTheme.typography.titleMedium,
                    fontSize = 15.sp,
                    fontWeight = if (lit) FontWeight.Bold else FontWeight.Medium,
                    color = t.ink.copy(alpha = if (lit) 1f else 0.75f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    // 徽章不加 weight，Compose 先量无 weight 的子项 → 徽章先拿足宽度，
                    // 名字吃剩余空间；名字过长时省略的是名字，徽章不会被挤没。
                    modifier = Modifier.weight(1f, fill = false),
                )
                // 徽章打在种名后面：保护等级在前、危险等级在后，**两枚一起出现**。
                // 口径统一在 ProtectionBadges：无行政级别 + IUCN 无危的种什么都不画。
                ProtectionBadges(e.iucn, e.cnLevel, Modifier.padding(start = 4.dp))
            }
            PaperSciName(e.sci, Modifier.padding(top = 1.dp))
            Row(
                Modifier.padding(top = 3.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                val badge = regionBadges(e.zooRegions)
                Text(
                    // 「暂缺」= 数据集没有该种的分布区代码（384 册里 50 种）。
                    // 旧文案「全国兜底」是错的：它原指"没被六区模型覆盖"，不是地理无归属。
                    badge.ifBlank { "分布暂缺" },
                    style = MaterialTheme.typography.labelSmall,
                    // 区域多时缩字号而不是截断成「+n」
                    fontSize = regionBadgeFontSize(e.zooRegions.size),
                    color = if (lit) t.olive else t.gray,
                    maxLines = 1,
                    // 不用 Ellipsis：宁可挤一点也不丢信息
                    overflow = TextOverflow.Clip,
                    softWrap = false,
                )
            }
        }
    }
}

/** 收集进度卡：点缀刻度 + 完成度（数字滚动 + 进度条注水）+ 加新羽毛。 */
@Composable
private fun ProgressHeader(state: GuideUiState) {
    val t = LocalPaperPalette.current.t
    val reduce = rememberReduceMotion()
    val fracT = if (state.total > 0) state.litCount.toFloat() / state.total else 0f
    val fill = remember { Animatable(0f) }
    LaunchedEffect(fracT) {
        if (reduce) fill.snapTo(fracT) else fill.animateTo(fracT, tween(600))
    }
    Column(
        Modifier
            .fillMaxWidth()
            .shadow(5.dp, RoundedCornerShape(18.dp), ambientColor = t.ink.copy(alpha = 0.10f), spotColor = t.ink.copy(alpha = 0.12f))
            .clip(RoundedCornerShape(18.dp))
            .background(t.card)
            .padding(16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            PaperTick()
            Text(
                "收集进度",
                fontSize = 15.sp,
                fontWeight = FontWeight.ExtraBold,
                color = t.ink,
            )
            Spacer(Modifier.weight(1f))
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    "${(fill.value * 100).toInt()}%",
                    fontSize = 20.sp, fontWeight = FontWeight.ExtraBold, color = t.moss,
                )
                Text("完成度", fontSize = 10.sp, color = t.gray)
            }
        }
        Box(
            Modifier
                .padding(top = 11.dp)
                .fillMaxWidth()
                .height(7.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(t.sage),
        ) {
            Box(
                Modifier
                    .fillMaxWidth(fill.value)
                    .fillMaxHeight()
                    .background(Brush.horizontalGradient(listOf(t.moss, t.moss2))),
            )
        }
        Row(
            Modifier.padding(top = 9.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "已点亮 ${state.litCount} / ${state.total} 种",
                fontSize = 11.5.sp,
                color = t.gray,
            )
            if (state.weekNew > 0) {
                Text(" · 本周加新 ", fontSize = 11.5.sp, color = t.gray)
                Text("+${state.weekNew}", fontSize = 11.5.sp, fontWeight = FontWeight.Bold, color = t.olive)
                Image(
                    painter = painterResource(R.drawable.ic_feather),
                    contentDescription = null,
                    modifier = Modifier
                        .padding(start = 3.dp)
                        .size(13.dp)
                        .graphicsLayer { rotationZ = -14f },
                    colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.olive),
                )
            }
        }
    }
}

/**
 * 「物种档案」卡片：物种性状（体型/繁殖/生态），纸感标本标签样式。
 *
 * 与「我的观测」的区别：这块是**物种知识**，不随用户记录变化，所以**未点亮也显示**——
 * 未点亮的详情页原本只有一句「尚未点亮」，加上档案后才是真正的知识卡
 * （见 docs/design/DESIGN.md §4.1「未点亮知识卡应升级为简介/辨识要点」）。
 *
 * 只渲染有数据的行；整组无数据则整组不出现。性状缺失（384 种里 3 种）显示「暂缺」。
 */
/**
 * 「保护状态」卡片：详情页只列三行，不标来源（2026-10-11 按用户要求精简）。
 *
 * ⚠️ 三行分属**三个互不换算的体系**，绝不能互相推断：
 *  1. IUCN 红色名录 = 国际风险评估（科学结论）；
 *  2. 国家重点保护野生动物名录 = 行政保护级别（不是评估结论）；
 *  3. 特有种 = 分布区属性。
 *
 * 「IUCN 高危」既不等于「国家一级保护」，反过来也不成立——金雕、白尾海雕 IUCN 无危
 * 却是国家一级；红头潜鸭、田鹀 IUCN 易危却未列入名录。所以三行必须**各自独立显示**，
 * 绝不能因为「IUCN 是 LC」就写「不受保护」（实测 77 个受保护种里 59 个是 LC）。
 *
 * ⚠️ 来源标注已从 UI 移除，但**数据来源仍然必须可查**：见
 * `docs/legal/LICENSES.md` §D 与 `assets/guide/protection.json` 的 `_meta`。
 * 「中国生物多样性红色名录」一行已整行删除（无公开机读数据源，长期显示「暂缺」无意义）。
 */
@Composable
private fun ProtectionCard(e: GuideEntry, t: PaperTokens) {
    Column(Modifier.padding(top = 16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            PaperTick()
            Text(
                "保护状态",
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = t.ink,
            )
        }
        ProtectionRow("IUCN", iucnName(e.iucn), t)
        ProtectionRow("保护级别", cnName(e.cnLevel), t)
        ProtectionRow(
            "特有种",
            when (e.traits) { null -> "暂缺"; else -> if (e.traits.endemic) "是" else "否" },
            t,
        )
    }
}

/** 保护状态单行：定宽标签 + 值。标签与值走 `alignByBaseline()` 保证首行基线对齐。 */
@Composable
private fun ProtectionRow(label: String, value: String, t: PaperTokens) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Text(
            label,
            fontSize = 12.sp,
            lineHeight = 17.sp,
            color = t.olive,
            modifier = Modifier.width(52.dp).alignByBaseline(),
        )
        Spacer(Modifier.width(6.dp))
        Text(
            value,
            fontSize = 12.5.sp,
            lineHeight = 17.sp,
            color = t.ink,
            modifier = Modifier.alignByBaseline(),
        )
    }
}

@Composable
private fun TraitsCard(tr: SpeciesTraits?, t: PaperTokens) {
    Column(Modifier.padding(top = 16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            PaperTick()
            Text(
                "物种档案",
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = t.ink,
            )
        }
        if (tr == null) {
            Text(
                "该种暂不在性状数据集覆盖范围内，暂无档案数据。",
                fontSize = 12.sp,
                color = t.gray,
                lineHeight = 17.sp,
                modifier = Modifier.padding(top = 6.dp),
            )
            return@Column
        }

        val taxon = listOfNotNull(tr.order, tr.family).joinToString(" · ")
        if (taxon.isNotBlank()) TraitRow("目 / 科", taxon, t)

        // —— 形态测量（原表分雌雄，故标出雄/雌）——
        val measures = listOfNotNull(
            fmtMeasured(tr.mass, "g")?.let { "体重" to it },
            fmtMeasured(tr.length, "mm")?.let { "体长" to it },
            fmtMeasured(tr.bill, "mm")?.let { "嘴峰" to it },
            fmtMeasured(tr.wing, "mm")?.let { "翅长" to it },
            fmtMeasured(tr.tail, "mm")?.let { "尾长" to it },
            fmtMeasured(tr.tarsus, "mm")?.let { "跗蹠长" to it },
        )
        if (measures.isNotEmpty()) {
            TraitGroup("形态测量", t)
            measures.forEach { TraitRow(it.first, it.second, t) }
        }

        // —— 繁殖 ——
        val breeding = listOfNotNull(
            fmtRange(tr.clutch, "枚")?.let { "窝卵数" to it },
            fmtEgg(tr.egg)?.let { "卵大小" to it },
            fmtCodes(tr.nestType, NEST_TYPE_NAME)?.let { "巢型" to it },
            fmtCodes(tr.nestSite, NEST_SITE_NAME)?.let { "巢址" to it },
        )
        if (breeding.isNotEmpty()) {
            TraitGroup("繁殖", t)
            breeding.forEach { TraitRow(it.first, it.second, t) }
        }

        // —— 生态 ——
        val ecology = listOfNotNull(
            tr.diet?.let { "食性" to it },
            fmtCodes(tr.flock, FLOCK_NAME)?.let { "集群" to it },
            fmtCodes(tr.migrant, MIGRANT_NAME)?.let { "居留型" to it },
            "特有种" to if (tr.endemic) "是" else "否",
            tr.provinces?.let { "分布省份" to "$it 省" },
        )
        if (ecology.isNotEmpty()) {
            TraitGroup("生态", t)
            ecology.forEach { TraitRow(it.first, it.second, t) }
        }

        Text(
            "数据源：中国鸟类生活史与生态学特征数据集（2021）· 数值为文献区间",
            fontSize = 10.sp,
            color = t.gray,
            lineHeight = 14.sp,
            modifier = Modifier.padding(top = 8.dp),
        )
    }
}

/** 档案分组小标题。 */
@Composable
private fun TraitGroup(title: String, t: PaperTokens) {
    Text(
        title,
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        color = t.olive,
        modifier = Modifier.padding(top = 10.dp, bottom = 1.dp),
    )
}

/** 档案单行：左侧定宽标签 + 右侧值（值可换行）。
 *
 * ⚠️ 2026-10-10 修正「项目和说明的文字没对齐」：原先标签 12sp、值 12.5sp 且没设
 * lineHeight，两者字号不同 → **基线不在同一条线上**，看着像上下错开。
 * 现在两侧都加 `alignByBaseline()`，用基线对齐而不是顶部对齐：
 * 保留"标签略小、灰色"的层级感，同时首行基线严格对齐；
 * 值换行时也只有首行参与对齐，后续行自然按 17sp 行距排。
 */
@Composable
private fun TraitRow(label: String, value: String, t: PaperTokens) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
        Text(
            label,
            fontSize = 12.sp,
            lineHeight = 17.sp,
            // ⚠️ 2026-10-10：原先用 t.gray，实机反馈「项目名称太淡」。
            // 12sp 的灰字在白卡上几乎糊掉；改用 t.olive（纸感体系的次级文字色），
            // 比 gray 深、比 ink 轻，层级还在但看得清。
            color = t.olive,
            modifier = Modifier.width(62.dp).alignByBaseline(),
        )
        // 标签最长约 48dp（「分布省份」），62dp 定宽本就有余量；
        // 这里再补 6dp 保险，万一以后出现更长的标签也不会和值贴���。
        Spacer(Modifier.width(6.dp))
        Text(
            value,
            fontSize = 12.5.sp,
            lineHeight = 17.sp,
            color = t.ink,
            modifier = Modifier.alignByBaseline(),
        )
    }
}

/** 物种详情弹层：点亮=大图+照片墙+观测时间线；未点亮=剪影知识卡（引导去拍）。 */
@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
private fun GuideDetailSheet(
    e: GuideEntry,
    onView: (ViewerReq) -> Unit,
    onSetCover: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val fmt = remember { SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()) }
    val lit = e.litAt != null
    val t = LocalPaperPalette.current.t
    // 详情页这一行改成**地理分布区**（一级区，数据集区划代码），
    // 不再展示"识别区"——后者是模型覆盖范围，和鸟分布在哪没关系。
    val regionLine = if (e.zooRegions.isEmpty()) "分布暂缺" else e.zooRegions.joinToString(" ")
    // 识别区另起一行小字，说清"能不能被区域模型认出来"是另一回事
    val modelLine = if (e.regions.isEmpty()) {
        "识别走全国模型"
    } else {
        "识别区：" + e.regions.joinToString(" ") { ModelRegistry.REGION_NAMES[it] ?: it }
    }
    // 大图当前展示哪一张：默认落在封面（= 用户长按指定的那张，或最新一张）
    var sel by remember(e.sci, e.cover) { mutableStateOf(e.photos.indexOf(e.coverPath)) }
    val heroPath = e.photos.getOrNull(sel) ?: e.coverPath

    PaperTheme {
        ModalBottomSheet(
            onDismissRequest = onDismiss,
            sheetState = sheetState,
            containerColor = t.card,
        ) {
            // ⚠️ 2026-10-10 修：这里原先是**没有滚动容器的 Column**。
            // ModalBottomSheet 自己不会滚内容，它只负责上滑与拖拽关闭手势；
            // 内容一旦比 sheet 高就**底部被切掉且滚不动**。
            // 加了「物种档案」 TraitsCard 后高度必然溢出，所以必须给 verticalScroll。
            // skipPartiallyExpanded = true 时 sheet 是全开展开态，滚动交给内容自己。
            Column(
                Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 20.dp, vertical = 4.dp),
            ) {
                if (lit && heroPath != null) {
                    val hero by rememberDecoded(heroPath, maxEdge = 640.dp)
                    val heroImg = hero
                    if (heroImg != null) {
                        Image(
                            heroImg.asImageBitmap(), contentDescription = e.zh,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(max = 220.dp)
                                .clip(RoundedCornerShape(14.dp))
                                .clickable {
                                    onView(ViewerReq(e.photos, sel.coerceAtLeast(0), e.zh, e.sci))
                                },
                        )
                        Spacer(Modifier.height(12.dp))
                    }
                } else {
                    // ⚠️ 2026-10-10 按用户要求：未点亮时详情页顶部要和**收集墙里的未点亮格**
                    // 是同一种观感（剪影 + silA→silB 渐变底），而不是原来那个
                    // 光秃秃的鸟图标浮在白卡上。
                    // 但「未点亮」的表现形式要和收集墙**区分开**：
                    //   收集墙 = 图标下方居中的胶囊（小、贴图、弱）
                    //   详情页 = 右上角角标 + 整图虚线感描边（角、外扩、明确）
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(176.dp)
                            .clip(RoundedCornerShape(14.dp))
                            .background(Brush.linearGradient(listOf(t.silA, t.silB))),
                        contentAlignment = Alignment.Center,
                    ) {
                        Image(
                            painter = painterResource(R.drawable.ic_bird),
                            contentDescription = null,
                            modifier = Modifier.size(72.dp).graphicsLayer { alpha = 0.5f },
                            colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.silInk),
                        )
                        // 区分点 1：内描边把整块圈成"待填的槽位"
                        Box(
                            Modifier
                                .fillMaxSize()
                                .border(
                                    1.dp, t.line.copy(alpha = 0.85f),
                                    RoundedCornerShape(14.dp),
                                ),
                        )
                        // 区分点 2：角标挪到右上角，而不是图标正下方
                        Box(
                            Modifier
                                .align(Alignment.TopEnd)
                                .padding(10.dp)
                                .clip(RoundedCornerShape(50))
                                .background(t.card.copy(alpha = 0.95f))
                                .padding(horizontal = 10.dp, vertical = 4.dp),
                        ) {
                            Text(
                                "未点亮",
                                fontSize = 10.5.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = t.gray,
                            )
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        e.zh.ifBlank { e.sci },
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = t.ink,
                        // 同卡片：徽章不加 weight，先量先拿宽度，名字过长时省略的是名字
                        modifier = Modifier.weight(1f, fill = false),
                    )
                    // 保护等级在前、危险等级在后，两枚一起出现（同卡片口径）
                    ProtectionBadges(e.iucn, e.cnLevel, Modifier.padding(start = 8.dp))
                }
                PaperSciName(e.sci, Modifier.padding(top = 2.dp), sizePx = 13)
                Row(
                    Modifier.padding(top = 6.dp),
                    verticalAlignment = Alignment.Top,
                ) {
                    Image(
                        painter = painterResource(R.drawable.ic_place),
                        contentDescription = null,
                        modifier = Modifier.size(14.dp),
                        colorFilter = androidx.compose.ui.graphics.ColorFilter.tint(t.olive),
                    )
                    Column(Modifier.padding(start = 6.dp)) {
                        Text(
                            "分布：$regionLine",
                            style = MaterialTheme.typography.bodyMedium,
                            color = t.ink,
                        )
                        // 识别区与地理分布是两件事，分开写清楚
                        Text(
                            modelLine,
                            fontSize = 11.sp,
                            color = t.gray,
                            modifier = Modifier.padding(top = 2.dp),
                        )
                    }
                }
                if (lit) {
                    Spacer(Modifier.height(12.dp))
                    // 照片条：左右滑动选图（大图跟随），长按设为对外展示的主图。
                    // 只有一张时不做多余手势，仍保留点击进大图。
                    LazyRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        lazyItemsIndexed(e.photos) { idx, path ->
                            val bmp by rememberDecoded(path, maxEdge = 128.dp)
                            val img = bmp
                            if (img != null) {
                                val isSel = idx == sel
                                val isCover = path == e.coverPath
                                Image(
                                    img.asImageBitmap(), contentDescription = null,
                                    contentScale = ContentScale.Crop,
                                    modifier = Modifier
                                        .size(64.dp)
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(t.sage)
                                        .border(
                                            if (isSel) 2.dp else 1.dp,
                                            if (isSel) t.olive else t.line,
                                            RoundedCornerShape(10.dp),
                                        )
                                        .combinedClickable(
                                            onClick = { sel = idx },
                                            onLongClick = {
                                                onSetCover(path)
                                                sel = idx
                                            },
                                        ),
                                )
                            }
                        }
                    }
                    if (e.photos.size > 1) {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "← 左右滑动换图 · 长按设为封面" +
                                if (e.cover != null) "（已指定）" else "（当前用最新一张）",
                            style = MaterialTheme.typography.bodySmall,
                            color = t.gray,
                        )
                    }
                    Spacer(Modifier.height(14.dp))
                    // 观测时间线：琥珀点 + 时间（流水明细归日志页）
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        PaperTick()
                        Text(
                            "我的观测 · 首见 ${fmt.format(Date(e.litAt!!))} · 共 ${e.count} 次",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            color = t.ink,
                        )
                    }
                    e.times.take(5).forEach { ts ->
                        Row(
                            Modifier.padding(top = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Box(
                                Modifier
                                    .size(5.dp)
                                    .clip(RoundedCornerShape(50))
                                    .background(t.amber),
                            )
                            Text(
                                "  ${fmt.format(Date(ts))}",
                                style = MaterialTheme.typography.bodySmall,
                                color = t.gray,
                            )
                        }
                    }
                    Text(
                        "流水明细见「日志」页（搜「${e.zh.ifBlank { e.sci }}」）",
                        style = MaterialTheme.typography.labelSmall,
                        color = t.gray,
                        modifier = Modifier.padding(top = 10.dp),
                    )
                } else {
                    // ⚠️ 2026-10-10 按用户要求删掉这里的「尚未点亮」文字：
                    // 未点亮的状态已由顶部那张图自己表达（剪影 + 角标），
                    // 再写一遍是重复，而且会把下面的物种档案往下顶。
                    Unit
                }
                // 保护状态：点亮与否都显示——它同样是物种知识，不是用户数据。
                // 放在「我的观测」之后、「物种档案」之前：先讲"这鸟是什么处境"，
                // 再讲"它长什么样"。四行全列，每行标出处来源。
                ProtectionCard(e, t)
                // 物种档案（性状）：点亮与否都显示——它是物种知识，不是用户数据。
                // 放最后，保证「我的观测」仍是最先看到的（与旧版一致）。
                TraitsCard(e.traits, t)
                Spacer(Modifier.height(24.dp))
            }
        }
    }
}

/** 图鉴页（全国一册 + 区域徽章；纸感收集册皮肤，docs/design/DESIGN.md）。 */
@Composable
fun GuideScreen(
    modifier: Modifier = Modifier,
    viewModel: GuideViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    // ⚠️ 2026-10-10 改 rememberSaveable：旋转/进程重建后原先详情页与大图请求直接丢失。
    var detailSci by rememberSaveable { mutableStateOf<String?>(null) }
    var viewReq by rememberSaveable(stateSaver = ViewerReqSaver) { mutableStateOf<ViewerReq?>(null) }

    PaperTheme {
        val t = LocalPaperPalette.current.t
        Column(
            modifier = modifier
                .fillMaxSize()
                .background(t.paper)
                // 状态栏避让（顶卡此前被时间/电量遮挡）
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(12.dp),
        ) {
            ProgressHeader(state)

            // 筛选一：点亮态 —— ⑦ 半边二：改为共享滑动指示块（此前 PaperChip 各自瞬切）
            val litLabels = listOf("全部", "已点亮", "未见")
            PaperChipRow(
                selectedIndex = listOf(GuideFilter.ALL, GuideFilter.LIT, GuideFilter.UNLIT)
                    .indexOf(state.litFilter).coerceAtLeast(0),
                onSelect = { i ->
                    viewModel.setLitFilter(
                        listOf(GuideFilter.ALL, GuideFilter.LIT, GuideFilter.UNLIT)[i]
                    )
                },
                labels = litLabels,
                modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                contentPadding = PaddingValues(top = 2.dp, bottom = 2.dp),
            )
            // 筛选二：地理分布区（七区一级制，口径见 ZooRegions；横滚，指示块跟随滚动内容）。
            // 与「点亮态」不同，这一排是**地理筛选**，所以七个区全列——
            // 蒙新/青藏没有识别模型，但数据里有它们的分布，照样能筛。
            val zooNames = ZooRegions.ORDER.map { ZooRegions.name(it) }
            PaperChipRow(
                selectedIndex = if (state.regionFilter == null) 0
                else (zooNames.indexOf(state.regionFilter) + 1).coerceAtLeast(0),
                onSelect = { i ->
                    if (i == 0) viewModel.setRegionFilter(null) else viewModel.setRegionFilter(zooNames[i - 1])
                },
                labels = listOf("全部分布区") + zooNames,
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                scrollable = true,
                contentPadding = PaddingValues(top = 2.dp, bottom = 2.dp),
            )
            PaperSearchField(
                value = state.query,
                onValueChange = viewModel::setQuery,
                placeholder = "搜索鸟种（中文名/学名）",
                modifier = Modifier.padding(top = 10.dp),
            )

            LazyVerticalGrid(
                columns = GridCells.Adaptive(minSize = 150.dp),
                modifier = Modifier.fillMaxSize().padding(top = 10.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    start = 4.dp, end = 4.dp, bottom = 96.dp,
                ),
            ) {
                // ⑤（v6 P1-3）首屏未落库时先铺骨架，别给用户一片空白。
                // 「真组件喂假数据」——骨架块与真卡同宽同圆角，交接时位置不会跳。
                if (!state.loaded) {
                    items(GuideSkeletonCount) { k ->
                        PaperSkeletonBox(
                            Modifier
                                .fillMaxWidth()
                                .aspectRatio(0.78f)
                                .animateItem(fadeInSpec = snap()),
                        )
                    }
                    return@LazyVerticalGrid
                }
                itemsIndexed(state.entries, key = { _, e -> e.sci }) { i, e ->
                    // ④（v6 P1-2）收集墙 stagger 调参：
                    // 原式 `tween(180 + (i % 10) * 25)` 是**渐强延迟**——第 30 项要等 900ms
                    // 才动，用户会以为卡了。改成抄 StaggeredAnimation 的
                    // **固定 25ms 步进 + 200ms 封顶**，再补上浮与缩放
                    // （原版 100px / 0.95 对我们的拍立得太大，收到 48px / 0.9）。
                    // 落位继续用我们自己的 spring(placementSpec)，不换成无回弹的 smooth。
                    // 淡入/上浮/缩放由自己的 Animatable 统一驱动——animateItem 的 fadeInSpec
                    // 把 alpha 挂在外层节点上，graphicsLayer 里读不到，别指望反推。
                    val reduceS = com.whatsbird.ui.theme.rememberReduceMotion()
                    val enterDelayMs = if (reduceS) 0 else (i * 25).coerceAtMost(200)
                    val enter = remember(i, reduceS) { Animatable(if (reduceS) 1f else 0f) }
                    LaunchedEffect(i, reduceS) {
                        if (!reduceS) {
                            enter.snapTo(0f)
                            delay(enterDelayMs.toLong())
                            enter.animateTo(
                                1f,
                                tween(300, easing = androidx.compose.animation.core.FastOutSlowInEasing),
                            )
                        }
                    }
                    GuideCell(
                    e, index = i, onView = { viewReq = it },
                    modifier = Modifier
                        .animateItem(
                            // 淡入交给自己的 enter，这里只保留位置动画（重排时）
                            fadeInSpec = snap(),
                            placementSpec = spring(dampingRatio = Spring.DampingRatioLowBouncy),
                        )
                        .graphicsLayer {
                            val v = enter.value
                            alpha = v
                            translationY = 48f * (1f - v)
                            scaleX = 0.9f + 0.1f * v
                            scaleY = 0.9f + 0.1f * v
                        },
                ) { detailSci = e.sci }
                }
            }
        }
    }

    val detail = state.entries.firstOrNull { it.sci == detailSci }
    if (detail != null) {
        val scope = rememberCoroutineScope()
        GuideDetailSheet(
            e = detail,
            onView = { viewReq = it },
            onSetCover = { path -> scope.launch { viewModel.setCover(detail.sci, path) } },
            onDismiss = { detailSci = null },
        )
    }
    viewReq?.let { v -> FullScreenImageView(v) { viewReq = null } }
}

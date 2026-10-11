package com.whatsbird.ui

import android.content.Context
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.whatsbird.ui.theme.CnProtectPill
import com.whatsbird.ui.theme.IucnPill
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * 物种保护状态（图鉴详情页「保护状态」卡片 + 卡片页徽章的数据模型）。
 *
 * 这里有**三套互不相同的体系**，务必别混用：
 *
 * 1. **IUCN 红色名录**（国际通用，科学评估）
 *    CR 极危 / EN 濒危 / VU 易危 / NT 近危 / LC 无危 / DD 数据缺乏 / NR 未评估。
 * 2. **国家重点保护野生动物名录（调整版）**（行政级别，不是评估结论）
 *    一级 / 二级；空 = 未列入重点保护名录。
 * 3. **有重要生态、科学、社会价值的陆生野生动物名录（2023 版，俗称"三有"）**
 *    三有；比一/二级低一档。⚠️ 与第 2 条**互斥**（三有名录明确排除重点保护种），
 *    骨架 384 种实测交集为 0，所以 2/3 共用 [ProtectionStatus.cn] 一个字段。
 *
 * ⚠️ 两套**经常不一致**，而且两个方向都有（本项目 384 种实测）：
 *  - 金雕、白尾海雕、黑鹳：IUCN 无危 LC，却是**国家一级**保护；
 *  - 红头潜鸭、尖尾滨鹬、田鹀：IUCN **易危 VU**，却**未列入**保护名录；
 *  - 77 个受保护种里有 **59 个（76%）IUCN 是无危** —— 所以"无危"绝不能读作"不受保护"；
 *  - 中华秋沙鸭：IUCN 全球评 EN 濒危，而中国本土红色名录评 CR 极危、另列为国家一级——
 *    同一只鸟三个体系三种答案。（**中国红名录本项目未接入**，见 docs/legal/LICENSES.md §D；
 *    这里提它是为了说明"为什么不同来源的数字不能互相换算"。）
 * 所以**不能用一套颜色画两种徽章**——颜色必须分成两套（见 PaperTheme 的
 * [com.whatsbird.ui.theme.IucnPill] 与 `CnProtectPill`）。
 *
 * **「三有」覆盖极广**：鸟纲 1028 种，本项目骨架 384 种里 **302 种是三有**
 * （重点保护只有 77 种），所以接上三有之后绝大多数卡片都会带徽章。
 * 这是用户 2026-10-11 明确要求的（"该是三有的打上三有"），不是数据噪声。
 * 三有名录只收鸟纲之外的类群按各自分节，本项目只取**鸟纲**那一节。
 * 来源：国家林业和草原局公告 2023 年第 17 号（官方附件是扫描件 PDF，
 * 改从转载页的 .doc 抽取，见 tools/bird_protection/extract_sanyou_birds.py）。
 *
 * 资产 `assets/guide/protection.json` 由 `tools/bird_protection/build_protection_asset.py`
 * 生成（覆盖 380/384）；来源与授权状态同样见 `docs/legal/LICENSES.md` §D。
 * ⚠️ 本资产**取代** guide_*.json 里的 `iucn` 字段成为权威源；guide 的 iucn 只作
 * 兜底（旧数据只有 148/384，且其中 9 条是 `NE` 占位、6 条等级已被重新评估修正）。
 */
data class ProtectionStatus(
    /** IUCN 等级代码，空 = 未评估/无数据。 */
    val iucn: String = "",
    /** 行政保护级别 `一级` / `二级` / `三有`，空 = 两份名录都没收。 */
    val cn: String = "",
)

/** 保护状态资产加载器。
 *
 * ⚠️ 与 `TraitsCatalog` 同一线程约束：**不要**在主线程/组合期首次调用。
 * 资产 ~17 KB、380 条，故加**进程级缓存**：启动时由 [com.whatsbird.WhatsBirdApp] 预热一次，
 * 之后图鉴 / 取景 / 日志 / 全屏看图各页读内存即得（见 [rememberProtectionIndex]）。
 * 资产缺失/损坏时返回空表，各页降级为不显示徽章，不崩。
 */
object ProtectionCatalog {
    private const val ASSET = "guide/protection.json"

    /** 进程级缓存。`@Volatile` 保证多线程可见；解析幂等，最坏情况并发解析两次。 */
    @Volatile
    private var cached: Map<String, ProtectionStatus>? = null

    /** 已解析的索引；尚未解析时为 null（调用方据此决定要不要异步加载）。 */
    val cachedOrNull: Map<String, ProtectionStatus>? get() = cached

    /** 解析资产；结果缓存，重复调用不再读盘。 */
    fun load(ctx: Context): Map<String, ProtectionStatus> {
        cached?.let { return it }
        val parsed = try {
            val root = JSONObject(ctx.assets.open(ASSET).bufferedReader().readText())
            val sp = root.optJSONObject("species")
            if (sp == null) {
                emptyMap()
            } else {
                val out = HashMap<String, ProtectionStatus>(sp.length() * 2)
                for (sci in sp.keys()) {
                    sp.optJSONObject(sci)?.let {
                        out[sci] = ProtectionStatus(
                            iucn = it.optString("iucn").trim().uppercase(),
                            cn = it.optString("cn").trim(),
                        )
                    }
                }
                out
            }
        } catch (e: Exception) {
            android.util.Log.w("ProtectionCatalog", "保护状态资产不可用，各页降级为无徽章", e)
            emptyMap()
        }
        cached = parsed
        // 常驻：每个进程只打一次，出问题时一眼能看出"是数据没加载，还是没匹配上"。
        android.util.Log.i("ProtectionCatalog", "已加载 $ASSET：${parsed.size} 条")
        return parsed
    }
}

// ------------------------------------------------------------------ 各页共享的读取入口

/**
 * 取保护状态索引。启动预热过就是内存命中；万一没热（进程刚起就进相机），
 * 退化成一次 IO 线程加载，**不在组合期读盘**。
 */
@Composable
fun rememberProtectionIndex(): Map<String, ProtectionStatus> {
    val ctx = LocalContext.current.applicationContext
    val initial = remember(ctx) { ProtectionCatalog.cachedOrNull ?: emptyMap() }
    val st by produceState(initial, ctx) {
        // ⚠️ 必须**无条件**赋值。不能写成 `if (cachedOrNull == null)`：
        // 预热线程可能在 `initial` 取值之后、本协程真正启动之前刚好把缓存写完，
        // 那样条件为假 → value 永远停在 emptyMap → 徽章一个都不显示。
        // load() 自带缓存，命中就是纯内存操作，重复赋值无代价。
        value = withContext(Dispatchers.IO) { ProtectionCatalog.load(ctx) }
    }
    return st
}

/**
 * 种名后紧跟的保护徽章组：**保护等级在前、危险等级在后**（2026-10-11 用户定序）。
 *
 * ⚠️ 两枚**要么一起画、要么都不画**。只在"有话说"时才出现：
 * 该种有行政保护级别，或 IUCN 属于受威胁等级（CR/EN/VU/NT）。
 * 但一旦出现，**IUCN 那枚一定画**（含"无危 LC"）——
 * 实测 77 个受保护种里 **56 个 IUCN 是无危**，只画行政那枚会让人误以为
 * "受保护的鸟 IUCN 那栏一定是红的"。
 *
 * 接上「三有」之后，骨架 384 种里只剩 **2 种**（小嘴乌鸦、大嘴乌鸦，两份名录都没收）
 * 什么都不画；其余 379 种至少带一枚（302 种是三有）。
 */
@Composable
fun ProtectionBadges(iucn: String, cnLevel: String, modifier: Modifier = Modifier) {
    val hasCn = cnLevel.isNotBlank()
    if (!hasCn && !iucnPinned(iucn)) return
    Row(modifier, verticalAlignment = Alignment.CenterVertically) {
        if (hasCn) CnProtectPill(cnLevel)
        if (iucn.isNotBlank()) {
            IucnPill(iucn, if (hasCn) Modifier.padding(start = 4.dp) else Modifier)
        }
    }
}

/**
 * 按**学名**取徽章组——取景页 / 日志页 / 全屏看图用这个（它们手上只有学名）。
 * 口径与图鉴卡片完全一致，因为最终都走 [ProtectionBadges]。
 */
@Composable
fun ProtectionPills(sci: String, modifier: Modifier = Modifier) {
    if (sci.isBlank()) return
    val p = rememberProtectionIndex()[sci]
    ProtectionBadges(iucn = p?.iucn ?: "", cnLevel = p?.cn ?: "", modifier = modifier)
}

// ------------------------------------------------------------------ 展示格式化

/** IUCN 等级全称。未知代码原样返回，不吞掉。 */
internal fun iucnName(code: String): String = when (code) {
    "CR" -> "极危 CR"
    "EN" -> "濒危 EN"
    "VU" -> "易危 VU"
    "NT" -> "近危 NT"
    "LC" -> "无危 LC"
    "DD" -> "数据缺乏 DD"
    "NR" -> "未评估 NR"
    "NE" -> "未评估 NE"
    "" -> "暂缺"
    else -> code
}

/**
 * 是否值得在卡片上挂 IUCN 徽章。
 * 只有四档受威胁等级才值得——LC/NR/NE 挂满 384 张卡片就是噪声。
 */
internal fun iucnPinned(code: String): Boolean =
    code == "CR" || code == "EN" || code == "VU" || code == "NT"

/** 行政保护级别全称。空 = 两份名录都没收。 */
internal fun cnName(level: String): String = when (level) {
    "一级" -> "国家一级保护"
    "二级" -> "国家二级保护"
    "三有" -> "三有保护动物"
    else -> "未列入保护名录"
}
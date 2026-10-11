package com.whatsbird.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import com.whatsbird.vision.ModelRegistry
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.settingsStore by preferencesDataStore(name = "settings")

/**
 * 全局行为偏好（设置页落库）。外观在 ThemePrefs，别混到这里。
 * 键少而稳：只放"用户真的会去设置页改"的东西，试验性参数留在代码常量。
 */
@Singleton
class AppPrefs @Inject constructor(@ApplicationContext context: Context) {

    private val store = context.settingsStore

    /** 取景页"自动放大"持久默认（进页初始态；页内芯片仍可会话级临时开关）。默认关=旧行为。 */
    val autoZoomDefault: Flow<Boolean> = store.data.map { it[AUTO_ZOOM] ?: false }

    suspend fun setAutoZoomDefault(v: Boolean) {
        store.edit { it[AUTO_ZOOM] = v }
    }

    /** 性能监控叠加层（电池温度 / 帧率 / 推理延迟）。默认关。 */
    val perfMonitor: Flow<Boolean> = store.data.map { it[PERF_MONITOR] ?: false }

    suspend fun setPerfMonitor(v: Boolean) {
        store.edit { it[PERF_MONITOR] = v }
    }

    /**
     * 图鉴物种封面覆写：学名 -> 照片路径。
     * 用户在物种详情里长按某张小图把它设成"对外展示的主图"，记在这里。
     * DataStore 的 stringPreferencesKey 只能存字符串，这里用 `sci|path;sci|path` 自行编码
     * （学名与路径都不含 `;`，安全）。
     */
    val covers: Flow<Map<String, String>> = store.data.map { p ->
        decodeCovers(p[COVERS])
    }

    /** 设封面；path 传 null 表示取消覆写（回到"最新一张"）。 */
    suspend fun setCover(sci: String, path: String?) {
        store.edit { p ->
            val m = decodeCovers(p[COVERS]).toMutableMap()
            if (path.isNullOrBlank()) m.remove(sci) else m[sci] = path
            p[COVERS] = m.entries.joinToString(";") { "${it.key}|${it.value}" }
        }
    }

    /**
     * 检测器档位覆盖（2026-10-10）：`auto`（按 RAM 自动）/ `flag`（旗舰 640）/ `mid`（中档 512）。
     * 默认 auto。⚠️ 存在的理由是**真机实测发现 auto 判错过一次**：
     * 标称 12GB 的小米 15，`totalMem` 只有 10.86 GiB，被 12 GiB 阈值判成低内存走中档，
     * 而中档的 tiny 比旗舰低 30%（正是 10x 小个体麻雀那一档）。
     * 阈值已改为 9 GiB，但保留手动覆盖——阈值只有一台实测样本，用户能自己纠偏。
     */
    val detTier: Flow<String> = store.data.map { p ->
        val raw = p[DET_TIER]
        when (raw) {
            ModelRegistry.TIER_FLAG, ModelRegistry.TIER_MID -> raw
            else -> ModelRegistry.TIER_AUTO
        }
    }

    suspend fun setDetTier(v: String) {
        store.edit { p ->
            p[DET_TIER] = when (v) {
                ModelRegistry.TIER_FLAG, ModelRegistry.TIER_MID -> v
                else -> ModelRegistry.TIER_AUTO
            }
        }
    }

    private fun decodeCovers(raw: String?): Map<String, String> {
        if (raw.isNullOrBlank()) return emptyMap()
        return raw.split(';').mapNotNull { e ->
            val i = e.indexOf('|')
            if (i <= 0) null else e.substring(0, i) to e.substring(i + 1)
        }.toMap()
    }

    private companion object {
        val AUTO_ZOOM = booleanPreferencesKey("auto_zoom_default")
        val PERF_MONITOR = booleanPreferencesKey("perf_monitor")
        val COVERS = stringPreferencesKey("guide_covers")
        val DET_TIER = stringPreferencesKey("detector_tier")
    }
}
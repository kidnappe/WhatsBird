package com.whatsbird.data

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.themeStore by preferencesDataStore(name = "theme")

/**
 * 外观偏好持久化（docs/design/DESIGN.md）：
 * - paperPalette：纸感色板 key（PaperPalettes；null = 默认"苔绿"）
 * - nightMode：夜间模式三态 "system"/"on"/"off"（null = system）
 * 历史 seedColor（壁纸动态取色）已于 2026-10-08 移除；
 * DataStore 里如有旧 seed_color 键不再读写，自然残留无害。
 */
@Singleton
class ThemePrefs @Inject constructor(@ApplicationContext context: Context) {

    private val store = context.themeStore

    val paperPalette: Flow<String?> =
        store.data.map { it[PAPER] }

    suspend fun setPaperPalette(key: String?) {
        store.edit {
            if (key == null) it.remove(PAPER) else it[PAPER] = key
        }
    }

    /** "system" | "on" | "off" */
    val nightMode: Flow<String?> =
        store.data.map { it[NIGHT] }

    suspend fun setNightMode(mode: String) {
        store.edit { it[NIGHT] = mode }
    }

    private companion object {
        val PAPER = stringPreferencesKey("paper_palette")
        val NIGHT = stringPreferencesKey("night_mode")
    }
}

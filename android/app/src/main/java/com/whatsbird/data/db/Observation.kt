package com.whatsbird.data.db

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Fts4
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

/**
 * 一条观测记录。
 * v2 字段：candidates=记录时的 Top-5 候选 JSON；confirmed=用户从候选中人工确认过；
 * photoPath=黄框裁剪图（files/records/ 下）；region=识别时使用的区域模型。
 * v3 字段：address=点击「记录此鸟」时刻由经纬度逆地理编码出的地址（可空，
 * 无定位后端/离线时为空，经纬度本身不受影响）。
 */
@Entity(tableName = "observations")
data class ObservationEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val speciesSci: String,
    val speciesZh: String,
    val confidence: Float,
    val lat: Double? = null,
    val lon: Double? = null,
    val observedAt: Long,
    val note: String = "",
    val candidates: String? = null,
    val confirmed: Boolean = false,
    val photoPath: String? = null,
    val region: String? = null,
    val address: String? = null,
)

/** FTS 全文检索（PLAN §7：Room 含 FTS），覆盖物种名与笔记。 */
@Fts4(contentEntity = ObservationEntity::class)
@Entity(tableName = "observations_fts")
data class ObservationFts(
    val speciesSci: String,
    val speciesZh: String,
    val note: String,
)

@Dao
interface ObservationDao {
    @Insert
    suspend fun insert(o: ObservationEntity): Long

    /** 日志详情编辑用：改判/备注都走整行回写。 */
    @androidx.room.Update
    suspend fun update(o: ObservationEntity)

    @Query("SELECT * FROM observations ORDER BY observedAt DESC")
    fun observeAll(): Flow<List<ObservationEntity>>

    @Query(
        "SELECT o.* FROM observations o " +
            "JOIN observations_fts f ON o.id = f.rowid " +
            "WHERE observations_fts MATCH :query " +
            "ORDER BY o.observedAt DESC",
    )
    fun search(query: String): Flow<List<ObservationEntity>>

    @Query("SELECT COUNT(*) FROM observations")
    suspend fun count(): Int

    @Query("DELETE FROM observations WHERE id = :id")
    suspend fun delete(id: Long)

    /** 设置页清空全部记录（FTS content 表由 Room 触发器同步清理）。 */
    @Query("DELETE FROM observations")
    suspend fun deleteAll()

    /** 图鉴/点亮聚合用：全部记录按时间正序（头部即首见），调用方按种聚合。 */
    @Query("SELECT * FROM observations ORDER BY observedAt ASC")
    fun observeAllAsc(): Flow<List<ObservationEntity>>
}

@Database(
    entities = [ObservationEntity::class, ObservationFts::class],
    version = 3,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun observationDao(): ObservationDao

    companion object {
        /** v1 -> v2：新增记录丰富度字段（旧记录字段留空）。 */
        val MIGRATION_1_2 = object : androidx.room.migration.Migration(1, 2) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE observations ADD COLUMN candidates TEXT")
                db.execSQL("ALTER TABLE observations ADD COLUMN confirmed INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE observations ADD COLUMN photoPath TEXT")
                db.execSQL("ALTER TABLE observations ADD COLUMN region TEXT")
            }
        }

        /** v2 -> v3：记录快门时刻的逆地理地址（旧行为 NULL，展示降级为经纬度）。 */
        val MIGRATION_2_3 = object : androidx.room.migration.Migration(2, 3) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE observations ADD COLUMN address TEXT")
            }
        }
    }
}

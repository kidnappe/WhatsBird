package com.whatsbird.di

import android.content.Context
import androidx.room.Room
import com.whatsbird.data.db.AppDatabase
import com.whatsbird.data.db.ObservationDao
import com.whatsbird.vision.BirdClassifier
import com.whatsbird.vision.BirdDetector
import com.whatsbird.vision.ModelRegistry
import com.whatsbird.vision.RegionClassifierRouter
import com.whatsbird.vision.TfLiteClassifier
import com.whatsbird.vision.YoloDetector
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Named
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun database(@ApplicationContext c: Context): AppDatabase =
        Room.databaseBuilder(c, AppDatabase::class.java, "whatsbird.db")
            .addMigrations(AppDatabase.MIGRATION_1_2, AppDatabase.MIGRATION_2_3)
            .build()

    @Provides
    fun observationDao(db: AppDatabase): ObservationDao = db.observationDao()

    @Provides
    @Singleton
    fun detector(@ApplicationContext c: Context): BirdDetector = YoloDetector(c)

    @Provides
    @Singleton
    @Named("huabei")
    fun huabeiClassifier(@ApplicationContext c: Context): BirdClassifier =
        TfLiteClassifier(
            c, ModelRegistry.HUABEI_ASSET, ModelRegistry.HUABEI_LABELS,
            ModelRegistry.HUABEI_DIM,
        )

    @Provides
    @Singleton
    @Named("huazhong")
    fun huazhongClassifier(@ApplicationContext c: Context): BirdClassifier =
        TfLiteClassifier(
            c, ModelRegistry.HUAZHONG_ASSET, ModelRegistry.HUAZHONG_LABELS,
            ModelRegistry.HUAZHONG_DIM,
        )

    @Provides
    @Singleton
    @Named("huanan_sw")
    fun huananSwClassifier(@ApplicationContext c: Context): BirdClassifier =
        TfLiteClassifier(
            c, ModelRegistry.HUANAN_SW_ASSET, ModelRegistry.HUANAN_SW_LABELS,
            ModelRegistry.HUANAN_SW_DIM,
        )

    @Provides
    @Singleton
    @Named("huadong")
    fun huadongClassifier(@ApplicationContext c: Context): BirdClassifier =
        TfLiteClassifier(
            c, ModelRegistry.HUADONG_ASSET, ModelRegistry.HUADONG_LABELS,
            ModelRegistry.HUADONG_DIM,
        )

    @Provides
    @Singleton
    @Named("dongbei")
    fun dongbeiClassifier(@ApplicationContext c: Context): BirdClassifier =
        TfLiteClassifier(
            c, ModelRegistry.DONGBEI_ASSET, ModelRegistry.DONGBEI_LABELS,
            ModelRegistry.DONGBEI_DIM,
        )

    @Provides
    @Singleton
    @Named("reject")
    fun rejectClassifier(@ApplicationContext c: Context): BirdClassifier =
        TfLiteClassifier(
            c, ModelRegistry.REJECT_ASSET, ModelRegistry.REJECT_LABELS,
            ModelRegistry.REJECT_DIM,
        )

    @Provides
    @Singleton
    @Named("cn373")
    fun cn373Classifier(@ApplicationContext c: Context): BirdClassifier =
        TfLiteClassifier(
            c, ModelRegistry.CN373_ASSET, ModelRegistry.CN373_LABELS,
            numSpecies = ModelRegistry.CN373_DIM,
            tensorDim = ModelRegistry.CN373_TENSOR_DIM,
        )
}

@Module
@InstallIn(SingletonComponent::class)
abstract class RouterModule {
    @Binds
    @Singleton
    abstract fun classifier(impl: RegionClassifierRouter): BirdClassifier
}

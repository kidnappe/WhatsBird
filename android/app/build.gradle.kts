import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
    id("com.google.dagger.hilt.android")
}

android {
    namespace = "com.whatsbird"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.whatsbird"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    // 正式签名：只有 android/keystore.properties 存在时才启用（该文件与 .jks 都不入库，
    // 见根 .gitignore）。公开仓库里没有这个文件，外部贡献者 assembleRelease 得到的是
    // 未签名包——这是预期行为，不是 bug。
    val keystorePropsFile = rootProject.file("keystore.properties")
    val hasReleaseKey = keystorePropsFile.exists()
    val keystoreProps = Properties().apply {
        if (hasReleaseKey) keystorePropsFile.inputStream().use { load(it) }
    }

    signingConfigs {
        if (hasReleaseKey) {
            create("release") {
                storeFile = file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (hasReleaseKey) signingConfig = signingConfigs.getByName("release")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")

            // 正式包只出 arm64-v8a（2026-10-11 v1.0 起）。TFLite 的 native 库按 ABI 各来一份：
            // x86/x86_64 只有模拟器用得上、armeabi-v7a 只剩 2016 年前的老机——都不带，
            // 包体从 129.9MB 降到 118.9MB。**只加在 release 上**：debug 保留全部 ABI，
            // 免得模拟器装不上。
            ndk {
                abiFilters += "arm64-v8a"
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    // material3 1.4+/1.5-alpha（JCA 原版 ToggleButton 所在通道）要求 compileSdk 36/37，
    // 暂不跟进；变焦挡位用 FilterChip 按 JCA 视觉重写（见 ZoomBar.kt 注释）
    val composeBom = platform("androidx.compose:compose-bom:2024.10.01")
    implementation(composeBom)
    implementation("androidx.compose.material3:material3")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    debugImplementation("androidx.compose.ui:ui-tooling")

    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // 图像模型走 LiteRT(TFLite)；geomodel 将来走 ONNX Runtime（见 PLAN §7）
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    val room = "2.6.1"
    implementation("androidx.room:room-runtime:$room")
    implementation("androidx.room:room-ktx:$room")
    ksp("androidx.room:room-compiler:$room")

    implementation("com.google.dagger:hilt-android:2.52")
    ksp("com.google.dagger:hilt-compiler:2.52")
    implementation("androidx.hilt:hilt-navigation-compose:1.2.0")

    // 种子色持久化
    implementation("androidx.datastore:datastore-preferences:1.1.1")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}

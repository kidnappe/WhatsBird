package com.whatsbird

import android.app.Application
import com.whatsbird.ui.ProtectionCatalog
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class WhatsBirdApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // 保护状态索引（assets/guide/protection.json，~17 KB）后台预热。
        // 取景页 / 日志页 / 全屏看图都要在种名后挂保护徽章，若留到各页首次组合时再读盘，
        // 就变成主线程做 assets IO——与 APP-CODE-REVIEW-20261010 记的图鉴首访问题同类。
        // 预热后各页走 ProtectionCatalog 的进程级缓存，纯内存命中。
        Thread {
            ProtectionCatalog.load(this)
        }.apply {
            name = "protection-warmup"
            isDaemon = true
            priority = Thread.MIN_PRIORITY
        }.start()
    }
}

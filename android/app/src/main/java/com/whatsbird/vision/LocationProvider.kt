package com.whatsbird.vision

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationManager
import android.os.Handler
import android.os.Looper
import dagger.hilt.android.qualifiers.ApplicationContext
import java.util.concurrent.atomic.AtomicBoolean
import javax.inject.Inject
import javax.inject.Singleton

/**
 * 位置源（两级 + 会话缓存）：
 *
 *  · [requestFresh]：**主动**取一次新坐标（权限弹窗授权后调用；数秒内回返，
 *    超时自动回退缓存位置）。重排先验与记录坐标都吃它——被动缓存可能停留在
 *    几小时前，先验的"最近采样点"需要新鲜坐标。
 *  · [lastKnown]：优先返回本次会话已取到的新坐标；没有则**被动**读系统缓存的
 *    最后已知位置（其他 App/系统留下的，零耗电零等待，但可能陈旧）。
 *
 *  两者都需要位置权限（MainActivity 启动时弹窗授权）。观鸟一次通常位置不挪窝，
 *  新坐标缓存整个会话；每次进相机（CameraViewModel init）刷新一次，耗电可忽略。
 */
@Singleton
class LocationProvider @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
    private val main = Handler(Looper.getMainLooper())

    @Volatile private var sessionFix: Location? = null
    private val requesting = AtomicBoolean(false)

    fun lastKnown(): Location? = sessionFix ?: readSystemCache()

    /**
     * 主动取一次新坐标（回调在主线程）。并发调用只允许一个在途；
     * [timeoutMs] 内没结果就回退 lastKnown()，晚到的真实结果仍会更新会话缓存。
     */
    @SuppressLint("MissingPermission") // 调用前应已授予 COARSE；无授权时安全兜底
    fun requestFresh(timeoutMs: Long = 8_000, onDone: (Location?) -> Unit) {
        val mgr = lm ?: run { onDone(lastKnown()); return }
        if (!requesting.compareAndSet(false, true)) { onDone(lastKnown()); return }
        val delivered = AtomicBoolean(false)
        fun deliver(loc: Location?) {
            if (delivered.compareAndSet(false, true)) {
                requesting.set(false)
                if (loc != null) sessionFix = loc
                onDone(lastKnown())
            }
        }
        val provider = when {
            mgr.isProviderEnabled(LocationManager.GPS_PROVIDER) -> LocationManager.GPS_PROVIDER
            mgr.isProviderEnabled(LocationManager.NETWORK_PROVIDER) -> LocationManager.NETWORK_PROVIDER
            else -> { deliver(readSystemCache()); return }
        }
        // 一次性监听（经典写法，兼容全 API）：首个更新即取即撤；超时回退缓存
        val listener = object : android.location.LocationListener {
            override fun onLocationChanged(location: Location) {
                try { mgr.removeUpdates(this) } catch (_: Exception) {}
                deliver(location)
            }
        }
        try {
            mgr.requestLocationUpdates(provider, 0L, 0f, listener, Looper.getMainLooper())
        } catch (t: Throwable) {
            deliver(readSystemCache()); return
        }
        main.postDelayed({
            try { mgr.removeUpdates(listener) } catch (_: Exception) {}
            deliver(readSystemCache())
        }, timeoutMs)
    }

    private fun readSystemCache(): Location? = try {
        lm?.getProviders(true)?.asSequence()
            ?.mapNotNull { lm.getLastKnownLocation(it) }
            ?.maxByOrNull { it.time }
    } catch (_: SecurityException) {
        null
    }
}

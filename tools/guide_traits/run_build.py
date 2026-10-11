# -*- coding: utf-8 -*-
"""一键构建：生成图鉴性状资产 → 自检不变量 → 重建 debug APK。

bat 只负责启动本脚本（见同目录 start_build.bat）。加 --no-apk 跳过 gradle。

不捕获子进程输出（stdio 继承），以免在受限环境下因命名管道被拒。
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANDROID = ROOT / "android"
APK = ANDROID / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
GEN = HERE / "build_traits_asset.py"


def run(cmd, cwd=None):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    try:
        return subprocess.run(cmd, cwd=cwd).returncode
    except FileNotFoundError as e:
        print(f"[FAIL] 找不到可执行文件: {e}")
        return 127


def main():
    no_apk = "--no-apk" in sys.argv

    print("=" * 52)
    print(" WhatsBird 图鉴性状资产构建")
    print("=" * 52)

    print("\n[1/3] 生成资产 + 匹配报告 ...")
    if run([sys.executable, str(GEN)]) != 0:
        print("\n[FAIL] 生成失败")
        return 1

    print("\n[2/3] 自检不变量（区间有序、卵长≥卵宽）...")
    if run([sys.executable, str(GEN), "--verify"]) != 0:
        print("\n[FAIL] 自检未通过——资产已写出但有异常，先别装机")
        return 1

    if no_apk:
        print("\n[3/3] 跳过 gradle（--no-apk）")
    else:
        print("\n[3/3] 重建 debug APK ...")
        gradlew = ANDROID / ("gradlew.bat" if os.name == "nt" else "gradlew")
        if run([str(gradlew), "assembleDebug", "--console=plain"], cwd=str(ANDROID)) != 0:
            print("\n[FAIL] gradle 失败（性状资产本身已生成成功）")
            return 1

    print("\n[OK] 完成")
    if APK.exists():
        print(f"   APK  : {APK}")
        print(f'   装机 : adb install -r "{APK}"')
    print("   文档 : docs/handoff/root/GUIDE-TRAITS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

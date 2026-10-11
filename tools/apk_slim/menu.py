"""tools/apk_slim 的菜单入口——bat 只做启动器，逻辑都在这里（cmd 里写 if/for 块是雷区）。

bat 里放中文要 GBK + CRLF、放菜单要延迟展开与两遍解析，两个坑都不如直接开 Python。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE = os.path.join(ROOT, r"dist\apk_slim\staging")
APK = os.path.join(ROOT, r"android\app\build\outputs\apk\release\app-release.apk")

MENU = """
============================================================
  APK slim toolkit  -  fp16 weights for the 7 classifiers
  read android\\models\\README.md first if you have not
============================================================

  1) build candidates only      fp16 + CAST -> dist\\apk_slim\\staging
  2) quick accuracy check      n=300 per model   (~2 min)
  3) full accuracy check       n=1500 per model  (~15 min)
  4) load smoke test           dtype / shape / invoke
  5) install into assets + build release APK
  0) quit
"""

ACTIONS = {
    "1": [sys.executable, r"tools\apk_slim\eval_fp16_vs_fp32.py", "--build-only"],
    "2": [sys.executable, r"tools\apk_slim\eval_fp16_vs_fp32.py", "--n", "300"],
    "3": [sys.executable, r"tools\apk_slim\eval_fp16_vs_fp32.py", "--n", "1500", "--seed", "7"],
    "4": [sys.executable, r"tools\apk_slim\smoke_load_models.py", STAGE],
    "5": [sys.executable, r"tools\copy_models_to_android.py"],
}


def main():
    os.chdir(ROOT)
    while True:
        print(MENU)
        try:
            c = input("Choose: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if c == "0":
            return 0
        if c not in ACTIONS:
            print("unknown choice\n")
            continue
        if c == "4" and not os.path.isdir(STAGE):
            print("staging is empty, building candidates first...\n")
            subprocess.run(ACTIONS["1"], check=False)
        rc = subprocess.run(ACTIONS[c], check=False).returncode
        if c == "5" and rc == 0:
            rc = subprocess.run(
                ["gradlew.bat", ":app:assembleRelease", "--offline"],
                cwd=os.path.join(ROOT, "android"), check=False).returncode
            if rc == 0 and os.path.exists(APK):
                print(f"\nAPK: {APK}\nsize: {os.path.getsize(APK) / 1048576:.2f} MB")
        print()


if __name__ == "__main__":
    sys.exit(main())
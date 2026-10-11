@echo off
chcp 65001 >nul
setlocal EnableExtensions
title WhatsBird guide traits builder
cd /d "%~dp0"

rem ============================================================
rem  WhatsBird - guide detail traits asset builder
rem
rem  in  : tools\dataset_traits\china_bird_traits.csv
rem        tools\guide_traits\rename_overrides.csv
rem        android\app\src\main\assets\models\labels_national_v2.txt
rem  out : android\app\src\main\assets\guide\traits.json
rem        tools\guide_traits\match_report.csv
rem
rem  Rebuilds the debug APK too. Pass --no-apk to skip gradle.
rem  All logic is in run_build.py; this file only launches it.
rem ============================================================

set PYTHONIOENCODING=utf-8

python "%~dp0run_build.py" %*
echo.
pause

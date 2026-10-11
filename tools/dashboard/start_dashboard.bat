@echo off
chcp 65001 >nul
title ObserveBird training dashboard

rem ============================================
rem  ObserveBird training dashboard launcher
rem  Usage:
rem    start_dashboard.bat                      -> default model dir, port 8787
rem    start_dashboard.bat "E:\path\to\out"     -> custom model dir
rem    start_dashboard.bat "E:\path\to\out" 8788 -> custom dir + port
rem  Set NO_BROWSER=1 to skip auto-opening the page.
rem ============================================

set REPO=E:\code\observebird
set PORT=8787
set OUT=E:\code\code tools\models\huabei_mnv4_300

if not "%~1"=="" set OUT=%~1
if not "%~2"=="" set PORT=%~2

set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com

set DL_DIR=E:\code\code tools\data
set DL_DATASET=E:\code\code tools\data\huabei\dataset.csv
set DL_TARGET=29650
set DL_BASELINE=12561

echo ================================================
echo  ObserveBird training dashboard
echo  model dir : %OUT%
echo  url       : http://127.0.0.1:%PORT%
echo  (Ctrl+C to stop, close window to exit)
echo ================================================

if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%

cd /d %REPO%
python -u tools\dashboard\train_dashboard.py --out "%OUT%" --port %PORT% ^
  --download-dir "%DL_DIR%" --download-dataset "%DL_DATASET%" ^
  --download-target %DL_TARGET% --download-baseline %DL_BASELINE%
echo.
echo dashboard exited (port busy or python missing?).
pause

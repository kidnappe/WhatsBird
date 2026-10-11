@echo off
chcp 65001 >nul
title ObserveBird shiji test viewer

rem ============================================
rem  real-device test image viewer
rem  Big slides + ground-truth labels for pointing
rem  the phone camera at the screen.
rem  Usage:
rem    start_img_view.bat          -> http://127.0.0.1:8789
rem    start_img_view.bat 8791     -> custom port
rem  Resample a new random batch (10 per region):
rem    python tools\img_view\img_view.py --resample 10
rem  Close this window to stop the server.
rem ============================================

set PORT=8789
if not "%~1"=="" set PORT=%~1

set PYTHONIOENCODING=utf-8

echo ================================================
echo  ObserveBird real-device test viewer
echo  url : http://127.0.0.1:%PORT%
echo  keys: Left/Right = prev/next   H = hide/show label
echo ================================================

rem already running -> do not start a second instance, just open the page
curl -s -o nul --max-time 2 http://127.0.0.1:%PORT%/api/list
if not errorlevel 1 (
    echo already running - opening browser only.
    if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%
    exit /b 0
)

if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%

cd /d E:\code\observebird
python -u tools\img_view\img_view.py --port %PORT%
echo.
echo viewer exited (port busy or python missing?).
pause

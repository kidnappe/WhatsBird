@echo off
chcp 65001 >nul
title ObserveBird B detector training monitor (read-only)

rem ============================================
rem  Read-only training monitor for this pack.
rem  Usage:
rem    start_monitor.bat          -> http://127.0.0.1:8787
rem    start_monitor.bat 8790     -> custom port
rem  Set NO_BROWSER=1 to skip auto-opening the page.
rem  Close this window to stop the monitor.
rem  It never touches the training process (read-only).
rem ============================================

set PORT=8787
if not "%~1"=="" set PORT=%~1

set PYTHONIOENCODING=utf-8

echo ================================================
echo  ObserveBird B detector monitor (read-only)
echo  root : %~dp0
echo  url  : http://127.0.0.1:%PORT%
echo  Close this window to stop the monitor.
echo ================================================

rem already running -> do not start a second instance, just open the page
curl -s -o nul --max-time 2 http://127.0.0.1:%PORT%/api/status
if not errorlevel 1 (
    echo already running - opening browser only.
    if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%
    exit /b 0
)

if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%

cd /d %~dp0
python -u detb_pack_monitor.py --root "%~dp0." --port %PORT%
echo.
echo monitor exited (port busy or python missing?).
pause

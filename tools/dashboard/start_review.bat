@echo off
chcp 65001 >nul
title ObserveBird review UI (detector B)

rem ============================================
rem  pseudo-label review UI launcher
rem  Usage:
rem    start_review.bat            -> http://127.0.0.1:8788
rem    start_review.bat 8790       -> custom port
rem  Set NO_BROWSER=1 to skip auto-opening the page.
rem  Close this window to stop the server.
rem ============================================

set PORT=8788
if not "%~1"=="" set PORT=%~1

set PYTHONIOENCODING=utf-8

echo ================================================
echo  ObserveBird review UI (detector B pseudo-labels)
echo  url : http://127.0.0.1:%PORT%
echo  keys: Space=pass  X=wrong-box  B=loose-box  G=bad-image
echo        Z=missed-bird  U=undo  Left=prev  (hint bar on page)
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
python -u tools\dashboard\review_ui.py --port %PORT%
echo.
echo review ui exited (port busy or python missing?).
pause

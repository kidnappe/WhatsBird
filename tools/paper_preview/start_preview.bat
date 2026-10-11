@echo off
chcp 65001 >nul
title ObserveBird paper-guide design preview

rem ============================================
rem  paper-style HTML design preview launcher
rem  Usage:
rem    start_preview.bat            -> http://127.0.0.1:8790/motion-v6-preview.html
rem                                (v6 motion harvest demo; layer-map.html + guide-preview.html also served)
rem  Set NO_BROWSER=1 to skip auto-opening the page.
rem  Close this window to stop the server.
rem ============================================

set PORT=8790

echo ================================================
echo  ObserveBird paper-style preview (design only)
echo  url : http://127.0.0.1:%PORT%/motion-v6-preview.html   (v6 motion harvest demo)
echo         http://127.0.0.1:%PORT%/motion-07-in-app.html      (item 7 on the REAL full app UI)
echo         http://127.0.0.1:%PORT%/motion-07-indicator.html  (item 7 only: 3-layer map)
echo         http://127.0.0.1:%PORT%/layer-map.html
echo         http://127.0.0.1:%PORT%/guide-preview.html
echo  file: docs\design\preview-paper\
echo ================================================

rem already running -> do not start a second instance, just open the page
curl -s -o nul --max-time 2 http://127.0.0.1:%PORT%/guide-preview.html
if not errorlevel 1 (
    echo already running - opening browser only.
    if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%/motion-v6-preview.html
    exit /b 0
)

if not defined NO_BROWSER start "" http://127.0.0.1:%PORT%/motion-v6-preview.html

cd /d E:\code\observebird\docs\design\preview-paper
python -m http.server %PORT% --bind 127.0.0.1
echo.
echo preview server exited (port busy or python missing?).
pause

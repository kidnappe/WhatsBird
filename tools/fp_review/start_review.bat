@echo off
chcp 65001 >nul
setlocal

rem ---------------------------------------------------------------
rem  E-1 true-FP human review UI  (see docs/handoff/root/PRIORITIES-20261009.md sec 11.5)
rem  Step 1 (if needed): python tools\fp_review\score_boxes.py
rem  Step 2 (this bat) : serve the review UI
rem  Port 8789 (8788 is taken by the pseudo-label review UI)
rem
rem  Keys: SPACE / RIGHT = bird (true positive)
rem        X  = not a bird (false positive)
rem        S  = unsure      LEFT = prev      U = undo
rem
rem  Usage:  start_review.bat              (sample 400)
rem          start_review.bat 600          (larger sample)
rem          start_review.bat --summary    (print current estimate only, no server)
rem ---------------------------------------------------------------

set PY=D:\software\Python312\python.exe
set DIR=%~dp0

if not exist "%PY%" (
    echo [ERROR] python not found: %PY%
    pause
    exit /b 1
)

rem Check scored material exists; if missing, tell the user to run scoring first.
if not exist "%DIR%..\..\data\fp_review\review_fp.json" (
    dir /b "%DIR%..\..\data\fp_review\scored_*.json" >nul 2>&1
    if errorlevel 1 (
        echo [MISS] No scored JSON found. Run first:
        echo        python tools\fp_review\score_boxes.py
        echo.
        pause
        exit /b 1
    )
)

set PYTHONIOENCODING=utf-8
"%PY%" -u "%DIR%serve_review.py" %*
set RC=%ERRORLEVEL%

echo.
if not %RC%==0 echo [EXIT] code %RC%
pause
exit /b %RC%
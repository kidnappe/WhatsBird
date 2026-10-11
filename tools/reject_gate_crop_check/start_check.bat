@echo off
chcp 65001 >nul
setlocal

rem ---------------------------------------------------------------
rem  Reject gate check at deployment crop ratio (BOX_EXPAND_RATIO=4.0)
rem  Tool:   tools\reject_gate_crop_check\check_reject_at_deploy_crop.py
rem  Report: docs\reports\APP-LAYER-AUDIT-20261010.md  section 2.2
rem  Local inference only. No download, no network.
rem
rem  Usage:  start_check.bat            (full 20000)
rem          start_check.bat 2000       (quick sample)
rem ---------------------------------------------------------------

set PY=D:\software\Python312\python.exe
set TOOL=%~dp0check_reject_at_deploy_crop.py

if not exist "%PY%" (
    echo [ERROR] python not found: %PY%
    pause
    exit /b 1
)

set N=20000
if not "%~1"=="" set N=%~1

echo [RUN] reject gate check, max-n=%N  (local GPU, no network)
set PYTHONIOENCODING=utf-8
"%PY%" -u "%TOOL%" --max-n %N% --batch 128
set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
    echo [DONE] result printed above. JSON landed in data\reject_crop_check\
) else (
    echo [FAIL] exit code %RC%
)
pause
exit /b %RC%
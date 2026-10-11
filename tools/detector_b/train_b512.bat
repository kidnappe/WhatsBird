@echo off
chcp 65001 >nul
title ObserveBird B detector training (yolox_bird_512)

rem ============================================
rem  B route detector training launcher
rem  Usage: train_b512.bat            (fresh warm start from yolox_s.pth, batch 8)
rem        train_b512.bat resume      (continue from YOLOX_outputs latest_ckpt.pth)
rem        train_b512.bat 4           (batch override; "resume 4" also works)
rem  Close window to stop training.
rem ============================================

set BATCH=8
set MODE=fresh
if /i "%~1"=="resume" (set MODE=resume) else if not "%~1"=="" set BATCH=%~1
if /i "%~2"=="resume" set MODE=resume
if not "%~2"=="" if /i not "%~2"=="resume" set BATCH=%~2

set PYTHONIOENCODING=utf-8

echo ================================================
echo  ObserveBird B detector training (yolox_bird_512)
echo  mode : %MODE%  batch: %BATCH%  fp16  50 epochs
echo  data : E:\code\code tools\data\detector_b\annotations
echo ================================================

cd /d "E:\code\code tools\yolox_repo"
rem yolox not develop-installed: repo root goes on PYTHONPATH (tools\train.py sys.path has only tools\)
set PYTHONPATH=E:\code\code tools\yolox_repo;%PYTHONPATH%
if "%MODE%"=="resume" (
    python -u tools\train.py -f E:\code\observebird\tools\detector_b\yolox_bird_exp.py --resume -b %BATCH% --fp16
) else (
    python -u tools\train.py -f E:\code\observebird\tools\detector_b\yolox_bird_exp.py -c "E:\code\code tools\detectors\yolox\yolox_s.pth" -b %BATCH% --fp16
)
echo.
echo training exited.
pause

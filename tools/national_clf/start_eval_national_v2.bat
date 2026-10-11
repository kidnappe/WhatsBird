@echo off
chcp 65001 >nul
title national v2 eval - head2head vs cn373 baseline
cd /d E:\code\observebird
echo [eval-v2] restores merged dataset, then runs v2 (GPU) and cn373 (CPU)
echo [eval-v2] on the same stratified val samples; long part is cn373 CPU pass
echo [eval-v2] progress/ETA prints to this window; cache allows resume.
echo [eval-v2] outputs: E:\code\code tools\models\national_v2_eval\summary.json
echo.
python -u tools\national_clf\eval_national_v2.py %*
echo.
echo [eval-v2] exited (code %errorlevel%). See summary.json.
pause

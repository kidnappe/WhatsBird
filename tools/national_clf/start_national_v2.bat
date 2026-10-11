@echo off
chcp 65001 >nul
title national v2 chain - fetch, merge, train
cd /d E:\code\observebird
echo [national v2] chain: gap fetch (141 species) - merge 148k - train 30 epochs
echo [national v2] dashboard: http://127.0.0.1:8787  (region: national)
echo [national v2] stdout log: E:\code\code tools\data\national\pipeline_v2_stdout.log
echo [national v2] milestone log: E:\code\code tools\data\national\pipeline.log
echo [national v2] window shows nothing after start - output goes to the log.
echo [national v2] close this window to STOP the chain.
echo.
python -u tools\national_clf\national_pipeline_v2.py >> "E:\code\code tools\data\national\pipeline_v2_stdout.log" 2>&1
echo.
echo [national v2] chain exited (exit code %errorlevel%). Check pipeline.log tail.
pause

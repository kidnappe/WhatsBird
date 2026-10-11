@echo off
chcp 65001 >nul
cd /d E:\code\observebird
if "%SILICONFLOW_API_KEY%"=="" (
  echo [ERROR] SILICONFLOW_API_KEY is not set.
  echo   1. Register at https://www.siliconflow.cn , copy your API key
  echo   2. Run:  set SILICONFLOW_API_KEY=sk-...
  echo      then re-run this bat.
  pause
  exit /b 1
)
echo Running VLM gap eval: 5 models x 150 images, output to
echo   E:\code\code tools\data\national\vlm_eval\
python -u tools\national_clf\eval_vlm_gap.py --n 150 --per 2
pause

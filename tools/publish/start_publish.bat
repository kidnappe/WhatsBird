@echo off
chcp 65001 >nul
setlocal
title WhatsBird - build public tree for GitHub

rem  One-click: assemble the public repo tree from a commit (whitelist based).
rem  Output goes to dist\public\  -- see AGENTS.md "public / internal two layers".
rem  Internal docs (handoff / reports / research / session / archive) are never
rem  exported, because this script whitelists directories instead of blacklisting files.
rem
rem  Options are passed through, e.g.:
rem    start_publish.bat -InitRepo          also git init + first commit (push-ready)
rem    start_publish.bat -Ref v1.0.0        export from a tag
rem    start_publish.bat -OutDir D:\pub     custom output directory

cd /d "%~dp0..\.."
echo.
echo  [WhatsBird] building public tree ...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish_public.ps1" %*
set RC=%ERRORLEVEL%

echo.
if not "%RC%"=="0" (
    echo  [FAILED] exit code %RC%
) else (
    echo  [DONE]
)
echo.
pause
endlocal

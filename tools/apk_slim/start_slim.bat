@echo off
setlocal EnableExtensions
cd /d "%~dp0..\.."
python tools\apk_slim\menu.py
if errorlevel 1 pause
endlocal
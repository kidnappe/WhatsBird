@echo off
chcp 65001 >nul
cd /d %~dp0..\..\
echo Generating offline admin-division geo index (DataV GeoAtlas)...
python tools\admin_geo\generate_admin_geo.py %*
echo.
echo Done. Output: android\app\src\main\assets\geo\admin_county.ndjson
pause

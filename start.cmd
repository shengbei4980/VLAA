@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist "%~dp0.venv\Scripts\python.exe" goto needs_install
if not exist "%~dp0frontend\node_modules\vite\bin\vite.js" goto needs_install
where node >nul 2>nul
if errorlevel 1 goto needs_install
"%~dp0.venv\Scripts\python.exe" "%~dp0run.py"
set "VLAA_START_EXIT=%errorlevel%"
if not "%VLAA_START_EXIT%"=="0" pause
exit /b %VLAA_START_EXIT%

:needs_install
echo 启动环境尚未安装完整。请先双击同一文件夹中的 install.cmd。
pause
exit /b 1

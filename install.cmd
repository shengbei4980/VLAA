@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "VLAA_SETUP_ARG="
if not "%~2"=="" goto usage
if "%~1"=="" goto find_python
if /i "%~1"=="--check" (
    set "VLAA_SETUP_ARG=--check"
    goto find_python
)
goto usage

:find_python
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 "%~dp0scripts\setup.py" %VLAA_SETUP_ARG%
set "VLAA_SETUP_EXIT=%errorlevel%"
goto finish

:use_python
where python >nul 2>nul
if errorlevel 1 goto missing_python
python "%~dp0scripts\setup.py" %VLAA_SETUP_ARG%
set "VLAA_SETUP_EXIT=%errorlevel%"
goto finish

:missing_python
echo 请先安装 Python 3.12，并在安装时勾选 Add python.exe to PATH。
set "VLAA_SETUP_EXIT=1"
goto finish

:usage
echo 使用方法 install.cmd 或 install.cmd --check
set "VLAA_SETUP_EXIT=2"

:finish
if not defined VLAA_SETUP_ARG pause
exit /b %VLAA_SETUP_EXIT%

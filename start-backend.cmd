@echo off
cd /d "%~dp0"
"%CD%\.venv\Scripts\python.exe" "%CD%\run.py"
if errorlevel 1 pause

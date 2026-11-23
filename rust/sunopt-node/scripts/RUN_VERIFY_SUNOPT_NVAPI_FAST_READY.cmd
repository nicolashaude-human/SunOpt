@echo off
setlocal
cd /d "%~dp0\..\..\.."
python rust\sunopt-node\scripts\verify-sunopt-nvapi-fast-ready.py
if errorlevel 1 exit /b 1
python rust\sunopt-node\scripts\test-sunopt-gpu-fast-ready-mock.py
exit /b %errorlevel%

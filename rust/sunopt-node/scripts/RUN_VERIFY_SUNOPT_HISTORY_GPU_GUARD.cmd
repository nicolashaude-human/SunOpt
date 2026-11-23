@echo off
setlocal
cd /d "%~dp0..\..\.."
node "rust\sunopt-node\scripts\verify-sunopt-history-gpu-guard.mjs"
exit /b %errorlevel%

@echo off
setlocal
cd /d "%~dp0..\..\.."
node "rust\sunopt-node\scripts\verify-sunopt-b14-documentation.mjs"
exit /b %errorlevel%

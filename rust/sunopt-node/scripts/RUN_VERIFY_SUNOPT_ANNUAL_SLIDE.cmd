@echo off
setlocal
cd /d "%~dp0\..\..\.."
node rust\sunopt-node\scripts\verify-sunopt-annual-slide.mjs
exit /b %errorlevel%

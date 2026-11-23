@echo off
setlocal
cd /d "%~dp0\..\..\.."
node "rust\sunopt-node\scripts\build-sunopt-standalone-presentation.mjs" %*
exit /b %errorlevel%

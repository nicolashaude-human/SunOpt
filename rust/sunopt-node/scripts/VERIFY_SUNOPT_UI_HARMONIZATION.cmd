@echo off
setlocal
cd /d "%~dp0..\..\.."
node rust\sunopt-node\scripts\verify-sunopt-ui-harmonization.mjs
exit /b %ERRORLEVEL%

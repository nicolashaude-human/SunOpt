@echo off
setlocal EnableExtensions
cd /d "%~dp0..\..\.."
if errorlevel 1 exit /b 1
node rust\sunopt-node\scripts\approach-a9-bridge-patch.mjs --restore
exit /b %errorlevel%

@echo off
setlocal
cd /d "%~dp0\..\..\.."
if errorlevel 1 exit /b 1
if not exist "node_modules\.bin\tsx.cmd" (
  echo [SunOpt] ERROR: tsx fehlt. Im Projekt npm install ausfuehren.
  exit /b 1
)
node_modules\.bin\tsx.cmd rust\sunopt-node\scripts\capture-sunopt-annual-slide.ts %*
exit /b %errorlevel%

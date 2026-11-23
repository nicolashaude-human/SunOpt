@echo off
setlocal
cd /d "%~dp0\..\..\.."
node rust\sunopt-node\scripts\verify-sunopt-static-navigation.mjs
if errorlevel 1 (
  echo SunOpt static navigation verification failed.
  exit /b 1
)
echo SunOpt static navigation verification PASS.

@echo off
setlocal
rem SunOpt UI inventory (read-only). Run from any working directory.
pushd "%~dp0\..\..\.."
if errorlevel 1 (
  echo Could not locate project root.
  exit /b 1
)
node "rust\sunopt-node\scripts\scan-sunopt-ui-history.mjs"
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" echo UI scan failed with exit code %RESULT%.
exit /b %RESULT%

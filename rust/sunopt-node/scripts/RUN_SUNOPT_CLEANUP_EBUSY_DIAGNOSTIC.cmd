@echo off
setlocal
cd /d "%~dp0..\..\.."
node "rust\sunopt-node\scripts\cleanup-ebusy-diagnostic.mjs"
if errorlevel 1 (
  echo DIAGNOSE FEHLGESCHLAGEN. 1>&2
  exit /b 1
)
endlocal

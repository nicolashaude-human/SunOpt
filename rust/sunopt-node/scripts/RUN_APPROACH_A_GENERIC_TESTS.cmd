@echo off
setlocal
set "SUNOPT_ROOT=%~dp0..\..\.."
cd /d "%SUNOPT_ROOT%"
if errorlevel 1 exit /b 2
echo === SunOpt Ansatz A: generischer Compiler-Pruefstand (G1-G8) ===
node "rust\sunopt-node\scripts\approach-a-generic-verify.mjs" --require-gpu --arch=sm_120
set "RESULT=%ERRORLEVEL%"
echo Bericht: "%CD%\updates\outbox\approach-a-generic-verify-*.json"
echo PRODUKTIONSPFAD UNVERAENDERT. Pruefung: %RESULT%
pause
exit /b %RESULT%

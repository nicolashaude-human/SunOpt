@echo off
setlocal
set "SUNOPT_ROOT=%~dp0..\..\.."
cd /d "%SUNOPT_ROOT%"
if errorlevel 1 exit /b 2
echo === SunOpt Ansatz A: generischer struktureller Compiler ===
echo Stufen: A1-Cache, TypeScript-zu-C++/CUDA, Paritaet, Original-A1+A2-Vergleich
echo Produktionspfad wird NICHT veraendert.
node "rust\sunopt-node\scripts\approach-a-structured-verify.mjs" --require-gpu --require-original-a --arch=sm_120
set "RESULT=%ERRORLEVEL%"
echo Bericht: "%CD%\updates\outbox\approach-a-structured-verify-*.json"
echo Produktionsfreigabe bleibt bis 28-Tank-Jahres-Paritaet gesperrt.
pause
exit /b %RESULT%

@echo off
setlocal
set "SUNOPT_ROOT=%~dp0..\..\.."
cd /d "%SUNOPT_ROOT%"
if errorlevel 1 exit /b 2
echo =============================================================
echo SunOpt Approach A - Ein Gesamttest
 echo (1) Generic CUDA + Paritaet + Original A1/A2 Benchmark
 echo (2) 28-Tank-Jahr: EXISTIERENDE Produktionspfade A und B
 echo (3) Freigabepruefung: generischer A-Jahreslauf separat gesperrt
 echo.
echo ACHTUNG: CUDA-Test und kompletter Jahreslauf koennen dauern.
echo Produktionsdateien werden nicht geaendert.
echo =============================================================
node "rust\sunopt-node\scripts\approach-a-one-test.mjs"
set "RESULT=%ERRORLEVEL%"
echo.
echo Gesamtergebnis unter updates\outbox\approach-a-one-test-*.json
if not "%RESULT%"=="0" echo Mindestens eine Teststufe fehlgeschlagen.
pause
exit /b %RESULT%

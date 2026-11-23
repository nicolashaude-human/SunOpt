@echo off
setlocal EnableExtensions
cd /d "%~dp0..\..\.."
if errorlevel 1 exit /b 1

echo ===== SunOpt A9: generischer CUDA-Quellcode + echter Jahrescontroller =====
echo [1/3] TypeScript f(x) -^> generisches CUDA, sm_120
node rust\sunopt-node\scripts\approach-a9-build.mjs --arch=sm_120
if errorlevel 1 exit /b 1

echo [2/3] A9 CUDA-Paritaet, 28 Tanks, 128 serielle Schritte
node rust\sunopt-node\scripts\approach-a9-verify.mjs --year-length
if errorlevel 1 exit /b 1

echo [3/3] Vollstaendiger produktiver Jahrescontroller: Original A / generisches A9
node rust\sunopt-node\scripts\approach-a9-annual-test.mjs
exit /b %errorlevel%

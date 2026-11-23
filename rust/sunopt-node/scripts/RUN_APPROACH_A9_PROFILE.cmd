@echo off
setlocal EnableExtensions
cd /d "%~dp0..\..\.."
if errorlevel 1 exit /b 1
echo ===== SunOpt A9: CUDA profiler and annual parity, research only =====
echo [1/3] Generate source-derived CUDA executable with phase timing
node rust\sunopt-node\scripts\approach-a9-build.mjs --arch=sm_120 --profile
if errorlevel 1 exit /b 1
echo [2/3] GPU stateful parity with POSITIVE reserve/request fixtures
node rust\sunopt-node\scripts\approach-a9-verify.mjs --year-length
if errorlevel 1 exit /b 1
echo [3/3] One isolated annual run per path, CPU vs CUDA phase breakdown
node rust\sunopt-node\scripts\approach-a9-annual-test.mjs --diagnostic
exit /b %errorlevel%

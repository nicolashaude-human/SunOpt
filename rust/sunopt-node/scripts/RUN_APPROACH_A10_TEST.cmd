@echo off
setlocal
cd /d "%~dp0..\..\.."
if errorlevel 1 exit /b 1
set "A10_SCRIPTS=rust\sunopt-node\scripts"
echo [A10] Compile baseline and generic integer-remainder CUDA, then compare 256 identical steps.
node "%A10_SCRIPTS%\approach-a10-intrem-benchmark.mjs" --arch=sm_120
if errorlevel 1 (echo [A10] STOP: comparison failed or GPU fastpath not faster & exit /b 1)
echo [A10] Verify source-generated GPU results with independent TypeScript reference.
node "%A10_SCRIPTS%\approach-a9-verify.mjs" --year-length
if errorlevel 1 (echo [A10] STOP: GPU reference parity failed & exit /b 1)
echo [A10] Real 28-tank, 35040-step stateful annual A vs A10 run (production restored in finally).
node "%A10_SCRIPTS%\approach-a9-annual-test.mjs" --diagnostic
if errorlevel 1 (echo [A10] STOP: annual parity or restore failed & exit /b 1)
echo [A10] Research complete. Production remains unchanged; inspect outbox JSON before deployment.
exit /b 0

@echo off
setlocal
cd /d "%~dp0..\..\.."
if defined SUNOPT_GPU_GUARD_PYTHON (
  "%SUNOPT_GPU_GUARD_PYTHON%" "rust\sunopt-node\scripts\probe-sunopt-gpu-profile.py"
) else (
  python "rust\sunopt-node\scripts\probe-sunopt-gpu-profile.py"
)
exit /b %errorlevel%

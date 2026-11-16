@echo off
setlocal
set "TARGET=C:\NextJS\SunOpt\sunopt\updates\outbox"

if not exist "%TARGET%\" (
    echo Ordner nicht gefunden: %TARGET%
    pause
    exit /b 1
)

cd /d "%TARGET%" || exit /b 1


for %%F in (*) do if /i not "%%F"=="%~nx0" del /f /q "%%F"
for /d %%D in (*) do rd /s /q "%%D"



$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\NextJS\SunOpt\sunopt"
$Tools = Join-Path $ProjectRoot "tools"
$Target = Join-Path $Tools "update_watcher.py"
$Source = Join-Path $PSScriptRoot "update_watcher.py"

if (-not (Test-Path $Source)) {
    throw "Missing source file: $Source"
}

if (-not (Test-Path $Tools)) {
    throw "SunOpt tools directory not found: $Tools"
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Backup = Join-Path $Tools "update_watcher.before-outbox-$Stamp.py"

if (Test-Path $Target) {
    Copy-Item $Target $Backup -Force
    Write-Host "Backup: $Backup"
}

Copy-Item $Source $Target -Force

& python $Target --status
if ($LASTEXITCODE -ne 0) {
    throw "New watcher failed --status."
}

Write-Host ""
Write-Host "Outbox watcher installed."
Write-Host "Outbox: $ProjectRoot\updates\outbox\llm.txt"

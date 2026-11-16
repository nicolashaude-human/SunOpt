param(
    [int]$Gpu = 0,
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $scriptDir "..")).Path

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $root "llm.txt"
}

Write-Host "SunOpt NVAPI Blackwell read-only probe" -ForegroundColor Cyan
Write-Host "GPU index: $Gpu"
Write-Host "Output:    $OutputPath"
Write-Host ""
Write-Host "No NVAPI write/tuning function will be called." -ForegroundColor Yellow

& python `
    (Join-Path $scriptDir "nvapi-blackwell-probe.py") `
    --gpu $Gpu `
    --output $OutputPath

$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "NVAPI probe failed with exit code $exitCode. Partial report is preserved in $OutputPath"
}

Write-Host ""
Write-Host "Probe completed. Upload llm.txt." -ForegroundColor Green

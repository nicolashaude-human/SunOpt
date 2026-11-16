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

Write-Host "SunOpt NVAPI controlled Blackwell write/readback/restore test" -ForegroundColor Cyan
Write-Host ""
Write-Host "This test will temporarily:" -ForegroundColor Yellow
Write-Host "  - lock the EXISTING 1050 mV / 2865 MHz V/F point"
Write-Host "  - set both detected fans to 70% manual"
Write-Host "  - run one existing SunOpt replay if available"
Write-Host "  - restore the exact original controls in finally"
Write-Host ""
Write-Host "It does NOT change V/F offsets, power limit, memory clock, or voltage boost."
Write-Host ""

& python `
    (Join-Path $scriptDir "nvapi-blackwell-write-test.py") `
    --gpu $Gpu `
    --output $OutputPath

$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "Controlled NVAPI write test failed with exit code $exitCode. Review the appended llm.txt block."
}

Write-Host ""
Write-Host "Controlled write test completed. Upload llm.txt." -ForegroundColor Green

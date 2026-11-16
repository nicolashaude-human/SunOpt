# rust/sunopt-node/scripts/run-blackwell-b14m-fp32-fastmath.ps1
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$NativeRoot = Join-Path $ProjectRoot "rust\sunopt-node"
$Runner = Join-Path $NativeRoot "scripts\blackwell-b14m-precision.mjs"
$Outbox = Join-Path $ProjectRoot "updates\outbox"
$Reference = Join-Path $Outbox "blackwell-b14m-strict-reference.bin"
$SweepReport = Join-Path $Outbox "blackwell-b14m-serial-sweep.txt"

New-Item -ItemType Directory -Force -Path $Outbox | Out-Null

function Invoke-Build([string]$FastMath, [string]$Label) {
    $env:SUNOPT_CUDA_FAST_MATH = $FastMath
    Push-Location $NativeRoot
    try {
        & npm run build 2>&1 |
            Tee-Object -FilePath (Join-Path $Outbox "b14m-build-$Label.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "B14M $Label build failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-Run([string]$Mode, [bool]$WriteReference) {
    Push-Location $ProjectRoot
    try {
        $write = if ($WriteReference) { "true" } else { "false" }
        & node $Runner "--mode" $Mode "--reference" $Reference "--write-reference" $write 2>&1 |
            Tee-Object -FilePath (Join-Path $Outbox "b14m-run-$Mode.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "B14M $Mode run failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$oldFastMath = $env:SUNOPT_CUDA_FAST_MATH

try {
    "=== B14M SERIAL FP32 / FASTMATH SWEEP ===" |
        Set-Content -Encoding UTF8 $SweepReport

    "STRICT BUILD" |
        Add-Content -Encoding UTF8 $SweepReport

    Invoke-Build "0" "strict"
    Invoke-Run "strict" $true

    "FASTMATH BUILD" |
        Add-Content -Encoding UTF8 $SweepReport

    Invoke-Build "1" "fastmath"
    Invoke-Run "fastmath" $false

    "RESTORE STRICT BUILD" |
        Add-Content -Encoding UTF8 $SweepReport

    Invoke-Build "0" "restore-strict"

    "B14M_SERIAL_SWEEP_RESULT=OK" |
        Add-Content -Encoding UTF8 $SweepReport

    Write-Host ""
    Write-Host "B14M_SERIAL_SWEEP_RESULT=OK"
    Write-Host "strict report:  updates\outbox\blackwell-b14m-strict.json"
    Write-Host "fastmath report: updates\outbox\blackwell-b14m-fastmath.json"
    Write-Host "llm:            updates\outbox\llm.txt"
}
finally {
    if ($null -eq $oldFastMath) {
        Remove-Item Env:SUNOPT_CUDA_FAST_MATH -ErrorAction SilentlyContinue
    }
    else {
        $env:SUNOPT_CUDA_FAST_MATH = $oldFastMath
    }
}

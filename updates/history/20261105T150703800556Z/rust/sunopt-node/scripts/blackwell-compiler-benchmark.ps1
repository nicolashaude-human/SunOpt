param(
    [int]$Repetitions = 3,
    [int]$DeviceId = 0,
    [string]$SnapshotPath = "",
    [string]$OutputPath = "",
    [switch]$SkipLaunchBounds
)
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
Set-Location $root
if ([string]::IsNullOrWhiteSpace($SnapshotPath)) { $SnapshotPath = Join-Path $root "blackwell-benchmark.bin" }
if ([string]::IsNullOrWhiteSpace($OutputPath)) { $OutputPath = Join-Path $root "llm.txt" }
$resultRoot = Join-Path $root "blackwell-benchmark-results"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
if (-not (Test-Path $SnapshotPath)) {
    Write-Host "Blackwell replay snapshot not found: $SnapshotPath" -ForegroundColor Yellow
    Write-Host "Restart Next.js and run the normal 28-tank benchmark once; Rust creates the snapshot automatically."
    exit 2
}
$originalArch = $env:SUNOPT_CUDA_ARCH
$originalNvccAppend = $env:NVCC_APPEND_FLAGS
$originalErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$results = @()
$report = New-Object System.Collections.Generic.List[string]
function Add-ReportLine {
    param(
        [AllowEmptyString()]
        [string]$Text = ""
    )

    [void]$script:report.Add($Text)
}
function Invoke-CapturedCommand {
    param(
        [scriptblock]$Command
    )

    try {
        return (
            & $Command 2>&1 |
            Out-String
        ).TrimEnd()
    } catch {
        return "ERROR: $($_.Exception.Message)"
    }
}
function Invoke-ForceBuild {
    & cargo clean -p sunopt-node *> $null
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "cargo clean failed with exit code $exitCode"
    }
}
function Get-CausalCalls([string]$p) {
    $inside=$false; $o=@()
    foreach($line in Get-Content $p) {
        if($line -match "Function\s+:\s+_Z19causal_batch_kernel") { $inside=$true; continue }
        if($inside -and $line -match "Function\s+:") { break }
        if($inside -and $line -match "CALL\.REL") { $o += $line.Trim() }
    }
    return $o
}
function Invoke-Variant([string]$label,[string]$arch,[int]$lb) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    $dir=Join-Path $resultRoot $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:SUNOPT_CUDA_ARCH=$arch
    $base=if($null -eq $originalNvccAppend){""}else{$originalNvccAppend.Trim()}
    $env:NVCC_APPEND_FLAGS=(($base+" -DSUNOPT_LAUNCH_BOUNDS_MODE=$lb").Trim())
    Invoke-ForceBuild
    $buildPath=Join-Path $dir "build.txt"

    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    $build = (
        & npm run build 2>&1 |
        Tee-Object -FilePath $buildPath |
        Out-String
    )

    $exit = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    $e=[ordered]@{label=$label;arch=$arch;launchBoundsMode=$lb;success=$false;medianWallMs=$null;checksumHex=$null}
    Add-ReportLine ""; Add-ReportLine "================================================================"; Add-ReportLine "VARIANT $label"; Add-ReportLine "================================================================"
    Add-ReportLine "SUNOPT_CUDA_ARCH=$arch"; Add-ReportLine "SUNOPT_LAUNCH_BOUNDS_MODE=$lb"; Add-ReportLine "NVCC_APPEND_FLAGS=$($env:NVCC_APPEND_FLAGS)"; Add-ReportLine "buildExitCode=$exit"
    Add-ReportLine "--- BUILD ---"; Add-ReportLine $build.TrimEnd()
    if($exit -ne 0){$script:results += [pscustomobject]$e; Set-Content -Encoding UTF8 -Path $OutputPath -Value $report; return}
    $addon=(Get-ChildItem $root -File -Filter *.node|Sort-Object LastWriteTime -Descending|Select-Object -First 1).FullName
    $hash=(Get-FileHash -Algorithm SHA256 $addon).Hash.ToLowerInvariant()
    $lelf=Join-Path $dir "lelf.txt"; $res=Join-Path $dir "resource-usage.txt"; $sass=Join-Path $dir "sass.txt"
    & cuobjdump -lelf $addon 2>&1 | Set-Content -Encoding UTF8 $lelf
    & cuobjdump -res-usage $addon 2>&1 | Set-Content -Encoding UTF8 $res
    & cuobjdump -sass $addon 2>&1 | Set-Content -Encoding UTF8 $sass
    $calls=@(Get-CausalCalls $sass)
    $nodeOut=(& node (Join-Path $scriptDir "blackwell-replay.mjs") $SnapshotPath $Repetitions $DeviceId 2>&1 | Out-String)
    $jsonLine=($nodeOut -split "`r?`n"|Where-Object{$_ -like "SUNOPT_BLACKWELL_JSON=*"}|Select-Object -Last 1)
    Add-ReportLine "--- ADDON ---"; Add-ReportLine "path=$addon"; Add-ReportLine "sha256=$hash"; Add-ReportLine "--- LELF ---"; Add-ReportLine (Get-Content $lelf -Raw).TrimEnd(); Add-ReportLine "--- RES-USAGE ---"; Add-ReportLine (Get-Content $res -Raw).TrimEnd(); Add-ReportLine "--- CAUSAL CALL.REL ---"; Add-ReportLine "count=$($calls.Count)"; foreach($c in $calls){Add-ReportLine $c}; Add-ReportLine "--- REPLAY RAW ---"; Add-ReportLine $nodeOut.TrimEnd()
    if(-not $jsonLine){$script:results += [pscustomobject]$e; Set-Content -Encoding UTF8 -Path $OutputPath -Value $report; return}
    $j=$jsonLine.Substring("SUNOPT_BLACKWELL_JSON=".Length)|ConvertFrom-Json
    $e.success=$true; $e.medianWallMs=[double]$j.medianWallMs; $e.checksumHex=[string]$j.checksumHex
    Add-ReportLine "--- REPLAY SUMMARY ---"; Add-ReportLine ("medianWallMs={0:N3}" -f $e.medianWallMs); Add-ReportLine ("wallMs=[{0}]" -f (($j.wallMs|ForEach-Object{"{0:N3}" -f [double]$_}) -join ", ")); Add-ReportLine "checksumHex=$($e.checksumHex)"
    $script:results += [pscustomobject]$e
    Set-Content -Encoding UTF8 -Path $OutputPath -Value $report
}
try {
    Add-ReportLine "SUNOPT BLACKWELL COMPILER AUTOTUNE"; Add-ReportLine "generated=$(Get-Date -Format o)"; Add-ReportLine "snapshot=$SnapshotPath"; Add-ReportLine "repetitions=$Repetitions"; Add-ReportLine "deviceId=$DeviceId"
    Add-ReportLine "--- TOOLCHAIN ---"; Add-ReportLine "nvcc:"; Add-ReportLine (Invoke-CapturedCommand{nvcc --version}); Add-ReportLine "nvidia-smi:"; Add-ReportLine (Invoke-CapturedCommand{nvidia-smi}); Add-ReportLine "rustc:"; Add-ReportLine (Invoke-CapturedCommand{rustc --version -v}); Add-ReportLine "cargo:"; Add-ReportLine (Invoke-CapturedCommand{cargo --version}); Add-ReportLine "node:"; Add-ReportLine (Invoke-CapturedCommand{node --version}); Add-ReportLine "npm:"; Add-ReportLine (Invoke-CapturedCommand{npm --version}); Add-ReportLine "git:"; Add-ReportLine (Invoke-CapturedCommand{git -C ../.. rev-parse HEAD}); Add-ReportLine (Invoke-CapturedCommand{git -C ../.. status --short}); Add-ReportLine "snapshotSha256=$((Get-FileHash -Algorithm SHA256 $SnapshotPath).Hash.ToLowerInvariant())"; Add-ReportLine "snapshotBytes=$((Get-Item $SnapshotPath).Length)"
    Invoke-Variant "target-sm120" "sm_120" 0
    Invoke-Variant "target-sm120f" "sm_120f" 0
    Invoke-Variant "target-sm120a" "sm_120a" 0
    $targets=@($results|Where-Object{$_.success -and $_.launchBoundsMode -eq 0}); if($targets.Count -eq 0){throw "No target variant succeeded."}
    $bestTarget=$targets|Sort-Object medianWallMs|Select-Object -First 1
    if(-not $SkipLaunchBounds){Invoke-Variant "launchbounds32-$($bestTarget.arch)" $bestTarget.arch 1; Invoke-Variant "launchbounds32x1-$($bestTarget.arch)" $bestTarget.arch 2}
    $ok=@($results|Where-Object{$_.success}); $winner=$ok|Sort-Object medianWallMs|Select-Object -First 1
    Add-ReportLine ""; Add-ReportLine "================================================================"; Add-ReportLine "RANKING"; Add-ReportLine "================================================================"
    foreach($x in ($ok|Sort-Object medianWallMs)){Add-ReportLine ("{0,-34} arch={1,-8} lb={2} medianWallMs={3:N3} checksum={4}" -f $x.label,$x.arch,$x.launchBoundsMode,[double]$x.medianWallMs,$x.checksumHex)}
    $checks=@($ok|Select-Object -ExpandProperty checksumHex -Unique); Add-ReportLine "checksumVariantCount=$($checks.Count)"; if($checks.Count -eq 1){Add-ReportLine "parity=PASS"}else{Add-ReportLine "parity=FAIL/WARNING"}
    Add-ReportLine "WINNER=$($winner.label)"; Add-ReportLine "WINNER_ARCH=$($winner.arch)"; Add-ReportLine "WINNER_LAUNCH_BOUNDS_MODE=$($winner.launchBoundsMode)"; Add-ReportLine ("WINNER_MEDIAN_WALL_MS={0:N3}" -f [double]$winner.medianWallMs)
    Write-Host "Rebuilding winner $($winner.label)" -ForegroundColor Green
    $env:SUNOPT_CUDA_ARCH=$winner.arch
    $base=if($null -eq $originalNvccAppend){""}else{$originalNvccAppend.Trim()}
    $env:NVCC_APPEND_FLAGS=(($base+" -DSUNOPT_LAUNCH_BOUNDS_MODE=$($winner.launchBoundsMode)").Trim())

    Invoke-ForceBuild

    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    $final = (
        & npm run build 2>&1 |
        Out-String
    )

    $finalExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference

    Add-ReportLine "--- FINAL WINNER REBUILD ---"
    Add-ReportLine $final.TrimEnd()
    Add-ReportLine "finalBuildExitCode=$finalExitCode"
    Set-Content -Encoding UTF8 -Path $OutputPath -Value $report
    Write-Host "Done. Upload: $OutputPath" -ForegroundColor Green
} catch {
    Add-ReportLine ""
    Add-ReportLine "================================================================"
    Add-ReportLine "AUTOTUNE ERROR"
    Add-ReportLine "================================================================"
    Add-ReportLine ("message=" + $_.Exception.Message)
    Add-ReportLine ("position=" + $_.InvocationInfo.PositionMessage)

    Set-Content -Encoding UTF8 -Path $OutputPath -Value $report

    Write-Host "Autotune failed. Partial report preserved: $OutputPath" -ForegroundColor Red

    throw
} finally {
    if($null -eq $originalArch){Remove-Item Env:SUNOPT_CUDA_ARCH -ErrorAction SilentlyContinue}else{$env:SUNOPT_CUDA_ARCH=$originalArch}
    if($null -eq $originalNvccAppend){Remove-Item Env:NVCC_APPEND_FLAGS -ErrorAction SilentlyContinue}else{$env:NVCC_APPEND_FLAGS=$originalNvccAppend}
    $ErrorActionPreference = $originalErrorActionPreference
}

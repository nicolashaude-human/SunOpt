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
$results = @()
$report = New-Object System.Collections.Generic.List[string]
function R([string]$t="") { $script:report.Add($t) }
function Cmd([scriptblock]$c) { try { (& $c 2>&1 | Out-String).TrimEnd() } catch { "ERROR: $($_.Exception.Message)" } }
function ForceBuild { & cargo clean -p sunopt-node 2>&1 | Out-Null }
function CausalCalls([string]$p) {
    $inside=$false; $o=@()
    foreach($line in Get-Content $p) {
        if($line -match "Function\s+:\s+_Z19causal_batch_kernel") { $inside=$true; continue }
        if($inside -and $line -match "Function\s+:") { break }
        if($inside -and $line -match "CALL\.REL") { $o += $line.Trim() }
    }
    return $o
}
function RunVariant([string]$label,[string]$arch,[int]$lb) {
    Write-Host "=== $label ===" -ForegroundColor Cyan
    $dir=Join-Path $resultRoot $label; New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $env:SUNOPT_CUDA_ARCH=$arch
    $base=if($null -eq $originalNvccAppend){""}else{$originalNvccAppend.Trim()}
    $env:NVCC_APPEND_FLAGS=(($base+" -DSUNOPT_LAUNCH_BOUNDS_MODE=$lb").Trim())
    ForceBuild
    $buildPath=Join-Path $dir "build.txt"
    $build=(& npm run build 2>&1 | Tee-Object -FilePath $buildPath | Out-String); $exit=$LASTEXITCODE
    $e=[ordered]@{label=$label;arch=$arch;launchBoundsMode=$lb;success=$false;medianWallMs=$null;checksumHex=$null}
    R ""; R "================================================================"; R "VARIANT $label"; R "================================================================"
    R "SUNOPT_CUDA_ARCH=$arch"; R "SUNOPT_LAUNCH_BOUNDS_MODE=$lb"; R "NVCC_APPEND_FLAGS=$($env:NVCC_APPEND_FLAGS)"; R "buildExitCode=$exit"
    R "--- BUILD ---"; R $build.TrimEnd()
    if($exit -ne 0){$script:results += [pscustomobject]$e; Set-Content -Encoding UTF8 -Path $OutputPath -Value $report; return}
    $addon=(Get-ChildItem $root -File -Filter *.node|Sort-Object LastWriteTime -Descending|Select-Object -First 1).FullName
    $hash=(Get-FileHash -Algorithm SHA256 $addon).Hash.ToLowerInvariant()
    $lelf=Join-Path $dir "lelf.txt"; $res=Join-Path $dir "resource-usage.txt"; $sass=Join-Path $dir "sass.txt"
    & cuobjdump -lelf $addon 2>&1 | Set-Content -Encoding UTF8 $lelf
    & cuobjdump -res-usage $addon 2>&1 | Set-Content -Encoding UTF8 $res
    & cuobjdump -sass $addon 2>&1 | Set-Content -Encoding UTF8 $sass
    $calls=@(CausalCalls $sass)
    $nodeOut=(& node (Join-Path $scriptDir "blackwell-replay.mjs") $SnapshotPath $Repetitions $DeviceId 2>&1 | Out-String)
    $jsonLine=($nodeOut -split "`r?`n"|Where-Object{$_ -like "SUNOPT_BLACKWELL_JSON=*"}|Select-Object -Last 1)
    R "--- ADDON ---"; R "path=$addon"; R "sha256=$hash"; R "--- LELF ---"; R (Get-Content $lelf -Raw).TrimEnd(); R "--- RES-USAGE ---"; R (Get-Content $res -Raw).TrimEnd(); R "--- CAUSAL CALL.REL ---"; R "count=$($calls.Count)"; foreach($c in $calls){R $c}; R "--- REPLAY RAW ---"; R $nodeOut.TrimEnd()
    if(-not $jsonLine){$script:results += [pscustomobject]$e; Set-Content -Encoding UTF8 -Path $OutputPath -Value $report; return}
    $j=$jsonLine.Substring("SUNOPT_BLACKWELL_JSON=".Length)|ConvertFrom-Json
    $e.success=$true; $e.medianWallMs=[double]$j.medianWallMs; $e.checksumHex=[string]$j.checksumHex
    R "--- REPLAY SUMMARY ---"; R ("medianWallMs={0:N3}" -f $e.medianWallMs); R ("wallMs=[{0}]" -f (($j.wallMs|ForEach-Object{"{0:N3}" -f [double]$_}) -join ", ")); R "checksumHex=$($e.checksumHex)"
    $script:results += [pscustomobject]$e
    Set-Content -Encoding UTF8 -Path $OutputPath -Value $report
}
try {
    R "SUNOPT BLACKWELL COMPILER AUTOTUNE"; R "generated=$(Get-Date -Format o)"; R "snapshot=$SnapshotPath"; R "repetitions=$Repetitions"; R "deviceId=$DeviceId"
    R "--- TOOLCHAIN ---"; R "nvcc:"; R (Cmd{nvcc --version}); R "nvidia-smi:"; R (Cmd{nvidia-smi}); R "rustc:"; R (Cmd{rustc --version -v}); R "cargo:"; R (Cmd{cargo --version}); R "node:"; R (Cmd{node --version}); R "npm:"; R (Cmd{npm --version}); R "git:"; R (Cmd{git -C ../.. rev-parse HEAD}); R (Cmd{git -C ../.. status --short}); R "snapshotSha256=$((Get-FileHash -Algorithm SHA256 $SnapshotPath).Hash.ToLowerInvariant())"; R "snapshotBytes=$((Get-Item $SnapshotPath).Length)"
    RunVariant "target-sm120" "sm_120" 0
    RunVariant "target-sm120f" "sm_120f" 0
    RunVariant "target-sm120a" "sm_120a" 0
    $targets=@($results|Where-Object{$_.success -and $_.launchBoundsMode -eq 0}); if($targets.Count -eq 0){throw "No target variant succeeded."}
    $bestTarget=$targets|Sort-Object medianWallMs|Select-Object -First 1
    if(-not $SkipLaunchBounds){RunVariant "launchbounds32-$($bestTarget.arch)" $bestTarget.arch 1; RunVariant "launchbounds32x1-$($bestTarget.arch)" $bestTarget.arch 2}
    $ok=@($results|Where-Object{$_.success}); $winner=$ok|Sort-Object medianWallMs|Select-Object -First 1
    R ""; R "================================================================"; R "RANKING"; R "================================================================"
    foreach($x in ($ok|Sort-Object medianWallMs)){R ("{0,-34} arch={1,-8} lb={2} medianWallMs={3:N3} checksum={4}" -f $x.label,$x.arch,$x.launchBoundsMode,[double]$x.medianWallMs,$x.checksumHex)}
    $checks=@($ok|Select-Object -ExpandProperty checksumHex -Unique); R "checksumVariantCount=$($checks.Count)"; if($checks.Count -eq 1){R "parity=PASS"}else{R "parity=FAIL/WARNING"}
    R "WINNER=$($winner.label)"; R "WINNER_ARCH=$($winner.arch)"; R "WINNER_LAUNCH_BOUNDS_MODE=$($winner.launchBoundsMode)"; R ("WINNER_MEDIAN_WALL_MS={0:N3}" -f [double]$winner.medianWallMs)
    Write-Host "Rebuilding winner $($winner.label)" -ForegroundColor Green
    $env:SUNOPT_CUDA_ARCH=$winner.arch; $base=if($null -eq $originalNvccAppend){""}else{$originalNvccAppend.Trim()}; $env:NVCC_APPEND_FLAGS=(($base+" -DSUNOPT_LAUNCH_BOUNDS_MODE=$($winner.launchBoundsMode)").Trim()); ForceBuild; $final=(& npm run build 2>&1|Out-String); R "--- FINAL WINNER REBUILD ---"; R $final.TrimEnd(); R "finalBuildExitCode=$LASTEXITCODE"
    Set-Content -Encoding UTF8 -Path $OutputPath -Value $report
    Write-Host "Done. Upload: $OutputPath" -ForegroundColor Green
} finally {
    if($null -eq $originalArch){Remove-Item Env:SUNOPT_CUDA_ARCH -ErrorAction SilentlyContinue}else{$env:SUNOPT_CUDA_ARCH=$originalArch}
    if($null -eq $originalNvccAppend){Remove-Item Env:NVCC_APPEND_FLAGS -ErrorAction SilentlyContinue}else{$env:NVCC_APPEND_FLAGS=$originalNvccAppend}
}

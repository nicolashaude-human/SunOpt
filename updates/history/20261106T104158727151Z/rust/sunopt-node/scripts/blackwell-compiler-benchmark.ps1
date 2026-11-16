param(
    [int]$Repetitions = 5,
    [int]$DeviceId = 0,
    [string]$SnapshotPath = "",
    [string]$OutputPath = "",
    [switch]$SkipLaunchBounds,
    [int]$NativeVfVoltageMv = 1050,
    [int]$NativeVfFrequencyMHz = 2865,
    [int]$ExpectedObservedGpuClockMHz = 2857,
    [int]$ExpectedFanPercent = 70,
    [int]$FanTolerancePercent = 3,
    [int]$TelemetryIntervalMs = 250,
    [int]$WarmupRepetitions = 1,
    [int]$ClockToleranceMHz = 5,
    [int]$ActiveUtilizationThresholdPercent = 50,
    [switch]$SkipClockValidation,
    [switch]$SkipFanValidation
)
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
Set-Location $root
if ([string]::IsNullOrWhiteSpace($SnapshotPath)) { $SnapshotPath = Join-Path $root "blackwell-benchmark.bin" }
if ([string]::IsNullOrWhiteSpace($OutputPath)) { $OutputPath = Join-Path $root "llm.txt" }
$resultRoot = Join-Path $root "blackwell-benchmark-results"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

$guardScript = Join-Path $scriptDir "nvapi-blackwell-benchmark-guard.py"
$guardRoot = Join-Path $resultRoot "nvapi-guard"
New-Item -ItemType Directory -Force -Path $guardRoot | Out-Null
$guardReadyFile = Join-Path $guardRoot "ready.json"
$guardStopFile = Join-Path $guardRoot "stop.signal"
$guardDoneFile = Join-Path $guardRoot "done.json"
$guardLogFile = Join-Path $guardRoot "guard.log"

$cudaSourcePath = Join-Path $root "cuda\cuda_context.cu"

if (-not (Test-Path $cudaSourcePath)) {
    throw "CUDA source not found: $cudaSourcePath"
}

$originalCudaSource = Get-Content -Raw -Encoding UTF8 $cudaSourcePath
if (-not (Test-Path $SnapshotPath)) {
    Write-Host "Blackwell replay snapshot not found: $SnapshotPath" -ForegroundColor Yellow
    Write-Host "Restart Next.js and run the normal 28-tank benchmark once; Rust creates the snapshot automatically."
    exit 2
}
$originalArch = $env:SUNOPT_CUDA_ARCH
$originalErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$results = @()
$report = New-Object System.Collections.Generic.List[string]
$script:NvapiGuardProcess = $null
$script:NvapiGuardStarted = $false
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
function Set-LaunchBoundsMode {
    param(
        [int]$Mode
    )

    if ($Mode -lt 0 -or $Mode -gt 2) {
        throw "Invalid SUNOPT_LAUNCH_BOUNDS_MODE: $Mode"
    }

    $source = Get-Content -Raw -Encoding UTF8 $cudaSourcePath

    $pattern = '(?m)^#define SUNOPT_LAUNCH_BOUNDS_MODE\s+\d+\s*$'
    $matches = [regex]::Matches($source, $pattern)

    if ($matches.Count -ne 1) {
        throw "Expected exactly one SUNOPT_LAUNCH_BOUNDS_MODE default define in cuda_context.cu; found $($matches.Count)"
    }

    $updated = [regex]::Replace(
        $source,
        $pattern,
        "#define SUNOPT_LAUNCH_BOUNDS_MODE $Mode",
        1
    )

    Set-Content -Encoding UTF8 -NoNewline -Path $cudaSourcePath -Value $updated
}


function Get-GpuQueryValue {
    param(
        [string]$Field
    )

    $value = (
        & nvidia-smi `
            -i $DeviceId `
            "--query-gpu=$Field" `
            "--format=csv,noheader,nounits" 2>$null |
        Select-Object -First 1
    )

    if ($null -eq $value) {
        return $null
    }

    return $value.ToString().Trim()
}

function Start-BenchmarkNvapiGuard {
    if (-not (Test-Path $guardScript)) {
        throw "NVAPI benchmark guard not found: $guardScript"
    }

    Remove-Item $guardReadyFile -ErrorAction SilentlyContinue
    Remove-Item $guardStopFile -ErrorAction SilentlyContinue
    Remove-Item $guardDoneFile -ErrorAction SilentlyContinue
    Remove-Item $guardLogFile -ErrorAction SilentlyContinue

    $args = @(
        $guardScript,
        "--gpu", "$DeviceId",
        "--target-voltage-mv", "$NativeVfVoltageMv",
        "--native-vf-frequency-mhz", "$NativeVfFrequencyMHz",
        "--fan-percent", "$ExpectedFanPercent",
        "--fan-tolerance", "$FanTolerancePercent",
        "--parent-pid", "$PID",
        "--ready-file", $guardReadyFile,
        "--stop-file", $guardStopFile,
        "--done-file", $guardDoneFile,
        "--log-file", $guardLogFile
    )

    $script:NvapiGuardProcess = Start-Process `
        -FilePath "python" `
        -ArgumentList $args `
        -WindowStyle Hidden `
        -PassThru

    $script:NvapiGuardStarted = $true

    $deadline = (Get-Date).AddSeconds(30)

    while ((Get-Date) -lt $deadline) {
        if (Test-Path $guardReadyFile) {
            $ready = Get-Content -Raw -Encoding UTF8 $guardReadyFile | ConvertFrom-Json

            Add-ReportLine "nvapiGuard=READY"
            Add-ReportLine "nvapiGuard.pid=$($script:NvapiGuardProcess.Id)"
            Add-ReportLine "nvapiGuard.gpuName=$($ready.gpuName)"
            Add-ReportLine "nvapiGuard.nativeVfSlot=$($ready.nativeVfSlot)"
            Add-ReportLine "nvapiGuard.targetVoltageMv=$($ready.targetVoltageMv)"
            Add-ReportLine "nvapiGuard.nativeVfFrequencyMHz=$($ready.nativeVfFrequencyMHz)"
            Add-ReportLine "nvapiGuard.fanPercent=$($ready.fanPercent)"
            Add-ReportLine "nvapiGuard.reportedFanPercent=$($ready.reportedFanPercent)"
            Add-ReportLine "nvapiGuard.temperatureC=$($ready.temperatureC)"
            return
        }

        if ($script:NvapiGuardProcess.HasExited) {
            $guardText = ""

            if (Test-Path $guardLogFile) {
                $guardText = Get-Content -Raw -Encoding UTF8 $guardLogFile
            }

            throw (
                "NVAPI benchmark guard exited before READY. ExitCode=" +
                $script:NvapiGuardProcess.ExitCode +
                "`n" +
                $guardText
            )
        }

        Start-Sleep -Milliseconds 250
    }

    throw "Timed out waiting for NVAPI benchmark guard READY state."
}

function Stop-BenchmarkNvapiGuard {
    if (-not $script:NvapiGuardStarted) {
        return
    }

    try {
        New-Item -ItemType File -Force -Path $guardStopFile | Out-Null

        $deadline = (Get-Date).AddSeconds(30)

        while (
            -not $script:NvapiGuardProcess.HasExited -and
            (Get-Date) -lt $deadline
        ) {
            Start-Sleep -Milliseconds 250
        }

        if (-not $script:NvapiGuardProcess.HasExited) {
            Add-ReportLine "nvapiGuard.stop=TIMEOUT"
            Write-Host (
                "WARNING: NVAPI guard did not exit after stop request. " +
                "Do not kill it manually; it is responsible for restoring GPU controls."
            ) -ForegroundColor Red
        } else {
            Add-ReportLine "nvapiGuard.stop=EXITED"
            Add-ReportLine "nvapiGuard.exitCode=$($script:NvapiGuardProcess.ExitCode)"
        }

        if (Test-Path $guardDoneFile) {
            $done = Get-Content -Raw -Encoding UTF8 $guardDoneFile | ConvertFrom-Json
            Add-ReportLine "nvapiGuard.restoreStatus=$($done.status)"
            Add-ReportLine "nvapiGuard.restoreOk=$($done.restoreOk)"
        }

        if (Test-Path $guardLogFile) {
            Add-ReportLine ""
            Add-ReportLine "================================================================"
            Add-ReportLine "NVAPI BENCHMARK GUARD LOG"
            Add-ReportLine "================================================================"
            Add-ReportLine (Get-Content -Raw -Encoding UTF8 $guardLogFile).TrimEnd()
        }
    } finally {
        $script:NvapiGuardStarted = $false
    }
}


function Test-BenchmarkFan {
    $fanValue = Get-GpuQueryValue -Field "fan.speed"

    Add-ReportLine "expectedFanPercent=$ExpectedFanPercent"
    Add-ReportLine "fanTolerancePercent=$FanTolerancePercent"
    Add-ReportLine "reportedFanPercent=$fanValue"

    if ($SkipFanValidation) {
        Add-ReportLine "fanValidation=SKIPPED"
        return
    }

    if (
        [string]::IsNullOrWhiteSpace($fanValue) -or
        $fanValue -eq "N/A"
    ) {
        Add-ReportLine "fanValidation=UNAVAILABLE"
        Write-Host "Fan speed is not reported by nvidia-smi; continuing without fan validation." -ForegroundColor Yellow
        return
    }

    $fanNumber = 0.0
    $parsed = [double]::TryParse(
        $fanValue,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$fanNumber
    )

    if (-not $parsed) {
        Add-ReportLine "fanValidation=UNPARSEABLE"
        throw "Could not parse fan.speed value: $fanValue"
    }

    $delta = [math]::Abs($fanNumber - $ExpectedFanPercent)
    Add-ReportLine ("fanDeltaPercent={0:F1}" -f $delta)

    if ($delta -gt $FanTolerancePercent) {
        Add-ReportLine "fanValidation=FAIL"
        throw (
            "GPU fan is reported at $fanNumber %, expected $ExpectedFanPercent +/- $FanTolerancePercent %. " +
            "Set the fan manually to $ExpectedFanPercent % and rerun."
        )
    }

    Add-ReportLine "fanValidation=PASS"
}

function Start-GpuTelemetry {
    param(
        [string]$Path
    )

    $errorPath = "$Path.stderr.txt"

    Remove-Item $Path -ErrorAction SilentlyContinue
    Remove-Item $errorPath -ErrorAction SilentlyContinue

    $query = (
        "--query-gpu=" +
        "timestamp,pstate,clocks.gr,clocks.mem,temperature.gpu," +
        "fan.speed,power.draw,power.limit,utilization.gpu"
    )

    $args = @(
        "-i", "$DeviceId",
        $query,
        "--format=csv,noheader,nounits",
        "-lms", "$TelemetryIntervalMs"
    )

    $process = Start-Process `
        -FilePath "nvidia-smi" `
        -ArgumentList $args `
        -RedirectStandardOutput $Path `
        -RedirectStandardError $errorPath `
        -WindowStyle Hidden `
        -PassThru

    return [pscustomobject]@{
        Process = $process
        Path = $Path
        ErrorPath = $errorPath
    }
}

function Stop-GpuTelemetry {
    param(
        $Telemetry
    )

    if ($null -eq $Telemetry) {
        return
    }

    try {
        if (-not $Telemetry.Process.HasExited) {
            Stop-Process -Id $Telemetry.Process.Id -Force -ErrorAction SilentlyContinue
            $Telemetry.Process.WaitForExit()
        }
    } catch {
        # The process may already have terminated after the benchmark.
    }
}

function Get-TelemetrySummary {
    param(
        [string]$Path
    )

    $rows = @()

    if (Test-Path $Path) {
        $rows = @(
            Get-Content $Path |
            Where-Object {
                -not [string]::IsNullOrWhiteSpace($_)
            }
        )
    }

    $allTemps = New-Object System.Collections.Generic.List[double]
    $activeTemps = New-Object System.Collections.Generic.List[double]
    $activeClocks = New-Object System.Collections.Generic.List[double]
    $activeFans = New-Object System.Collections.Generic.List[double]
    $activePowers = New-Object System.Collections.Generic.List[double]
    $activeUtils = New-Object System.Collections.Generic.List[double]
    $activeSamples = 0

    foreach ($line in $rows) {
        $parts = @($line -split "," | ForEach-Object { $_.Trim() })

        if ($parts.Count -lt 9) {
            continue
        }

        $clock = 0.0
        $temp = 0.0
        $fan = 0.0
        $power = 0.0
        $util = 0.0

        $clockOk = [double]::TryParse(
            $parts[2],
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$clock
        )
        $tempOk = [double]::TryParse(
            $parts[4],
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$temp
        )
        $fanOk = [double]::TryParse(
            $parts[5],
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$fan
        )
        $powerOk = [double]::TryParse(
            $parts[6],
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$power
        )
        $utilOk = [double]::TryParse(
            $parts[8],
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$util
        )

        if ($tempOk) {
            [void]$allTemps.Add($temp)
        }

        if (
            $utilOk -and
            $util -ge $ActiveUtilizationThresholdPercent
        ) {
            $activeSamples += 1

            if ($tempOk)  { [void]$activeTemps.Add($temp) }
            if ($clockOk) { [void]$activeClocks.Add($clock) }
            if ($fanOk)   { [void]$activeFans.Add($fan) }
            if ($powerOk) { [void]$activePowers.Add($power) }
            [void]$activeUtils.Add($util)
        }
    }

    function Get-Stats($values) {
        if ($values.Count -eq 0) {
            return [pscustomobject]@{
                Min=$null
                Max=$null
                Avg=$null
            }
        }

        return [pscustomobject]@{
            Min=($values | Measure-Object -Minimum).Minimum
            Max=($values | Measure-Object -Maximum).Maximum
            Avg=($values | Measure-Object -Average).Average
        }
    }

    $allT = Get-Stats $allTemps
    $t = Get-Stats $activeTemps
    $c = Get-Stats $activeClocks
    $f = Get-Stats $activeFans
    $p = Get-Stats $activePowers
    $u = Get-Stats $activeUtils

    return [pscustomobject]@{
        Samples = $rows.Count
        ActiveSamples = $activeSamples
        AllTempMinC = $allT.Min
        AllTempMaxC = $allT.Max
        TempMinC = $t.Min
        TempMaxC = $t.Max
        TempAvgC = $t.Avg
        ClockMinMHz = $c.Min
        ClockMaxMHz = $c.Max
        ClockAvgMHz = $c.Avg
        FanMinPercent = $f.Min
        FanMaxPercent = $f.Max
        FanAvgPercent = $f.Avg
        PowerMaxW = $p.Max
        UtilAvgPercent = $u.Avg
    }
}

function Test-ObservedClock {
    param(
        $TelemetrySummary
    )

    if ($SkipClockValidation) {
        Add-ReportLine "observedClockValidation=SKIPPED"
        return
    }

    if (
        $TelemetrySummary.ActiveSamples -le 0 -or
        $null -eq $TelemetrySummary.ClockMinMHz -or
        $null -eq $TelemetrySummary.ClockMaxMHz
    ) {
        Add-ReportLine "observedClockValidation=FAIL_NO_ACTIVE_SAMPLES"
        throw "No active GPU telemetry samples were captured for clock validation."
    }

    $minAllowed = $ExpectedObservedGpuClockMHz - $ClockToleranceMHz
    $maxAllowed = $ExpectedObservedGpuClockMHz + $ClockToleranceMHz

    Add-ReportLine "observedClockTargetMHz=$ExpectedObservedGpuClockMHz"
    Add-ReportLine "observedClockToleranceMHz=$ClockToleranceMHz"
    Add-ReportLine "observedClockAllowedMHz=$minAllowed..$maxAllowed"
    Add-ReportLine "observedClockActiveMinMHz=$($TelemetrySummary.ClockMinMHz)"
    Add-ReportLine "observedClockActiveMaxMHz=$($TelemetrySummary.ClockMaxMHz)"
    Add-ReportLine "observedClockActiveAvgMHz=$($TelemetrySummary.ClockAvgMHz)"

    if (
        $TelemetrySummary.ClockMinMHz -lt $minAllowed -or
        $TelemetrySummary.ClockMaxMHz -gt $maxAllowed
    ) {
        Add-ReportLine "observedClockValidation=FAIL"
        throw (
            "Observed active GPU clock left the allowed range " +
            "$minAllowed..$maxAllowed MHz."
        )
    }

    Add-ReportLine "observedClockValidation=PASS"
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

    Set-LaunchBoundsMode -Mode $lb

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
    $e=[ordered]@{
        label=$label
        arch=$arch
        launchBoundsMode=$lb
        success=$false
        medianWallMs=$null
        checksumHex=$null
        addonSha256=$null
        telemetrySamples=0
        activeTelemetrySamples=0
        tempMinC=$null
        tempMaxC=$null
        tempAvgC=$null
        clockMinMHz=$null
        clockMaxMHz=$null
        clockAvgMHz=$null
        fanMinPercent=$null
        fanMaxPercent=$null
        fanAvgPercent=$null
    }
    Add-ReportLine ""; Add-ReportLine "================================================================"; Add-ReportLine "VARIANT $label"; Add-ReportLine "================================================================"
    Add-ReportLine "SUNOPT_CUDA_ARCH=$arch"
    Add-ReportLine "SUNOPT_LAUNCH_BOUNDS_MODE=$lb"
    Add-ReportLine "launchBoundsInjection=source-default-define"
    Add-ReportLine "buildExitCode=$exit"
    Add-ReportLine "--- BUILD ---"; Add-ReportLine $build.TrimEnd()
    if($exit -ne 0){$script:results += [pscustomobject]$e; Set-Content -Encoding UTF8 -Path $OutputPath -Value $report; return}
    $addon=(Get-ChildItem $root -File -Filter *.node|Sort-Object LastWriteTime -Descending|Select-Object -First 1).FullName
    $hash=(Get-FileHash -Algorithm SHA256 $addon).Hash.ToLowerInvariant()
    $e.addonSha256=$hash
    $lelf=Join-Path $dir "lelf.txt"; $res=Join-Path $dir "resource-usage.txt"; $sass=Join-Path $dir "sass.txt"
    & cuobjdump -lelf $addon 2>&1 | Set-Content -Encoding UTF8 $lelf
    & cuobjdump -res-usage $addon 2>&1 | Set-Content -Encoding UTF8 $res
    & cuobjdump -sass $addon 2>&1 | Set-Content -Encoding UTF8 $sass
    $calls=@(Get-CausalCalls $sass)

    if ($WarmupRepetitions -gt 0) {
        Write-Host "  warmup x$WarmupRepetitions" -ForegroundColor DarkGray

        & node `
            (Join-Path $scriptDir "blackwell-replay.mjs") `
            $SnapshotPath `
            $WarmupRepetitions `
            $DeviceId *> $null
    }

    $telemetryPath = Join-Path $dir "gpu-telemetry.csv"
    $telemetry = Start-GpuTelemetry -Path $telemetryPath

    try {
        Start-Sleep -Milliseconds ([math]::Max(250, $TelemetryIntervalMs))

        $nodeOut=(
            & node `
                (Join-Path $scriptDir "blackwell-replay.mjs") `
                $SnapshotPath `
                $Repetitions `
                $DeviceId 2>&1 |
            Out-String
        )
    } finally {
        Stop-GpuTelemetry -Telemetry $telemetry
    }

    $telemetrySummary = Get-TelemetrySummary -Path $telemetryPath

    $e.telemetrySamples = $telemetrySummary.Samples
    $e.activeTelemetrySamples = $telemetrySummary.ActiveSamples
    $e.tempMinC = $telemetrySummary.TempMinC
    $e.tempMaxC = $telemetrySummary.TempMaxC
    $e.tempAvgC = $telemetrySummary.TempAvgC
    $e.clockMinMHz = $telemetrySummary.ClockMinMHz
    $e.clockMaxMHz = $telemetrySummary.ClockMaxMHz
    $e.clockAvgMHz = $telemetrySummary.ClockAvgMHz
    $e.fanMinPercent = $telemetrySummary.FanMinPercent
    $e.fanMaxPercent = $telemetrySummary.FanMaxPercent
    $e.fanAvgPercent = $telemetrySummary.FanAvgPercent

    Test-ObservedClock -TelemetrySummary $telemetrySummary

    $jsonLine=($nodeOut -split "`r?`n"|Where-Object{$_ -like "SUNOPT_BLACKWELL_JSON=*"}|Select-Object -Last 1)
    Add-ReportLine "--- ADDON ---"
    Add-ReportLine "path=$addon"
    Add-ReportLine "sha256=$hash"
    Add-ReportLine "--- LELF ---"
    Add-ReportLine (Get-Content $lelf -Raw).TrimEnd()
    Add-ReportLine "--- RES-USAGE ---"
    Add-ReportLine (Get-Content $res -Raw).TrimEnd()
    Add-ReportLine "--- CAUSAL CALL.REL ---"
    Add-ReportLine "count=$($calls.Count)"

    foreach($c in $calls){
        Add-ReportLine $c
    }

    Add-ReportLine "--- GPU TELEMETRY SUMMARY ---"
    Add-ReportLine "telemetryIntervalMs=$TelemetryIntervalMs"
    Add-ReportLine "telemetrySamples=$($telemetrySummary.Samples)"
    Add-ReportLine "activeUtilizationThresholdPercent=$ActiveUtilizationThresholdPercent"
    Add-ReportLine "activeTelemetrySamples=$($telemetrySummary.ActiveSamples)"
    Add-ReportLine ("temperatureC.all.min={0}" -f $telemetrySummary.AllTempMinC)
    Add-ReportLine ("temperatureC.all.max={0}" -f $telemetrySummary.AllTempMaxC)
    Add-ReportLine ("temperatureC.active.min={0}" -f $telemetrySummary.TempMinC)
    Add-ReportLine ("temperatureC.active.max={0}" -f $telemetrySummary.TempMaxC)
    Add-ReportLine ("temperatureC.active.avg={0}" -f $telemetrySummary.TempAvgC)
    Add-ReportLine ("gpuClockMHz.min={0}" -f $telemetrySummary.ClockMinMHz)
    Add-ReportLine ("gpuClockMHz.max={0}" -f $telemetrySummary.ClockMaxMHz)
    Add-ReportLine ("gpuClockMHz.avg={0}" -f $telemetrySummary.ClockAvgMHz)
    Add-ReportLine ("fanPercent.min={0}" -f $telemetrySummary.FanMinPercent)
    Add-ReportLine ("fanPercent.max={0}" -f $telemetrySummary.FanMaxPercent)
    Add-ReportLine ("fanPercent.avg={0}" -f $telemetrySummary.FanAvgPercent)
    Add-ReportLine ("powerW.max={0}" -f $telemetrySummary.PowerMaxW)
    Add-ReportLine ("utilizationPercent.avg={0}" -f $telemetrySummary.UtilAvgPercent)

    Add-ReportLine "--- GPU TELEMETRY TICKS ---"
    Add-ReportLine "timestamp,pstate,clocks.gr MHz,clocks.mem MHz,temperature.gpu C,fan.speed %,power.draw W,power.limit W,utilization.gpu %"

    if (Test-Path $telemetryPath) {
        foreach ($telemetryLine in Get-Content $telemetryPath) {
            Add-ReportLine $telemetryLine
        }
    }

    if (
        $null -ne $telemetry -and
        (Test-Path $telemetry.ErrorPath)
    ) {
        $telemetryError = (Get-Content $telemetry.ErrorPath -Raw).Trim()

        if (-not [string]::IsNullOrWhiteSpace($telemetryError)) {
            Add-ReportLine "--- GPU TELEMETRY STDERR ---"
            Add-ReportLine $telemetryError
        }
    }

    Add-ReportLine "--- REPLAY RAW ---"
    Add-ReportLine $nodeOut.TrimEnd()
    if(-not $jsonLine){$script:results += [pscustomobject]$e; Set-Content -Encoding UTF8 -Path $OutputPath -Value $report; return}
    $j=$jsonLine.Substring("SUNOPT_BLACKWELL_JSON=".Length)|ConvertFrom-Json
    $e.success=$true; $e.medianWallMs=[double]$j.medianWallMs; $e.checksumHex=[string]$j.checksumHex
    Add-ReportLine "--- REPLAY SUMMARY ---"; Add-ReportLine ("medianWallMs={0:N3}" -f $e.medianWallMs); Add-ReportLine ("wallMs=[{0}]" -f (($j.wallMs|ForEach-Object{"{0:N3}" -f [double]$_}) -join ", ")); Add-ReportLine "checksumHex=$($e.checksumHex)"
    $script:results += [pscustomobject]$e
    Set-Content -Encoding UTF8 -Path $OutputPath -Value $report
}
try {
    Add-ReportLine "SUNOPT BLACKWELL COMPILER AUTOTUNE"
    Add-ReportLine "generated=$(Get-Date -Format o)"
    Add-ReportLine "snapshot=$SnapshotPath"
    Add-ReportLine "repetitions=$Repetitions"
    Add-ReportLine "warmupRepetitions=$WarmupRepetitions"
    Add-ReportLine "deviceId=$DeviceId"
    Add-ReportLine "gpuControlMode=nvapi-native-vf-lock"
    Add-ReportLine "nativeVfVoltageMv=$NativeVfVoltageMv"
    Add-ReportLine "nativeVfFrequencyMHz=$NativeVfFrequencyMHz"
    Add-ReportLine "expectedObservedGpuClockMHz=$ExpectedObservedGpuClockMHz"
    Add-ReportLine "clockToleranceMHz=$ClockToleranceMHz"
    Add-ReportLine "activeUtilizationThresholdPercent=$ActiveUtilizationThresholdPercent"
    Add-ReportLine "expectedFanPercent=$ExpectedFanPercent"
    Add-ReportLine "telemetryIntervalMs=$TelemetryIntervalMs"

    Start-BenchmarkNvapiGuard
    Test-BenchmarkFan
    Add-ReportLine "--- TOOLCHAIN ---"; Add-ReportLine "nvcc:"; Add-ReportLine (Invoke-CapturedCommand{nvcc --version}); Add-ReportLine "nvidia-smi:"; Add-ReportLine (Invoke-CapturedCommand{nvidia-smi}); Add-ReportLine "rustc:"; Add-ReportLine (Invoke-CapturedCommand{rustc --version -v}); Add-ReportLine "cargo:"; Add-ReportLine (Invoke-CapturedCommand{cargo --version}); Add-ReportLine "node:"; Add-ReportLine (Invoke-CapturedCommand{node --version}); Add-ReportLine "npm:"; Add-ReportLine (Invoke-CapturedCommand{npm --version}); Add-ReportLine "git:"; Add-ReportLine (Invoke-CapturedCommand{git -C ../.. rev-parse HEAD}); Add-ReportLine (Invoke-CapturedCommand{git -C ../.. status --short}); Add-ReportLine "snapshotSha256=$((Get-FileHash -Algorithm SHA256 $SnapshotPath).Hash.ToLowerInvariant())"; Add-ReportLine "snapshotBytes=$((Get-Item $SnapshotPath).Length)"
    Invoke-Variant "target-sm120" "sm_120" 0
    Invoke-Variant "target-sm120f" "sm_120f" 0
    Invoke-Variant "target-sm120a" "sm_120a" 0
    $targets=@($results|Where-Object{$_.success -and $_.launchBoundsMode -eq 0}); if($targets.Count -eq 0){throw "No target variant succeeded."}
    $bestTarget=$targets|Sort-Object medianWallMs|Select-Object -First 1
    if(-not $SkipLaunchBounds){Invoke-Variant "launchbounds32-$($bestTarget.arch)" $bestTarget.arch 1; Invoke-Variant "launchbounds32x1-$($bestTarget.arch)" $bestTarget.arch 2}
    $ok=@($results|Where-Object{$_.success})

    $bestTargetBaseline = @(
        $ok |
        Where-Object {
            $_.arch -eq $bestTarget.arch -and
            $_.launchBoundsMode -eq 0
        }
    ) |
    Select-Object -First 1

    if (-not $SkipLaunchBounds -and $bestTargetBaseline) {
        foreach ($launchVariant in @(
            $ok |
            Where-Object {
                $_.arch -eq $bestTarget.arch -and
                $_.launchBoundsMode -gt 0
            }
        )) {
            if ($launchVariant.addonSha256 -eq $bestTargetBaseline.addonSha256) {
                throw (
                    "Launch-bounds variant $($launchVariant.label) produced the same addon SHA256 as " +
                    "$($bestTargetBaseline.label). Benchmark is invalid."
                )
            }
        }
    }

    $winner=$ok|Sort-Object medianWallMs|Select-Object -First 1
    Add-ReportLine ""; Add-ReportLine "================================================================"; Add-ReportLine "RANKING"; Add-ReportLine "================================================================"
    foreach($x in ($ok|Sort-Object medianWallMs)){
        Add-ReportLine (
            "{0,-34} arch={1,-8} lb={2} medianWallMs={3:N3} checksum={4} addonSha256={5}" -f
            $x.label,
            $x.arch,
            $x.launchBoundsMode,
            [double]$x.medianWallMs,
            $x.checksumHex,
            $x.addonSha256
        )

        Add-ReportLine (
            "  telemetry samples={0} active={1} tempC={2}-{3} avg={4} clockMHz={5}-{6} avg={7} fan%={8}-{9} avg={10}" -f
            $x.telemetrySamples,
            $x.activeTelemetrySamples,
            $x.tempMinC,
            $x.tempMaxC,
            $x.tempAvgC,
            $x.clockMinMHz,
            $x.clockMaxMHz,
            $x.clockAvgMHz,
            $x.fanMinPercent,
            $x.fanMaxPercent,
            $x.fanAvgPercent
        )
    }
    $checks=@($ok|Select-Object -ExpandProperty checksumHex -Unique); Add-ReportLine "checksumVariantCount=$($checks.Count)"; if($checks.Count -eq 1){Add-ReportLine "parity=PASS"}else{Add-ReportLine "parity=FAIL/WARNING"}
    Add-ReportLine "benchmarkGpuControl=NVAPI"
    Add-ReportLine "benchmarkNativeVfPoint=$NativeVfVoltageMv mV / $NativeVfFrequencyMHz MHz"
    Add-ReportLine "benchmarkExpectedObservedClockMHz=$ExpectedObservedGpuClockMHz"
    Add-ReportLine "benchmarkFanPercent=$ExpectedFanPercent"
    Add-ReportLine "WINNER=$($winner.label)"; Add-ReportLine "WINNER_ARCH=$($winner.arch)"; Add-ReportLine "WINNER_LAUNCH_BOUNDS_MODE=$($winner.launchBoundsMode)"; Add-ReportLine ("WINNER_MEDIAN_WALL_MS={0:N3}" -f [double]$winner.medianWallMs)
    Write-Host "Rebuilding winner $($winner.label)" -ForegroundColor Green
    $env:SUNOPT_CUDA_ARCH=$winner.arch

    Set-LaunchBoundsMode -Mode $winner.launchBoundsMode

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
    if ($null -ne $originalCudaSource) {
        Set-Content -Encoding UTF8 -NoNewline -Path $cudaSourcePath -Value $originalCudaSource
    }

    Stop-BenchmarkNvapiGuard

    if($null -eq $originalArch){Remove-Item Env:SUNOPT_CUDA_ARCH -ErrorAction SilentlyContinue}else{$env:SUNOPT_CUDA_ARCH=$originalArch}
    $ErrorActionPreference = $originalErrorActionPreference

    Set-Content -Encoding UTF8 -Path $OutputPath -Value $report
}

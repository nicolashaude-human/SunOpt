param(
    [int]$DeviceId = 0,
    [string]$SnapshotPath = "",
    [string]$OutputPath = "",
    [int]$VoltageMv = 1050,
    [int]$NativeFrequencyMHz = 2865,
    [int]$FanPercent = 70,
    [int]$TemperatureStopC = 60,
    [double]$PowerStopW = 140.0,
    [int]$RepetitionsPerPoint = 3,
    [int]$WarmupRepetitions = 1,
    [int]$TelemetryIntervalMs = 250,
    [int]$ClockSpanToleranceMHz = 15,
    [int]$ExpectedMemoryClockMHz = 13801,
    [int]$MemoryClockToleranceMHz = 25,
    [int]$CapBandPoints = 8,
    [string]$ExpectedChecksum = "218451794297e021"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
Set-Location $root

if ([string]::IsNullOrWhiteSpace($SnapshotPath)) {
    $SnapshotPath = Join-Path $root "blackwell-benchmark.bin"
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $root "llm-core-offset-sweep.txt"
}

$guardScript = Join-Path $scriptDir "nvapi-blackwell-benchmark-guard.py"
$replayScript = Join-Path $scriptDir "blackwell-replay.mjs"
$resultRoot = Join-Path $root "blackwell-core-offset-sweep-results"

New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

foreach ($required in @($guardScript, $replayScript, $SnapshotPath)) {
    if (-not (Test-Path $required)) {
        throw "Required file not found: $required"
    }
}

# Corrected second-pass sweep.
# +600 is repeated first as the last proven stable point,
# then the core offset advances in 50 MHz steps.
$offsetsMHz = @(600, 650, 700, 750, 800, 850, 900)

$report = New-Object System.Collections.Generic.List[string]
$results = New-Object System.Collections.Generic.List[object]
$stopSweep = $false
$stopReason = ""
$previousActiveClock = $null

function Add-Line {
    param([AllowEmptyString()][string]$Text = "")

    [void]$script:report.Add($Text)
    Set-Content -Encoding UTF8 -Path $OutputPath -Value $script:report
}

function Parse-DoubleInvariant {
    param([string]$Text)

    $v = 0.0
    if ([double]::TryParse(
        $Text,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$v
    )) {
        return $v
    }

    return $null
}

function Start-Telemetry {
    param([string]$Path)

    $err = "$Path.stderr.txt"
    Remove-Item $Path -ErrorAction SilentlyContinue
    Remove-Item $err -ErrorAction SilentlyContinue

    $query = (
        "--query-gpu=" +
        "timestamp,pstate,clocks.gr,clocks.mem,temperature.gpu," +
        "fan.speed,power.draw,power.limit,utilization.gpu"
    )

    $p = Start-Process `
        -FilePath "nvidia-smi" `
        -ArgumentList @(
            "-i", "$DeviceId",
            $query,
            "--format=csv,noheader,nounits",
            "-lms", "$TelemetryIntervalMs"
        ) `
        -RedirectStandardOutput $Path `
        -RedirectStandardError $err `
        -WindowStyle Hidden `
        -PassThru

    return [pscustomobject]@{
        Process = $p
        Path = $Path
        ErrorPath = $err
    }
}

function Stop-Telemetry {
    param($Telemetry)

    if ($null -eq $Telemetry) {
        return
    }

    try {
        if (-not $Telemetry.Process.HasExited) {
            Stop-Process -Id $Telemetry.Process.Id -Force -ErrorAction SilentlyContinue
            $Telemetry.Process.WaitForExit()
        }
    } catch {
    }
}

function Get-Stats {
    param([string]$Path)

    $temps = @()
    $activeTemps = @()
    $activeClocks = @()
    $activeMemoryClocks = @()
    $activePowers = @()
    $fans = @()

    if (Test-Path $Path) {
        foreach ($line in Get-Content $Path) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }

            $p = @($line.Split(",") | ForEach-Object { $_.Trim() })
            if ($p.Count -lt 9) { continue }

            $clock = Parse-DoubleInvariant $p[2]
            $memoryClock = Parse-DoubleInvariant $p[3]
            $temp = Parse-DoubleInvariant $p[4]
            $fan = Parse-DoubleInvariant $p[5]
            $power = Parse-DoubleInvariant $p[6]
            $util = Parse-DoubleInvariant $p[8]

            if ($null -ne $temp) { $temps += $temp }
            if ($null -ne $fan) { $fans += $fan }

            if ($null -ne $util -and $util -ge 50) {
                if ($null -ne $clock) { $activeClocks += $clock }
                if ($null -ne $memoryClock) { $activeMemoryClocks += $memoryClock }
                if ($null -ne $temp) { $activeTemps += $temp }
                if ($null -ne $power) { $activePowers += $power }
            }
        }
    }

    function Min($x) {
        if ($x.Count -eq 0) { return $null }
        return ($x | Measure-Object -Minimum).Minimum
    }

    function Max($x) {
        if ($x.Count -eq 0) { return $null }
        return ($x | Measure-Object -Maximum).Maximum
    }

    function Avg($x) {
        if ($x.Count -eq 0) { return $null }
        return ($x | Measure-Object -Average).Average
    }

    return [pscustomobject]@{
        TempMaxC = Max $temps
        ActiveTempMaxC = Max $activeTemps
        ActiveTempAvgC = Avg $activeTemps
        ActiveClockMinMHz = Min $activeClocks
        ActiveClockMaxMHz = Max $activeClocks
        ActiveClockAvgMHz = Avg $activeClocks
        ActiveMemoryClockMinMHz = Min $activeMemoryClocks
        ActiveMemoryClockMaxMHz = Max $activeMemoryClocks
        ActiveMemoryClockAvgMHz = Avg $activeMemoryClocks
        ActivePowerMaxW = Max $activePowers
        FanMin = Min $fans
        FanMax = Max $fans
    }
}

function Start-Guard {
    param(
        [int]$OffsetMHz,
        [string]$Dir
    )

    $ready = Join-Path $Dir "guard-ready.json"
    $stop = Join-Path $Dir "guard-stop.signal"
    $done = Join-Path $Dir "guard-done.json"
    $log = Join-Path $Dir "guard.log"

    foreach ($p in @($ready, $stop, $done, $log)) {
        Remove-Item $p -ErrorAction SilentlyContinue
    }

    $args = @(
        $guardScript,
        "--gpu", "$DeviceId",
        "--target-voltage-mv", "$VoltageMv",
        "--native-vf-frequency-mhz", "$NativeFrequencyMHz",
        "--fan-percent", "$FanPercent",
        "--fan-tolerance", "3",
        "--curve-offset-mhz", "$OffsetMHz",
        "--cap-band-points", "$CapBandPoints",
        "--parent-pid", "$PID",
        "--ready-file", $ready,
        "--stop-file", $stop,
        "--done-file", $done,
        "--log-file", $log
    )

    $proc = Start-Process `
        -FilePath "python" `
        -ArgumentList $args `
        -WindowStyle Hidden `
        -PassThru

    $deadline = (Get-Date).AddSeconds(30)

    while ((Get-Date) -lt $deadline) {
        if (Test-Path $ready) {
            return [pscustomobject]@{
                Process = $proc
                Ready = Get-Content -Raw -Encoding UTF8 $ready | ConvertFrom-Json
                StopPath = $stop
                DonePath = $done
                LogPath = $log
            }
        }

        if ($proc.HasExited) {
            $raw = ""
            if (Test-Path $log) {
                $t = Get-Content -Raw -Encoding UTF8 $log -ErrorAction SilentlyContinue
                if ($null -ne $t) { $raw = $t.ToString() }
            }
            throw "NVAPI guard failed before READY. ExitCode=$($proc.ExitCode)`n$raw"
        }

        Start-Sleep -Milliseconds 250
    }

    throw "Timed out waiting for NVAPI guard."
}

function Stop-Guard {
    param($Guard)

    if ($null -eq $Guard) { return }

    New-Item -ItemType File -Force -Path $Guard.StopPath | Out-Null

    $deadline = (Get-Date).AddSeconds(30)
    while (-not $Guard.Process.HasExited -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 250
    }

    if (-not $Guard.Process.HasExited) {
        throw "NVAPI guard restore timed out."
    }

    if (-not (Test-Path $Guard.DonePath)) {
        throw "NVAPI guard restore status missing."
    }

    $done = Get-Content -Raw -Encoding UTF8 $Guard.DonePath | ConvertFrom-Json
    if (-not $done.restoreOk) {
        throw "NVAPI guard restore failed."
    }

    if (Test-Path $Guard.LogPath) {
        $raw = Get-Content -Raw -Encoding UTF8 $Guard.LogPath -ErrorAction SilentlyContinue
        if ($null -ne $raw) {
            Add-Line "--- GUARD LOG ---"
            Add-Line $raw.ToString().TrimEnd()
        }
    }

    return $done
}

function Invoke-Replay {
    $out = (
        & node $replayScript $SnapshotPath 1 $DeviceId 2>&1 |
        Out-String
    )

    $line = (
        $out -split "`r?`n" |
        Where-Object { $_ -like "SUNOPT_BLACKWELL_JSON=*" } |
        Select-Object -Last 1
    )

    if (-not $line) {
        throw "Replay did not emit JSON.`n$out"
    }

    $j = $line.Substring("SUNOPT_BLACKWELL_JSON=".Length) | ConvertFrom-Json

    if ([string]$j.checksumHex -ne $ExpectedChecksum) {
        throw "Checksum mismatch: $($j.checksumHex)"
    }

    return [pscustomobject]@{
        WallMs = [double]$j.medianWallMs
        KernelMs = [double]$j.kernelMs[0]
        GpuTotalMs = [double]$j.gpuTotalMs[0]
        Checksum = [string]$j.checksumHex
    }
}

Add-Line "SUNOPT BLACKWELL CORE V/F OFFSET SWEEP"
Add-Line "generated=$(Get-Date -Format o)"
Add-Line "deviceId=$DeviceId"
Add-Line "voltageCapMv=$VoltageMv"
Add-Line "nativeCapFrequencyMHz=$NativeFrequencyMHz"
Add-Line "fanPercent=$FanPercent"
Add-Line "temperatureStopC=$TemperatureStopC"
Add-Line "powerStopW=$PowerStopW"
Add-Line "offsetsMHz=$($offsetsMHz -join ',')"
Add-Line "lastProvenStableOffsetMHz=600"
Add-Line "lastProvenObservedCoreMHz=3457"
Add-Line "expectedMemoryClockMHz=$ExpectedMemoryClockMHz"
Add-Line "memoryClockToleranceMHz=$MemoryClockToleranceMHz"
Add-Line "capBandPoints=$CapBandPoints"
Add-Line "repetitionsPerPoint=$RepetitionsPerPoint"
Add-Line "noVoltageIncrease=YES"
Add-Line "noPowerLimitChange=YES"
Add-Line "noMemoryClockChange=YES"
Add-Line "expectedChecksum=$ExpectedChecksum"
Add-Line ""

try {
    foreach ($offset in $offsetsMHz) {
        if ($stopSweep) { break }

        $dir = Join-Path $resultRoot ("offset-{0}MHz" -f $offset)
        New-Item -ItemType Directory -Force -Path $dir | Out-Null

        Add-Line "================================================================"
        Add-Line ("OFFSET +{0} MHz" -f $offset)
        Add-Line "================================================================"

        $guard = $null
        $telemetry = $null
        $pointStop = ""
        $walls = @()

        try {
            $guard = Start-Guard -OffsetMHz $offset -Dir $dir

            Add-Line "guard=READY"
            Add-Line "guard.curveOffsetMHz=$($guard.Ready.curveOffsetMHz)"
            Add-Line "guard.reportedFanPercent=$($guard.Ready.reportedFanPercent)"
            Add-Line "guard.temperatureC=$($guard.Ready.temperatureC)"

            if ($WarmupRepetitions -gt 0) {
                & node `
                    $replayScript `
                    $SnapshotPath `
                    $WarmupRepetitions `
                    $DeviceId *> $null

                if ($LASTEXITCODE -ne 0) {
                    throw "Warmup failed with exit code $LASTEXITCODE"
                }
                Add-Line "warmup=PASS"
            }

            $telemetryPath = Join-Path $dir "gpu-telemetry.csv"
            $telemetry = Start-Telemetry -Path $telemetryPath
            Start-Sleep -Milliseconds ([math]::Max(250, $TelemetryIntervalMs))

            for ($i = 1; $i -le $RepetitionsPerPoint; $i++) {
                $r = Invoke-Replay
                $walls += $r.WallMs
                Add-Line (
                    "replay={0} wallMs={1:N3} kernelMs={2:N3} gpuTotalMs={3:N3} checksum={4}" -f
                    $i, $r.WallMs, $r.KernelMs, $r.GpuTotalMs, $r.Checksum
                )
            }

            Stop-Telemetry $telemetry
            $telemetry = $null

            $stats = Get-Stats $telemetryPath

            Add-Line "--- TELEMETRY SUMMARY ---"
            Add-Line "temperatureC.max=$($stats.TempMaxC)"
            Add-Line "temperatureC.active.max=$($stats.ActiveTempMaxC)"
            Add-Line "temperatureC.active.avg=$($stats.ActiveTempAvgC)"
            Add-Line "clockMHz.active.min=$($stats.ActiveClockMinMHz)"
            Add-Line "clockMHz.active.max=$($stats.ActiveClockMaxMHz)"
            Add-Line "clockMHz.active.avg=$($stats.ActiveClockAvgMHz)"
            Add-Line "memoryClockMHz.active.min=$($stats.ActiveMemoryClockMinMHz)"
            Add-Line "memoryClockMHz.active.max=$($stats.ActiveMemoryClockMaxMHz)"
            Add-Line "memoryClockMHz.active.avg=$($stats.ActiveMemoryClockAvgMHz)"
            Add-Line "powerW.active.max=$($stats.ActivePowerMaxW)"
            Add-Line "fanPercent.min=$($stats.FanMin)"
            Add-Line "fanPercent.max=$($stats.FanMax)"

            if (
                $null -ne $stats.ActiveClockMinMHz -and
                $null -ne $stats.ActiveClockMaxMHz
            ) {
                $span = (
                    [double]$stats.ActiveClockMaxMHz -
                    [double]$stats.ActiveClockMinMHz
                )
                Add-Line "clockMHz.active.span=$span"
                if ($span -gt $ClockSpanToleranceMHz) {
                    $pointStop = "CLOCK_INSTABILITY"
                }
            } else {
                $pointStop = "NO_ACTIVE_CLOCK_SAMPLES"
            }

            if (
                $null -eq $stats.ActiveMemoryClockMinMHz -or
                $null -eq $stats.ActiveMemoryClockMaxMHz
            ) {
                $pointStop = "NO_ACTIVE_MEMORY_CLOCK_SAMPLES"
            } else {
                $memoryMinAllowed = (
                    $ExpectedMemoryClockMHz - $MemoryClockToleranceMHz
                )
                $memoryMaxAllowed = (
                    $ExpectedMemoryClockMHz + $MemoryClockToleranceMHz
                )

                if (
                    [double]$stats.ActiveMemoryClockMinMHz -lt $memoryMinAllowed -or
                    [double]$stats.ActiveMemoryClockMaxMHz -gt $memoryMaxAllowed
                ) {
                    $pointStop = "MEMORY_CLOCK_CHANGED"
                }
            }

            if (
                $null -ne $stats.TempMaxC -and
                [double]$stats.TempMaxC -ge $TemperatureStopC
            ) {
                $pointStop = "TEMPERATURE_STOP"
            }

            if (
                $null -ne $stats.ActivePowerMaxW -and
                [double]$stats.ActivePowerMaxW -ge $PowerStopW
            ) {
                $pointStop = "POWER_STOP"
            }

            # The whole purpose of this sweep is to raise the *observed* clock.
            # If the driver stops responding to larger V/F deltas, stop there.
            if ($null -ne $previousActiveClock -and $null -ne $stats.ActiveClockAvgMHz) {
                $gain = (
                    [double]$stats.ActiveClockAvgMHz -
                    [double]$previousActiveClock
                )
                Add-Line "clockGainVsPreviousPointMHz=$gain"

                if ($gain -lt 20) {
                    $pointStop = "CLOCK_PLATEAU"
                }
            }

            if ($null -ne $stats.ActiveClockAvgMHz) {
                $previousActiveClock = [double]$stats.ActiveClockAvgMHz
            }

            if ($walls.Count -gt 0) {
                $sorted = @($walls | Sort-Object)
                $median = if (($sorted.Count % 2) -eq 1) {
                    [double]$sorted[[int][math]::Floor($sorted.Count / 2)]
                } else {
                    $m = [int]($sorted.Count / 2)
                    ([double]$sorted[$m - 1] + [double]$sorted[$m]) / 2.0
                }
                Add-Line ("medianWallMs={0:N3}" -f $median)
            }

            Add-Line "--- TELEMETRY TICKS ---"
            if (Test-Path $telemetryPath) {
                foreach ($line in Get-Content $telemetryPath) {
                    Add-Line $line
                }
            }

            if ([string]::IsNullOrWhiteSpace($pointStop)) {
                Add-Line "pointResult=PASS"
            } else {
                Add-Line "pointResult=STOP"
                Add-Line "pointStopReason=$pointStop"
                $stopSweep = $true
                $stopReason = $pointStop
            }

            [void]$results.Add([pscustomobject]@{
                OffsetMHz = $offset
                ClockAvgMHz = $stats.ActiveClockAvgMHz
                ClockMinMHz = $stats.ActiveClockMinMHz
                ClockMaxMHz = $stats.ActiveClockMaxMHz
                MemoryClockMinMHz = $stats.ActiveMemoryClockMinMHz
                MemoryClockMaxMHz = $stats.ActiveMemoryClockMaxMHz
                MemoryClockAvgMHz = $stats.ActiveMemoryClockAvgMHz
                TempMaxC = $stats.TempMaxC
                PowerMaxW = $stats.ActivePowerMaxW
                Stop = $pointStop
            })
        } catch {
            Add-Line "pointResult=ERROR"
            Add-Line "errorType=$($_.Exception.GetType().Name)"
            Add-Line "errorMessage=$($_.Exception.Message)"
            $stopSweep = $true
            $stopReason = "ERROR"
        } finally {
            Stop-Telemetry $telemetry

            if ($null -ne $guard) {
                try {
                    $done = Stop-Guard $guard
                    Add-Line "guardRestore=$($done.status)"
                    Add-Line "guardRestoreOk=$($done.restoreOk)"
                } catch {
                    Add-Line "guardRestore=ERROR"
                    Add-Line "guardRestoreError=$($_.Exception.Message)"
                    $stopSweep = $true
                    $stopReason = "RESTORE_ERROR"
                }
            }
        }

        Add-Line ""
    }
} finally {
    Add-Line "================================================================"
    Add-Line "SWEEP SUMMARY"
    Add-Line "================================================================"

    foreach ($r in $results) {
        Add-Line (
            "offsetMHz={0} coreMHz={1}-{2} avg={3} memoryMHz={4}-{5} avg={6} tempMaxC={7} powerMaxW={8} stop={9}" -f
            $r.OffsetMHz,
            $r.ClockMinMHz,
            $r.ClockMaxMHz,
            $r.ClockAvgMHz,
            $r.MemoryClockMinMHz,
            $r.MemoryClockMaxMHz,
            $r.MemoryClockAvgMHz,
            $r.TempMaxC,
            $r.PowerMaxW,
            $r.Stop
        )
    }

    Add-Line "SWEEP_STOPPED=$($stopSweep)"
    if ($stopSweep) {
        Add-Line "SWEEP_STOP_REASON=$stopReason"
    } else {
        Add-Line "SWEEP_STOP_REASON=NONE"
        Add-Line "OFFSET_CEILING_REACHED=900MHz"
        Add-Line "NEXT_STEP=Evaluate stability/performance before extending the offset ceiling."
    }

    Add-Line "generatedEnd=$(Get-Date -Format o)"
}

Write-Host ""
Write-Host "Core V/F offset sweep finished." -ForegroundColor Green
Write-Host "Report: $OutputPath"

param(
    [int]$DeviceId = 0,
    [string]$SnapshotPath = "",
    [string]$OutputPath = "",
    [int]$FanPercent = 70,
    [int]$FanTolerancePercent = 3,
    [int]$TemperatureStopC = 60,
    [double]$PowerStopW = 140.0,
    [int]$RepetitionsPerPoint = 5,
    [int]$WarmupRepetitions = 1,
    [int]$TelemetryIntervalMs = 250,
    [int]$ClockSpanToleranceMHz = 15,
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
    $OutputPath = Join-Path $root "llm-vf-sweep.txt"
}

$guardScript = Join-Path $scriptDir "nvapi-blackwell-benchmark-guard.py"
$replayScript = Join-Path $scriptDir "blackwell-replay.mjs"
$resultRoot = Join-Path $root "blackwell-vf-sweep-results"

New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

if (-not (Test-Path $guardScript)) {
    throw "NVAPI guard not found: $guardScript"
}

if (-not (Test-Path $replayScript)) {
    throw "Replay script not found: $replayScript"
}

if (-not (Test-Path $SnapshotPath)) {
    throw "Replay snapshot not found: $SnapshotPath"
}

# Coarse native V/F sweep, all points observed directly in this RTX 5060's curve.
# We deliberately stop at the highest verified native core point.
$points = @(
    [pscustomobject]@{ Slot = 96;  VoltageMv = 1050; FrequencyMHz = 2865 },
    [pscustomobject]@{ Slot = 104; VoltageMv = 1100; FrequencyMHz = 2955 },
    [pscustomobject]@{ Slot = 112; VoltageMv = 1150; FrequencyMHz = 3037 },
    [pscustomobject]@{ Slot = 120; VoltageMv = 1200; FrequencyMHz = 3120 },
    [pscustomobject]@{ Slot = 126; VoltageMv = 1240; FrequencyMHz = 3172 }
)

$report = New-Object System.Collections.Generic.List[string]
$results = New-Object System.Collections.Generic.List[object]
$stopSweep = $false
$stopReason = ""

function Add-ReportLine {
    param(
        [AllowEmptyString()]
        [string]$Text = ""
    )

    [void]$script:report.Add($Text)
    Set-Content -Encoding UTF8 -Path $OutputPath -Value $script:report
}

function Parse-DoubleInvariant {
    param([string]$Text)

    $v = 0.0
    $ok = [double]::TryParse(
        $Text,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$v
    )

    if (-not $ok) {
        return $null
    }

    return $v
}

function Get-GpuSample {
    $query = (
        "--query-gpu=" +
        "timestamp,pstate,clocks.gr,clocks.mem,temperature.gpu," +
        "fan.speed,power.draw,power.limit,utilization.gpu"
    )

    $line = (
        & nvidia-smi `
            -i $DeviceId `
            $query `
            "--format=csv,noheader,nounits" 2>$null |
        Select-Object -First 1
    )

    if ($null -eq $line) {
        return $null
    }

    $parts = @($line.ToString().Split(",") | ForEach-Object { $_.Trim() })

    if ($parts.Count -lt 9) {
        return $null
    }

    return [pscustomobject]@{
        Timestamp = $parts[0]
        Pstate = $parts[1]
        ClockMHz = Parse-DoubleInvariant $parts[2]
        MemoryClockMHz = Parse-DoubleInvariant $parts[3]
        TemperatureC = Parse-DoubleInvariant $parts[4]
        FanPercent = Parse-DoubleInvariant $parts[5]
        PowerW = Parse-DoubleInvariant $parts[6]
        PowerLimitW = Parse-DoubleInvariant $parts[7]
        UtilPercent = Parse-DoubleInvariant $parts[8]
        Raw = $line.ToString()
    }
}

function Start-Telemetry {
    param([string]$Path)

    $stderr = "$Path.stderr.txt"

    Remove-Item $Path -ErrorAction SilentlyContinue
    Remove-Item $stderr -ErrorAction SilentlyContinue

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

    $p = Start-Process `
        -FilePath "nvidia-smi" `
        -ArgumentList $args `
        -RedirectStandardOutput $Path `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru

    return [pscustomobject]@{
        Process = $p
        Path = $Path
        ErrorPath = $stderr
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
        # Already exited.
    }
}

function Get-TelemetryStats {
    param([string]$Path)

    $temps = New-Object System.Collections.Generic.List[double]
    $clocks = New-Object System.Collections.Generic.List[double]
    $fans = New-Object System.Collections.Generic.List[double]
    $powers = New-Object System.Collections.Generic.List[double]
    $utils = New-Object System.Collections.Generic.List[double]

    if (Test-Path $Path) {
        foreach ($line in (Get-Content $Path)) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            $parts = @($line.Split(",") | ForEach-Object { $_.Trim() })

            if ($parts.Count -lt 9) {
                continue
            }

            $clock = Parse-DoubleInvariant $parts[2]
            $temp = Parse-DoubleInvariant $parts[4]
            $fan = Parse-DoubleInvariant $parts[5]
            $power = Parse-DoubleInvariant $parts[6]
            $util = Parse-DoubleInvariant $parts[8]

            if ($null -ne $temp) { [void]$temps.Add($temp) }
            if ($null -ne $clock) { [void]$clocks.Add($clock) }
            if ($null -ne $fan) { [void]$fans.Add($fan) }
            if ($null -ne $power) { [void]$powers.Add($power) }
            if ($null -ne $util) { [void]$utils.Add($util) }
        }
    }

    $activeClocks = New-Object System.Collections.Generic.List[double]
    $activeTemps = New-Object System.Collections.Generic.List[double]
    $activePowers = New-Object System.Collections.Generic.List[double]

    if (Test-Path $Path) {
        foreach ($line in (Get-Content $Path)) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            $parts = @($line.Split(",") | ForEach-Object { $_.Trim() })

            if ($parts.Count -lt 9) {
                continue
            }

            $util = Parse-DoubleInvariant $parts[8]
            if ($null -eq $util -or $util -lt 50) {
                continue
            }

            $clock = Parse-DoubleInvariant $parts[2]
            $temp = Parse-DoubleInvariant $parts[4]
            $power = Parse-DoubleInvariant $parts[6]

            if ($null -ne $clock) { [void]$activeClocks.Add($clock) }
            if ($null -ne $temp) { [void]$activeTemps.Add($temp) }
            if ($null -ne $power) { [void]$activePowers.Add($power) }
        }
    }

    function Stat-Min($v) {
        if ($v.Count -eq 0) { return $null }
        return ($v | Measure-Object -Minimum).Minimum
    }

    function Stat-Max($v) {
        if ($v.Count -eq 0) { return $null }
        return ($v | Measure-Object -Maximum).Maximum
    }

    function Stat-Avg($v) {
        if ($v.Count -eq 0) { return $null }
        return ($v | Measure-Object -Average).Average
    }

    return [pscustomobject]@{
        Samples = $temps.Count
        ActiveSamples = $activeTemps.Count
        TempMinC = Stat-Min $temps
        TempMaxC = Stat-Max $temps
        TempAvgC = Stat-Avg $temps
        ActiveTempMinC = Stat-Min $activeTemps
        ActiveTempMaxC = Stat-Max $activeTemps
        ActiveTempAvgC = Stat-Avg $activeTemps
        ActiveClockMinMHz = Stat-Min $activeClocks
        ActiveClockMaxMHz = Stat-Max $activeClocks
        ActiveClockAvgMHz = Stat-Avg $activeClocks
        ActivePowerMaxW = Stat-Max $activePowers
        FanMinPercent = Stat-Min $fans
        FanMaxPercent = Stat-Max $fans
        FanAvgPercent = Stat-Avg $fans
        UtilAvgPercent = Stat-Avg $utils
    }
}

function Start-Guard {
    param(
        [int]$VoltageMv,
        [int]$FrequencyMHz,
        [string]$Dir
    )

    $ready = Join-Path $Dir "guard-ready.json"
    $stop = Join-Path $Dir "guard-stop.signal"
    $done = Join-Path $Dir "guard-done.json"
    $log = Join-Path $Dir "guard.log"

    Remove-Item $ready -ErrorAction SilentlyContinue
    Remove-Item $stop -ErrorAction SilentlyContinue
    Remove-Item $done -ErrorAction SilentlyContinue
    Remove-Item $log -ErrorAction SilentlyContinue

    $args = @(
        $guardScript,
        "--gpu", "$DeviceId",
        "--target-voltage-mv", "$VoltageMv",
        "--native-vf-frequency-mhz", "$FrequencyMHz",
        "--fan-percent", "$FanPercent",
        "--fan-tolerance", "$FanTolerancePercent",
        "--parent-pid", "$PID",
        "--ready-file", $ready,
        "--stop-file", $stop,
        "--done-file", $done,
        "--log-file", $log
    )

    $p = Start-Process `
        -FilePath "python" `
        -ArgumentList $args `
        -WindowStyle Hidden `
        -PassThru

    $deadline = (Get-Date).AddSeconds(30)

    while ((Get-Date) -lt $deadline) {
        if (Test-Path $ready) {
            $readyData = Get-Content -Raw -Encoding UTF8 $ready | ConvertFrom-Json

            return [pscustomobject]@{
                Process = $p
                ReadyPath = $ready
                StopPath = $stop
                DonePath = $done
                LogPath = $log
                ReadyData = $readyData
            }
        }

        if ($p.HasExited) {
            $guardLog = ""
            if (Test-Path $log) {
                $raw = Get-Content -Raw -Encoding UTF8 $log -ErrorAction SilentlyContinue
                if ($null -ne $raw) {
                    $guardLog = $raw.ToString()
                }
            }

            throw "NVAPI guard exited before READY. ExitCode=$($p.ExitCode)`n$guardLog"
        }

        Start-Sleep -Milliseconds 250
    }

    throw "Timed out waiting for NVAPI guard."
}

function Stop-Guard {
    param($Guard)

    if ($null -eq $Guard) {
        return
    }

    try {
        New-Item -ItemType File -Force -Path $Guard.StopPath | Out-Null

        $deadline = (Get-Date).AddSeconds(30)

        while (-not $Guard.Process.HasExited -and (Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 250
        }

        if (-not $Guard.Process.HasExited) {
            throw "NVAPI guard restore timed out."
        }

        if (-not (Test-Path $Guard.DonePath)) {
            throw "NVAPI guard did not produce restore status."
        }

        $done = Get-Content -Raw -Encoding UTF8 $Guard.DonePath | ConvertFrom-Json

        if (-not $done.restoreOk) {
            throw "NVAPI guard reported restore failure."
        }

        return $done
    } finally {
        if (Test-Path $Guard.LogPath) {
            $raw = Get-Content -Raw -Encoding UTF8 $Guard.LogPath -ErrorAction SilentlyContinue

            if ($null -ne $raw) {
                Add-ReportLine "--- GUARD LOG ---"
                Add-ReportLine $raw.ToString().TrimEnd()
            }
        }
    }
}

function Invoke-ReplayOnce {
    $out = (
        & node `
            $replayScript `
            $SnapshotPath `
            1 `
            $DeviceId 2>&1 |
        Out-String
    )

    $jsonLine = (
        $out -split "`r?`n" |
        Where-Object { $_ -like "SUNOPT_BLACKWELL_JSON=*" } |
        Select-Object -Last 1
    )

    if (-not $jsonLine) {
        throw "Replay did not emit SUNOPT_BLACKWELL_JSON.`n$out"
    }

    $j = $jsonLine.Substring("SUNOPT_BLACKWELL_JSON=".Length) | ConvertFrom-Json

    if ([string]$j.checksumHex -ne $ExpectedChecksum) {
        throw (
            "Checksum mismatch. Expected $ExpectedChecksum, " +
            "got $($j.checksumHex)."
        )
    }

    return [pscustomobject]@{
        WallMs = [double]$j.medianWallMs
        KernelMs = [double]$j.kernelMs[0]
        GpuTotalMs = [double]$j.gpuTotalMs[0]
        Checksum = [string]$j.checksumHex
        Raw = $out.TrimEnd()
    }
}

Add-ReportLine "SUNOPT BLACKWELL NATIVE V/F TEMPERATURE SWEEP"
Add-ReportLine "generated=$(Get-Date -Format o)"
Add-ReportLine "deviceId=$DeviceId"
Add-ReportLine "snapshot=$SnapshotPath"
Add-ReportLine "snapshotSha256=$((Get-FileHash -Algorithm SHA256 $SnapshotPath).Hash.ToLowerInvariant())"
Add-ReportLine "fanPercent=$FanPercent"
Add-ReportLine "temperatureStopC=$TemperatureStopC"
Add-ReportLine "powerStopW=$PowerStopW"
Add-ReportLine "repetitionsPerPoint=$RepetitionsPerPoint"
Add-ReportLine "warmupRepetitions=$WarmupRepetitions"
Add-ReportLine "expectedChecksum=$ExpectedChecksum"
Add-ReportLine "mode=native-vf-points-only"
Add-ReportLine "noFrequencyOffset=YES"
Add-ReportLine "noPowerLimitChange=YES"
Add-ReportLine "noMemoryClockChange=YES"
Add-ReportLine ""

$addon = Get-ChildItem $root -File -Filter *.node |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -ne $addon) {
    Add-ReportLine "addon=$($addon.FullName)"
    Add-ReportLine "addonSha256=$((Get-FileHash -Algorithm SHA256 $addon.FullName).Hash.ToLowerInvariant())"
}

Add-ReportLine ""

try {
    foreach ($point in $points) {
        if ($stopSweep) {
            break
        }

        $label = "slot-$($point.Slot)-$($point.VoltageMv)mV-$($point.FrequencyMHz)MHz"
        $dir = Join-Path $resultRoot $label
        New-Item -ItemType Directory -Force -Path $dir | Out-Null

        Add-ReportLine "================================================================"
        Add-ReportLine "POINT $label"
        Add-ReportLine "================================================================"
        Add-ReportLine "nativeSlot=$($point.Slot)"
        Add-ReportLine "targetVoltageMv=$($point.VoltageMv)"
        Add-ReportLine "nativeFrequencyMHz=$($point.FrequencyMHz)"

        $guard = $null
        $telemetry = $null
        $pointStop = ""
        $wall = New-Object System.Collections.Generic.List[double]

        try {
            $guard = Start-Guard `
                -VoltageMv $point.VoltageMv `
                -FrequencyMHz $point.FrequencyMHz `
                -Dir $dir

            Add-ReportLine "guard=READY"
            Add-ReportLine "guard.nativeVfSlot=$($guard.ReadyData.nativeVfSlot)"
            Add-ReportLine "guard.reportedFanPercent=$($guard.ReadyData.reportedFanPercent)"
            Add-ReportLine "guard.temperatureC=$($guard.ReadyData.temperatureC)"

            if ($WarmupRepetitions -gt 0) {
                Add-ReportLine "warmup.begin=$WarmupRepetitions"

                & node `
                    $replayScript `
                    $SnapshotPath `
                    $WarmupRepetitions `
                    $DeviceId *> $null

                if ($LASTEXITCODE -ne 0) {
                    throw "Warmup replay failed with exit code $LASTEXITCODE."
                }

                Add-ReportLine "warmup=PASS"
            }

            $telemetryPath = Join-Path $dir "gpu-telemetry.csv"
            $telemetry = Start-Telemetry -Path $telemetryPath

            Start-Sleep -Milliseconds ([math]::Max(250, $TelemetryIntervalMs))

            for ($i = 1; $i -le $RepetitionsPerPoint; $i++) {
                $r = Invoke-ReplayOnce
                [void]$wall.Add($r.WallMs)

                $sample = Get-GpuSample

                Add-ReportLine (
                    "replay=$i wallMs={0:N3} kernelMs={1:N3} gpuTotalMs={2:N3} checksum={3}" -f
                    $r.WallMs,
                    $r.KernelMs,
                    $r.GpuTotalMs,
                    $r.Checksum
                )

                if ($null -ne $sample) {
                    Add-ReportLine (
                        "postReplay=$i clockMHz={0} tempC={1} fanPercent={2} powerW={3} utilPercent={4}" -f
                        $sample.ClockMHz,
                        $sample.TemperatureC,
                        $sample.FanPercent,
                        $sample.PowerW,
                        $sample.UtilPercent
                    )

                    if (
                        $null -ne $sample.TemperatureC -and
                        $sample.TemperatureC -ge $TemperatureStopC
                    ) {
                        $pointStop = "TEMPERATURE_STOP"
                        break
                    }

                    if (
                        $null -ne $sample.PowerW -and
                        $sample.PowerW -ge $PowerStopW
                    ) {
                        $pointStop = "POWER_STOP"
                        break
                    }
                }
            }

            Stop-Telemetry -Telemetry $telemetry
            $telemetry = $null

            $stats = Get-TelemetryStats -Path $telemetryPath

            Add-ReportLine "--- TELEMETRY SUMMARY ---"
            Add-ReportLine "samples=$($stats.Samples)"
            Add-ReportLine "activeSamples=$($stats.ActiveSamples)"
            Add-ReportLine "temperatureC.min=$($stats.TempMinC)"
            Add-ReportLine "temperatureC.max=$($stats.TempMaxC)"
            Add-ReportLine "temperatureC.avg=$($stats.TempAvgC)"
            Add-ReportLine "temperatureC.active.min=$($stats.ActiveTempMinC)"
            Add-ReportLine "temperatureC.active.max=$($stats.ActiveTempMaxC)"
            Add-ReportLine "temperatureC.active.avg=$($stats.ActiveTempAvgC)"
            Add-ReportLine "clockMHz.active.min=$($stats.ActiveClockMinMHz)"
            Add-ReportLine "clockMHz.active.max=$($stats.ActiveClockMaxMHz)"
            Add-ReportLine "clockMHz.active.avg=$($stats.ActiveClockAvgMHz)"
            Add-ReportLine "powerW.active.max=$($stats.ActivePowerMaxW)"
            Add-ReportLine "fanPercent.min=$($stats.FanMinPercent)"
            Add-ReportLine "fanPercent.max=$($stats.FanMaxPercent)"
            Add-ReportLine "fanPercent.avg=$($stats.FanAvgPercent)"
            Add-ReportLine "utilizationPercent.avg=$($stats.UtilAvgPercent)"

            if (
                $stats.ActiveClockMinMHz -ne $null -and
                $stats.ActiveClockMaxMHz -ne $null
            ) {
                $clockSpan = [double]$stats.ActiveClockMaxMHz - [double]$stats.ActiveClockMinMHz
                Add-ReportLine "clockMHz.active.span=$clockSpan"

                if ($clockSpan -gt $ClockSpanToleranceMHz) {
                    $pointStop = "CLOCK_INSTABILITY"
                }
            }

            if (
                $stats.TempMaxC -ne $null -and
                [double]$stats.TempMaxC -ge $TemperatureStopC
            ) {
                $pointStop = "TEMPERATURE_STOP"
            }

            if (
                $stats.ActivePowerMaxW -ne $null -and
                [double]$stats.ActivePowerMaxW -ge $PowerStopW
            ) {
                $pointStop = "POWER_STOP"
            }

            if ($wall.Count -gt 0) {
                $sorted = @($wall | Sort-Object)
                $mid = [int][math]::Floor($sorted.Count / 2)

                if (($sorted.Count % 2) -eq 1) {
                    $median = [double]$sorted[$mid]
                } else {
                    $median = ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
                }

                Add-ReportLine ("medianWallMs={0:N3}" -f $median)
            }

            Add-ReportLine "--- TELEMETRY TICKS ---"

            if (Test-Path $telemetryPath) {
                foreach ($line in (Get-Content $telemetryPath)) {
                    Add-ReportLine $line
                }
            }

            if ([string]::IsNullOrWhiteSpace($pointStop)) {
                Add-ReportLine "pointResult=PASS"
            } else {
                Add-ReportLine "pointResult=STOP"
                Add-ReportLine "pointStopReason=$pointStop"
                $stopSweep = $true
                $stopReason = $pointStop
            }

            [void]$results.Add([pscustomobject]@{
                Slot = $point.Slot
                VoltageMv = $point.VoltageMv
                FrequencyMHz = $point.FrequencyMHz
                TempMaxC = $stats.TempMaxC
                ActiveClockMinMHz = $stats.ActiveClockMinMHz
                ActiveClockMaxMHz = $stats.ActiveClockMaxMHz
                ActiveClockAvgMHz = $stats.ActiveClockAvgMHz
                ActivePowerMaxW = $stats.ActivePowerMaxW
                StopReason = $pointStop
            })
        } catch {
            Add-ReportLine "pointResult=ERROR"
            Add-ReportLine "errorType=$($_.Exception.GetType().Name)"
            Add-ReportLine "errorMessage=$($_.Exception.Message)"
            $stopSweep = $true
            $stopReason = "ERROR"
        } finally {
            Stop-Telemetry -Telemetry $telemetry

            if ($null -ne $guard) {
                try {
                    $done = Stop-Guard -Guard $guard
                    Add-ReportLine "guardRestore=$($done.status)"
                    Add-ReportLine "guardRestoreOk=$($done.restoreOk)"
                } catch {
                    Add-ReportLine "guardRestore=ERROR"
                    Add-ReportLine "guardRestoreError=$($_.Exception.Message)"
                    $stopSweep = $true
                    $stopReason = "RESTORE_ERROR"
                }
            }
        }

        Add-ReportLine ""
    }
} finally {
    Add-ReportLine "================================================================"
    Add-ReportLine "SWEEP SUMMARY"
    Add-ReportLine "================================================================"

    foreach ($r in $results) {
        Add-ReportLine (
            "slot={0} voltageMv={1} nativeMHz={2} tempMaxC={3} " +
            "clockActiveMHz={4}-{5} avg={6} powerMaxW={7} stop={8}" -f
            $r.Slot,
            $r.VoltageMv,
            $r.FrequencyMHz,
            $r.TempMaxC,
            $r.ActiveClockMinMHz,
            $r.ActiveClockMaxMHz,
            $r.ActiveClockAvgMHz,
            $r.ActivePowerMaxW,
            $r.StopReason
        )
    }

    if ($stopSweep) {
        Add-ReportLine "SWEEP_STOPPED=YES"
        Add-ReportLine "SWEEP_STOP_REASON=$stopReason"
    } else {
        Add-ReportLine "SWEEP_STOPPED=NO"
        Add-ReportLine "NATIVE_CURVE_MAX_REACHED=YES"
        Add-ReportLine "NEXT_STEP=Do not increase further automatically; evaluate positive frequency offset separately."
    }

    Add-ReportLine "generatedEnd=$(Get-Date -Format o)"
}

Write-Host ""
Write-Host "Native V/F sweep finished." -ForegroundColor Green
Write-Host "Report: $OutputPath"

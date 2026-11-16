$ErrorActionPreference = "Stop"

$Root = "C:\NextJS\SunOpt\sunopt"
$NativeRoot = Join-Path $Root "rust\sunopt-node"
$PackageName = "20261106-blackwell-annual-graph-replay.zip"
$ExpectedParent = "e115df7c8c18a10021969b2637a8c5840d5d15acdd1904689d5473651a25319b"
$ExpectedRelease = "57faaf90ded5256577a062a4bba3a5df64d50e442556bd03eff855d1ba178de5"

Set-Location $Root

Write-Host "=== 1. Stop competing SunOpt update watchers ==="
$watchers = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^(python|pythonw)(\.exe)?$' -and
            $_.CommandLine -match 'update_watcher\.py'
        }
)

foreach ($watcher in $watchers) {
    Write-Host "Stopping watcher PID $($watcher.ProcessId): $($watcher.CommandLine)"
    Stop-Process -Id $watcher.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Milliseconds 500

Write-Host "=== 2. Remove stale atomic-write temp files ==="
$staleFiles = @(
    (Join-Path $Root "rust\sunopt-node\cuda\cuda_context.h.updating"),
    (Join-Path $Root "rust\sunopt-node\cuda\cuda_context.cu.updating"),
    (Join-Path $Root "rust\sunopt-node\src\cuda_context.rs.updating"),
    (Join-Path $Root "rust\sunopt-node\scripts\blackwell-graph-replay.mjs.updating"),
    (Join-Path $Root "updates\installed-release.json.updating"),
    (Join-Path $Root "updates\history\hash.txt.updating")
)

foreach ($path in $staleFiles) {
    if (Test-Path -LiteralPath $path) {
        Write-Host "Removing $path"
        Remove-Item -LiteralPath $path -Force
    }
}

function Get-InstalledReleaseHash {
    $releaseFile = Join-Path $Root "updates\installed-release.json"
    if (-not (Test-Path -LiteralPath $releaseFile)) {
        return ("0" * 64)
    }

    return ((Get-Content -LiteralPath $releaseFile -Raw | ConvertFrom-Json).sha256).ToLowerInvariant()
}

function Find-RejectedPackage {
    $rejected = Join-Path $Root "updates\rejected"
    if (-not (Test-Path -LiteralPath $rejected)) {
        return $null
    }

    $items = @(
        Get-ChildItem -LiteralPath $rejected -File |
            Where-Object { $_.Name -like "*$PackageName" } |
            Sort-Object LastWriteTime -Descending
    )

    if ($items.Count -eq 0) {
        return $null
    }

    return $items[0].FullName
}

function Requeue-Package {
    $inbox = Join-Path $Root "updates\inbox"
    New-Item -ItemType Directory -Path $inbox -Force | Out-Null

    $destination = Join-Path $inbox $PackageName
    if (Test-Path -LiteralPath $destination) {
        return $destination
    }

    $source = Find-RejectedPackage
    if ($null -eq $source) {
        throw "Could not find $PackageName in updates\rejected. Re-run make_blackwell_annual_graph_replay_update.py first."
    }

    Write-Host "Requeueing $source"
    Copy-Item -LiteralPath $source -Destination $destination -Force
    return $destination
}

Write-Host "=== 3. Verify hash-chain state ==="
$currentHash = Get-InstalledReleaseHash
Write-Host "Installed release: $currentHash"

if ($currentHash -eq $ExpectedRelease) {
    Write-Host "Graph replay package is already installed."
}
elseif ($currentHash -ne $ExpectedParent) {
    throw "Unexpected installed release. Expected parent $ExpectedParent or installed release $ExpectedRelease, got $currentHash. Refusing to alter the hash chain."
}
else {
    Write-Host "=== 4. Reinstall rejected package with retry ==="

    $installed = $false

    for ($attempt = 1; $attempt -le 5; $attempt++) {
        Write-Host "Install attempt $attempt/5"
        Requeue-Package | Out-Null

        & python ".\tools\update_watcher.py" --once

        $currentHash = Get-InstalledReleaseHash
        $scriptPath = Join-Path $NativeRoot "scripts\blackwell-graph-replay.mjs"

        if (
            $currentHash -eq $ExpectedRelease -and
            (Test-Path -LiteralPath $scriptPath)
        ) {
            $installed = $true
            break
        }

        if ($currentHash -ne $ExpectedParent) {
            throw "Watcher left unexpected release state $currentHash after failed install."
        }

        Start-Sleep -Seconds 1
    }

    if (-not $installed) {
        throw "Package could not be installed after 5 attempts. A process other than update_watcher is persistently locking cuda_context.rs."
    }
}

Write-Host "=== 5. Locate Visual Studio C++ toolchain ==="
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe not found at $vswhere"
}

$vsInstall = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
if ([string]::IsNullOrWhiteSpace($vsInstall)) {
    throw "No Visual Studio installation with the x64 C++ toolchain was found."
}

$devCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $devCmd)) {
    throw "VsDevCmd.bat not found: $devCmd"
}

Write-Host "Visual Studio: $vsInstall"
Write-Host "=== 6. Build sm_120f and run annual graph replay ==="

$env:SUNOPT_CUDA_ARCH = "sm_120f"

$buildCommand = (
    'call "' + $devCmd + '" -arch=x64 -host_arch=x64' +
    ' && cd /d "' + $NativeRoot + '"' +
    ' && where cl.exe' +
    ' && npm run build' +
    ' && node .\scripts\blackwell-graph-replay.mjs .\blackwell-benchmark.bin 3 0'
)

& $env:ComSpec /d /s /c $buildCommand
if ($LASTEXITCODE -ne 0) {
    throw "Build or graph replay failed with exit code $LASTEXITCODE"
}

Write-Host "=== DONE ==="
Write-Host "Expected parity checksum from the previous Blackwell replay: 218451794297e021"

# rust/sunopt-node/install-protected-tools.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$SourceRoot = Join-Path $ProjectRoot "rust\sunopt-node\protected"
$ToolsRoot = Join-Path $ProjectRoot "tools"

if (-not (Test-Path $SourceRoot)) {
    throw "Protected source directory not found: $SourceRoot"
}

if (-not (Test-Path $ToolsRoot)) {
    New-Item -ItemType Directory -Path $ToolsRoot | Out-Null
}

$Mappings = @(
    @{
        Source = Join-Path $SourceRoot "enable-native-npm-hooks.mjs"
        Destination = Join-Path $ToolsRoot "enable-native-npm-hooks.mjs"
    }
)

foreach ($Mapping in $Mappings) {
    if (-not (Test-Path $Mapping.Source)) {
        throw "Protected source file not found: $($Mapping.Source)"
    }

    Copy-Item `
        -LiteralPath $Mapping.Source `
        -Destination $Mapping.Destination `
        -Force

    $SourceHash =
        (Get-FileHash -Algorithm SHA256 -LiteralPath $Mapping.Source).Hash

    $DestinationHash =
        (Get-FileHash -Algorithm SHA256 -LiteralPath $Mapping.Destination).Hash

    if ($SourceHash -ne $DestinationHash) {
        throw "SHA-256 verification failed for $($Mapping.Destination)"
    }

    Write-Host "[SunOpt] Installed protected file:" $Mapping.Destination
    Write-Host "[SunOpt] SHA-256:" $DestinationHash
}

Write-Host "[SunOpt] Protected tools installation complete."

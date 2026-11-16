param()

$ErrorActionPreference = "Stop"

$root = (Get-Location).Path
$runner = Join-Path $root "rust\sunopt-node\scripts\blackwell-b14i1-lockstep.mjs"
$outbox = Join-Path $root "updates\outbox"
$report = Join-Path $outbox "b14i1-runner-fix.txt"

New-Item -ItemType Directory -Force -Path $outbox | Out-Null

if (-not (Test-Path $runner)) {
    throw "Runner not found: $runner"
}

$text = [System.IO.File]::ReadAllText($runner)

$beforeBackticks = ([regex]::Matches($text, '\\`')).Count
$beforeInterpolations = ([regex]::Matches($text, '\\\$\{')).Count

# String.raw in the generator preserved the JavaScript escaping.
# In the generated runner these backslashes are invalid syntax.
$text = $text.Replace('\`', '`')
$text = $text.Replace('\${', '${')

[System.IO.File]::WriteAllText(
    $runner,
    $text,
    [System.Text.UTF8Encoding]::new($false)
)

$afterBackticks = ([regex]::Matches($text, '\\`')).Count
$afterInterpolations = ([regex]::Matches($text, '\\\$\{')).Count

@"
B14I1 RUNNER FIX
runner=$runner
escapedBackticksBefore=$beforeBackticks
escapedInterpolationsBefore=$beforeInterpolations
escapedBackticksAfter=$afterBackticks
escapedInterpolationsAfter=$afterInterpolations
B14I1_RUNNER_FIX_RESULT=OK
"@ | Set-Content -Encoding UTF8 $report

Get-Content $report

if ($afterBackticks -ne 0 -or $afterInterpolations -ne 0) {
    throw "Escaped JavaScript template syntax remains in the runner."
}

Write-Host ""
Write-Host "Now run:"
Write-Host "  node .\rust\sunopt-node\scripts\blackwell-b14i1-lockstep.mjs"
Write-Host ""
Write-Host "Benchmark report:"
Write-Host "  updates\outbox\blackwell-b14i1-lockstep.json"

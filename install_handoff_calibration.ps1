$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$Source = Join-Path $PSScriptRoot "calibration\so101-follower.json"
if (-not (Test-Path -LiteralPath $Source)) {
    throw "The handoff package does not contain calibration\so101-follower.json."
}

$TargetDirectory = Join-Path $env:USERPROFILE `
    ".cache\huggingface\lerobot\calibration\robots\so_follower"
$Target = Join-Path $TargetDirectory "so101-follower.json"
New-Item -ItemType Directory -Path $TargetDirectory -Force | Out-Null

if (Test-Path -LiteralPath $Target) {
    $Backup = "$Target.backup-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    Copy-Item -LiteralPath $Target -Destination $Backup
    Write-Host "Existing calibration backed up to: $Backup" -ForegroundColor Yellow
}

Copy-Item -LiteralPath $Source -Destination $Target -Force
Write-Host "Installed calibration: $Target" -ForegroundColor Green
Write-Host "Only use this calibration for the matching physical SO101 arm." -ForegroundColor Yellow

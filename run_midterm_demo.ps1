param(
    [switch]$Lan,
    [switch]$HighSensitivity,
    [switch]$Smooth
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv_perception\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv_perception. Install requirements-perception.txt first."
}

$Status = Invoke-RestMethod -TimeoutSec 3 http://127.0.0.1:8000/api/status
if (-not $Status.robot.connected) {
    throw "Robot is not connected. Start the robot service first."
}
if ($Status.robot.stopped) {
    throw "Emergency stop is latched. Restart the robot service before the demo."
}

$WebHost = if ($Lan) { "0.0.0.0" } else { "127.0.0.1" }
if ($Smooth) {
    $DemoTitle = "JARVIS Midterm Demo - Smooth Tracking"
    $FaceConfidence = 0.6
    $HandConfidence = 0.35
    $ServoEndpoint = "http://127.0.0.1:8000/api/servo/smooth"
    $ServoRate = 10.0
    $ServoPanGain = 8.0
    $ServoDeadband = 0.07
    $ServoMaxStep = 1.5
    $ServoStepDeg = 1.5
    $ServoVerticalRate = 10.0
    $ServoVerticalDeadband = 0.06
    $ServoVerticalMaxStep = 1.5
    $ServoVerticalMinStep = 0.0
    $PointingPanGain = 12.0
    $PointingVerticalGain = 10.0
    $PointingMaxStep = 2.0
    $PointingStepDeg = 2.0
    $PointingDeadband = 0.08
    $PointingConfidence = 0.55
    $PointingRate = 5.0
    $PointingVerticalRate = 5.0
    $PointingSettle = 0.35
} elseif ($HighSensitivity) {
    $DemoTitle = "JARVIS Midterm Demo - High Sensitivity"
    $FaceConfidence = 0.55
    $HandConfidence = 0.4
    $ServoEndpoint = "http://127.0.0.1:8000/api/servo"
    $ServoRate = 3.0
    $ServoPanGain = 14.0
    $ServoDeadband = 0.05
    $ServoMaxStep = 3.0
    $ServoStepDeg = 3.0
    $ServoVerticalRate = 0.5
    $ServoVerticalDeadband = 0.15
    $ServoVerticalMaxStep = 6.0
    $ServoVerticalMinStep = 6.0
    $PointingPanGain = 16.0
    $PointingVerticalGain = 10.0
    $PointingMaxStep = 6.0
    $PointingStepDeg = 3.0
    $PointingDeadband = 0.12
    $PointingConfidence = 0.55
    $PointingRate = 1.2
    $PointingVerticalRate = 0.25
    $PointingSettle = 0.6
} else {
    $DemoTitle = "JARVIS Midterm Demo"
    $FaceConfidence = 0.5
    $HandConfidence = 0.5
    $ServoEndpoint = "http://127.0.0.1:8000/api/servo"
    $ServoRate = 2.5
    $ServoPanGain = 10.0
    $ServoDeadband = 0.08
    $ServoMaxStep = 3.0
    $ServoStepDeg = 3.0
    $ServoVerticalRate = 0.35
    $ServoVerticalDeadband = 0.2
    $ServoVerticalMaxStep = 6.0
    $ServoVerticalMinStep = 6.0
    $PointingPanGain = 12.0
    $PointingVerticalGain = 8.0
    $PointingMaxStep = 6.0
    $PointingStepDeg = 3.0
    $PointingDeadband = 0.18
    $PointingConfidence = 0.62
    $PointingRate = 1.0
    $PointingVerticalRate = 0.2
    $PointingSettle = 0.8
}

if ($Lan) {
    Write-Host "LAN mode: phone and PC must be on the same Wi-Fi." -ForegroundColor Cyan
    Write-Host "Use the PC IPv4 address and port 8768 in the phone browser." -ForegroundColor Cyan
    ipconfig | Select-String "IPv4"
} else {
    Write-Host "JARVIS midterm demo: http://127.0.0.1:8768/" -ForegroundColor Cyan
}
$ModeLabel = if ($Smooth) { "SMOOTH" } elseif ($HighSensitivity) { "HIGH" } else { "BALANCED" }
Write-Host "Sensitivity: $ModeLabel. Motion and joint limits remain enabled." -ForegroundColor Yellow

& $Python -m jarvis.perception.cli `
    --source "http://127.0.0.1:8000/api/frame" `
    --headless `
    --web-preview 8768 `
    --web-host $WebHost `
    --demo-title $DemoTitle `
    --face-confidence $FaceConfidence `
    --hand-confidence $HandConfidence `
    --servo-execute `
    --servo-endpoint $ServoEndpoint `
    --servo-rate-hz $ServoRate `
    --servo-pan-gain $ServoPanGain `
    --servo-deadband $ServoDeadband `
    --servo-vertical-rate-hz $ServoVerticalRate `
    --servo-step-deg $ServoStepDeg `
    --servo-max-step $ServoMaxStep `
    --servo-vertical-max-step $ServoVerticalMaxStep `
    --servo-vertical-min-step $ServoVerticalMinStep `
    --servo-vertical-deadband $ServoVerticalDeadband `
    --servo-vertical-joint wrist_flex `
    --enable-vertical-servo `
    --allow-experimental-vertical `
    --pointing-execute `
    --pointing-pan-gain $PointingPanGain `
    --pointing-vertical-gain $PointingVerticalGain `
    --pointing-max-step $PointingMaxStep `
    --pointing-deadband $PointingDeadband `
    --pointing-min-confidence $PointingConfidence `
    --pointing-rate-hz $PointingRate `
    --pointing-vertical-rate-hz $PointingVerticalRate `
    --pointing-step-deg $PointingStepDeg `
    --pointing-vertical-max-step $ServoVerticalMaxStep `
    --pointing-vertical-min-step $ServoVerticalMinStep `
    --pointing-max-steps 0 `
    --pointing-settle-s $PointingSettle `
    --pointing-vertical-joint wrist_flex `
    --enable-pointing-vertical `
    --allow-experimental-pointing-vertical `
    @args

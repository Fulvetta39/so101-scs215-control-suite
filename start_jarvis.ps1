$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$MainPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PerceptionPython = Join-Path $ProjectRoot ".venv_perception\Scripts\python.exe"

function Test-JarvisEnvironment {
    if (-not (Test-Path -LiteralPath $MainPython) -or -not (Test-Path -LiteralPath $PerceptionPython)) {
        return $false
    }
    & $MainPython -c "import fastapi, uvicorn, lerobot" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    & $PerceptionPython -c "import cv2, mediapipe" 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-JarvisEnvironment)) {
    Write-Host "First run detected. Installing Python environments and dependencies..." -ForegroundColor Cyan
    & (Join-Path $ProjectRoot "bootstrap_handoff.ps1")
    if (-not (Test-JarvisEnvironment)) {
        throw "Environment installation did not complete. Check the messages above."
    }
}

$Answer = Read-Host "Clear the robot workspace and keep the emergency stop reachable. Type READY to start"
if ($Answer.Trim().ToUpperInvariant() -ne "READY") {
    Write-Host "Start cancelled." -ForegroundColor Yellow
    exit 1
}

$MainCommand = "Set-Location -LiteralPath '$ProjectRoot'; & '.\run.ps1' -Config config.robot.toml"
Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $MainCommand) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Normal

$Ready = $false
for ($Attempt = 0; $Attempt -lt 60; $Attempt += 1) {
    Start-Sleep -Milliseconds 500
    try {
        $Status = Invoke-RestMethod -TimeoutSec 2 http://127.0.0.1:8000/api/status
        if ($Status.robot.connected -and -not $Status.robot.stopped) {
            $Ready = $true
            break
        }
    } catch {
        continue
    }
}

if (-not $Ready) {
    Write-Host "Robot service did not become ready within 30 seconds. Check the main service window." -ForegroundColor Red
    exit 1
}

$DemoCommand = "Set-Location -LiteralPath '$ProjectRoot'; & '.\run_midterm_demo.ps1' -Smooth"
Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $DemoCommand) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Normal

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:8000/demo"
Write-Host "JARVIS started. Unified console: http://127.0.0.1:8000/demo" -ForegroundColor Green

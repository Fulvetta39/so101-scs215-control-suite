param(
    [switch]$SkipRobotDependencies,
    [switch]$VerifyOnly,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Test-Python312 {
    param([string]$Candidate)
    if (-not $Candidate) {
        return $false
    }
    try {
        $version = & $Candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return $LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.12"
    } catch {
        return $false
    }
}

function Invoke-Checked {
    param(
        [string]$Description,
        [string]$FilePath,
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Find-Python312 {
    if ($PythonExe -and (Test-Python312 $PythonExe)) {
        return $PythonExe
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidate = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate -and (Test-Python312 $candidate.Trim())) {
            return $candidate.Trim()
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        if (Test-Python312 (Get-Command python).Source) {
            return (Get-Command python).Source
        }
    }
    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:ProgramFiles "Python312\python.exe"),
        "C:\Python312\python.exe"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -and (Test-Python312 $Candidate)) {
            return $Candidate
        }
    }
    throw "Python 3.12 was not found. Install Python 3.12, or run .\bootstrap_handoff.ps1 -PythonExe 'C:\full\path\python.exe'."
}

$MainPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$PerceptionPython = Join-Path $PSScriptRoot ".venv_perception\Scripts\python.exe"
$Python = $null
if (-not (Test-Path -LiteralPath $MainPython) -or -not (Test-Path -LiteralPath $PerceptionPython)) {
    $Python = Find-Python312
    Write-Host "Using base Python: $Python" -ForegroundColor Cyan
}

if ($VerifyOnly) {
    if (-not (Test-Path -LiteralPath $MainPython) -or -not (Test-Path -LiteralPath $PerceptionPython)) {
        throw "VerifyOnly requires existing .venv and .venv_perception. Run without -VerifyOnly to create them."
    }
    if (-not (Test-Python312 $MainPython) -or -not (Test-Python312 $PerceptionPython)) {
        throw "An existing virtual environment is not Python 3.12."
    }
    Write-Host "Existing Python environments are valid:" -ForegroundColor Green
    Write-Host "  $MainPython"
    Write-Host "  $PerceptionPython"
    & $MainPython handoff_doctor.py
    exit 0
}

if (-not (Test-Path -LiteralPath $MainPython)) {
    Invoke-Checked "Create .venv" $Python @("-m", "venv", ".venv")
}
if (-not (Test-Python312 $MainPython)) {
    throw "The existing .venv is not Python 3.12. Rename or remove it, then run this script again."
}
Invoke-Checked "Upgrade pip in .venv" $MainPython @("-m", "pip", "install", "-U", "pip")
if (-not $SkipRobotDependencies) {
    Invoke-Checked "Install main robot dependencies" $MainPython @(
        "-m", "pip", "install", "-e", ".[test,robot]"
    )
} else {
    Invoke-Checked "Install main test dependencies" $MainPython @(
        "-m", "pip", "install", "-e", ".[test]"
    )
}

if (-not (Test-Path -LiteralPath $PerceptionPython)) {
    Invoke-Checked "Create .venv_perception" $Python @("-m", "venv", ".venv_perception")
}
if (-not (Test-Python312 $PerceptionPython)) {
    throw "The existing .venv_perception is not Python 3.12. Rename or remove it, then run this script again."
}

Invoke-Checked "Upgrade pip in .venv_perception" $PerceptionPython @(
    "-m", "pip", "install", "-U", "pip"
)
Invoke-Checked "Install perception dependencies" $PerceptionPython @(
    "-m", "pip", "install", "-r", "requirements-perception.txt"
)
Invoke-Checked "Verify main Python packages" $MainPython @(
    "-c", "import fastapi, uvicorn"
)
Invoke-Checked "Verify perception Python packages" $PerceptionPython @(
    "-c", "import cv2, mediapipe"
)
if (-not $SkipRobotDependencies) {
    Invoke-Checked "Verify LeRobot package" $MainPython @(
        "-c", "import lerobot"
    )
}

if (Test-Path -LiteralPath "calibration\so101-follower.json") {
    & (Join-Path $PSScriptRoot "install_handoff_calibration.ps1")
}

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env. Edit it and add your own DASHSCOPE_API_KEY." -ForegroundColor Yellow
}

Write-Host "`nRunning tests..." -ForegroundColor Cyan
Invoke-Checked "Run automated tests" $MainPython @("-m", "pytest", "-q")

Write-Host "`nEnvironment setup completed." -ForegroundColor Green
Write-Host "Next: add DASHSCOPE_API_KEY to .env, confirm COM/camera in config.robot.toml, then start the service." -ForegroundColor Yellow

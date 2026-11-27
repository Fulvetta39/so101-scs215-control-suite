param(
    [string]$Config = "config.toml"
)

$ErrorActionPreference = "Stop"
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Missing .venv. Create the virtual environment and install dependencies first."
}

& $VenvPython -m so101_agent --config $Config


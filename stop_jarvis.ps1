$ErrorActionPreference = "Continue"

try {
    # Shutdown keeps the previous behaviour: remove torque before closing.
    Invoke-RestMethod -TimeoutSec 3 -Method Post http://127.0.0.1:8000/api/estop | Out-Null
} catch {
}

$Lines = netstat -ano -p tcp
foreach ($Port in @(8768, 8000)) {
    $Match = $Lines | Select-String ":$Port\s+.*LISTENING\s+(\d+)" | Select-Object -First 1
    if ($Match -and $Match.Matches.Count -gt 0) {
        $OwnerId = [int]$Match.Matches[0].Groups[1].Value
        if ($OwnerId -gt 0) {
            Stop-Process -Id $OwnerId -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "JARVIS stopped. Emergency hold was triggered and program ports are closed." -ForegroundColor Green

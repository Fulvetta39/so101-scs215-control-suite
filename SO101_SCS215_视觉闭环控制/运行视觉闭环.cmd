@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"

set "PROJECT_PYTHON=..\.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" goto python_missing
if /I "%~1"=="--check" goto check_ok

echo ========================================
echo SO-101 Black Ball Visual Closed Loop
echo Camera: GENERAL WEBCAM 1, rotate 180 deg
echo Arm: COM5, smooth adaptive moves, max 3.0 cm total
echo Type uppercase MOVE after all safety checks pass.
echo Press Ctrl+C to stop and release torque.
echo ========================================
echo.

"%PROJECT_PYTHON%" -B visual_closed_loop.py run --camera 1 --rotate 180 --headless --port COM5 --real --min-step-mm 2 --max-step-mm 10 --gain-mm 24 --settle-ms 120 --flush-frames 2 --max-moves 8 --max-total-cm 3.0
set "EXIT_CODE=%ERRORLEVEL%"
goto finished

:check_ok
echo Launcher check passed.
endlocal & exit /b 0

:python_missing
echo ERROR: Project Python was not found at:
echo %CD%\%PROJECT_PYTHON%
set "EXIT_CODE=1"

:finished
echo.
echo Program finished with exit code %EXIT_CODE%.
pause
endlocal & exit /b %EXIT_CODE%

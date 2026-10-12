@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"

set "PROJECT_PYTHON=.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" goto python_missing

"%PROJECT_PYTHON%" -B launcher.py %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finished

:python_missing
echo ERROR: Project Python was not found at:
echo %CD%\%PROJECT_PYTHON%
echo Create .venv and install requirements.txt first.
set "EXIT_CODE=1"

:finished
echo.
echo Launcher finished with exit code %EXIT_CODE%.
pause
endlocal & exit /b %EXIT_CODE%

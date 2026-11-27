@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_jarvis.ps1"
if errorlevel 1 pause

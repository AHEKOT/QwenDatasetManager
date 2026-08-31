@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Qwen Dataset Manager...

REM Keep the complete Python process tree tied to this console. The PowerShell
REM launcher places it in a Windows Job Object with KILL_ON_JOB_CLOSE, so
REM closing this console cannot leave the Flask server running in background.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_server.ps1"
exit /b %ERRORLEVEL%

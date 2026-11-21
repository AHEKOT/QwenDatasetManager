@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "QDM_INSTALL_EXIT=%ERRORLEVEL%"

if not "%QDM_INSTALL_EXIT%"=="0" (
    echo.
    echo Installation failed. See the error above.
) else (
    echo.
    echo Installation finished successfully.
)

if not defined QDM_NO_PAUSE pause
exit /b %QDM_INSTALL_EXIT%

@echo off
setlocal
cd /d "%~dp0"
echo 🚀 Starting Qwen Dataset Manager...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the application
python app.py

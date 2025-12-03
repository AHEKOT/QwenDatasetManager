@echo off
echo 🚀 Installing Qwen Dataset Manager...

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo ✅ Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Installation complete!
echo.
echo To run the application:
echo   run.cmd
echo.
pause

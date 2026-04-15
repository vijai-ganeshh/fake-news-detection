@echo off
REM Fake News Detection - Run Script (Windows CMD)
REM ================================================

echo.
echo ==========================================
echo   Fake News Detection - Server Startup
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Virtual environment
set VENV_DIR=.venv

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

REM Install dependencies
if not exist "%VENV_DIR%\.deps_installed" (
    echo Installing dependencies (this may take a few minutes)...
    pip install --upgrade pip -q
    pip install -r requirements.txt
    echo. > "%VENV_DIR%\.deps_installed"
    echo Dependencies installed successfully!
) else (
    echo Dependencies already installed
)

echo.
echo ==========================================
echo Starting server...
echo ==========================================
echo.
echo Web Interface: http://localhost:8000
echo API Docs:      http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

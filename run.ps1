# Fake News Detection - Run Script (Windows PowerShell)
# ======================================================

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Fake News Detection - Server Startup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: " -NoNewline
    Write-Host $pythonVersion -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Virtual environment
$venvDir = ".venv"
$venvActivate = "$venvDir\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvDir
}

Write-Host "Activating virtual environment..."
& $venvActivate

# Install dependencies
$depsMarker = "$venvDir\.deps_installed"

if (-not (Test-Path $depsMarker)) {
    Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
    pip install --upgrade pip -q
    pip install -r requirements.txt
    New-Item -Path $depsMarker -ItemType File -Force | Out-Null
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "Dependencies already installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Starting server..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Web Interface: " -NoNewline
Write-Host "http://localhost:8000" -ForegroundColor Blue
Write-Host "API Docs:      " -NoNewline
Write-Host "http://localhost:8000/docs" -ForegroundColor Blue
Write-Host ""
Write-Host "Press Ctrl+C to stop the server"
Write-Host ""

# Run server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

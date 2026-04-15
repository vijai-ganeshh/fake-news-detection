#!/bin/bash

# Fake News Detection - Run Script (Linux/macOS)
# ================================================

set -e

echo "=========================================="
echo "  Fake News Detection - Server Startup"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "Python version: ${GREEN}$PYTHON_VERSION${NC}"

# Virtual environment
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install dependencies
if [ ! -f "$VENV_DIR/.deps_installed" ]; then
    echo -e "${YELLOW}Installing dependencies (this may take a few minutes)...${NC}"
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    touch "$VENV_DIR/.deps_installed"
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
else
    echo -e "${GREEN}Dependencies already installed${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Starting server...${NC}"
echo "=========================================="
echo ""
echo "Web Interface: http://localhost:8000"
echo "API Docs:      http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

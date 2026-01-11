#!/bin/bash

# AI Content Detector - Run Script

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Checking dependencies..."
pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Load env vars if .env exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "Starting AI Content Detector..."
echo "Open http://localhost:8000 in your browser"
echo ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

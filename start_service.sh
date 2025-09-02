#!/bin/bash

echo "Starting Stock Chart Analysis API..."
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements if needed
echo "Installing requirements..."
pip install -r requirements.txt

# Start the service
echo
echo "Starting the service on http://localhost:8000"
echo "Press Ctrl+C to stop the service"
echo
python main.py

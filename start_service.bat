@echo off
echo Starting Stock Chart Analysis API...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Creating one...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate

REM Install requirements if needed
echo Installing requirements...
pip install -r requirements.txt

REM Start the service
echo.
echo Starting the service on http://localhost:8000
echo Press Ctrl+C to stop the service
echo.
python main.py

pause

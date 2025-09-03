"""
Main entry point for the Stock Chart Analysis API
This file imports and runs the enhanced version
"""

# Import the enhanced app
from main_enhanced import app

# This allows Render to find the app when running 'uvicorn main:app'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

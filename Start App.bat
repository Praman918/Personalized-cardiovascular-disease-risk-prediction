@echo off
title Personalized Disease Risk Manager
color 0A
echo.
echo  ================================================
echo   Heart  Personalized Disease Risk Manager
echo   Powered by Federated Learning
echo  ================================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python is not installed or not in PATH.
    echo  Download Python 3.10+ from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [1/4] Python found.

:: Create venv if it doesn't exist
if not exist ".venv" (
    echo  [2/4] Creating virtual environment...
    python -m venv .venv
) else (
    echo  [2/4] Virtual environment already exists.
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install dependencies
echo  [3/4] Installing dependencies (first time may take a few minutes)...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo  ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

:: Train model if needed
if not exist "global_model.pth" (
    echo.
    echo  [4/4] No trained model found. Running Federated Learning simulation...
    echo  This takes about 2-3 minutes. Please wait.
    echo.
    python run_simulation.py
)

:: Launch the app
echo.
echo  ================================================
echo   Starting the app at http://localhost:8501
echo   Your browser should open automatically.
echo   Keep this window open while using the app.
echo  ================================================
echo.

:: Open browser after 5 seconds in background
start /b cmd /c "timeout /t 5 >nul && start http://localhost:8501"

:: Run streamlit
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause

@echo off
REM ============================================
REM Video RAG - Setup and Run Script
REM ============================================

echo.
echo ============================================
echo VIDEO RAG - ENVIRONMENT SETUP
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo.
    echo [2/4] Virtual environment already exists, skipping creation...
)

REM Activate virtual environment
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo [4/4] Installing requirements...
if exist "requirements.txt" (
    echo Installing packages from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
) else (
    echo WARNING: requirements.txt not found!
)

REM Install additional packages if needed
if exist "rag_library\requirements.txt" (
    echo.
    echo Installing additional packages from rag_library\requirements.txt...
    pip install -r rag_library\requirements.txt
)

echo.
echo ============================================
echo SETUP COMPLETE!
echo ============================================
echo.
echo Environment is ready. You can now run the application.
echo.

REM Ask if user wants to run the application now
set /p run_now="Do you want to run the application now? (y/n): "
if /i "%run_now%"=="y" (
    echo.
    echo ============================================
    echo RUNNING APPLICATION
    echo ============================================
    echo.
    python main.py
)

pause

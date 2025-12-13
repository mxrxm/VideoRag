@echo off
REM ============================================
REM Video RAG - Quick Run Script
REM ============================================

echo.
echo ============================================
echo VIDEO RAG - STARTING APPLICATION
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create the environment and install dependencies.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Run the application
echo.
echo Starting application...
echo.
python main.py

pause

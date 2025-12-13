@echo off
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

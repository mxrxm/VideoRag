@echo off

REM Check if venv exists, create if not
IF NOT EXIST venv (
    python -m venv venv
) ELSE (
    echo venv folder already exists, skipping creation...
)

REM Activate the virtual environment
call .\venv\Scripts\activate.bat

REM Set path to Python inside venv
set PYTHON="venv\Scripts\Python.exe"
echo Using Python at %PYTHON%

REM Run the Python script
%PYTHON% Launcher.py
IF ERRORLEVEL 1 (
    echo.
    echo Launch unsuccessful. Exiting.
) ELSE (
    echo.
    echo Script finished successfully!
)

pause

@echo off
REM ============================================
REM Video RAG - Streamlit Web Interface
REM ============================================

echo.
echo ============================================
echo VIDEO RAG - STREAMLIT WEB APP
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

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo Streamlit not found! Installing...
    pip install streamlit
)

REM Run Streamlit app
echo.
echo Starting Streamlit app...
echo.
echo The app will open in your default browser automatically.
echo Press Ctrl+C to stop the server.
echo.

streamlit run streamlit_app.py

pause

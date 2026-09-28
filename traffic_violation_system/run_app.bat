@echo off
REM run_app.bat — Run the Streamlit UI with proper Python path

cd /d "%~dp0"

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo  Traffic Violation System — Streamlit UI
echo ============================================================
echo.

set PYTHON="C:\Users\mujah\AppData\Local\Programs\Python\Python310\python.exe"

%PYTHON% -m streamlit run app/main.py

pause

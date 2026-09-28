@echo off
REM run_ui.bat — Launch the Streamlit UI
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo Starting Traffic Violation UI...
streamlit run app/main.py --server.port 8501 --server.headless false
pause

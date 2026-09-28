@echo off
REM run_api.bat — Launch the FastAPI backend server
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo Starting Traffic Violation API server...
echo API docs: http://127.0.0.1:8000/docs
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
pause

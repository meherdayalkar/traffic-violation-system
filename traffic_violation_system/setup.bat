@echo off
REM setup.bat — Minimal install for systems that already have YOLO + PyTorch
REM Run ONCE from the project folder

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo  Traffic Violation System — Quick Setup
echo  (Skipping PyTorch/YOLO — already installed)
echo ============================================================
echo.

python --version
echo.

echo [1/4] Installing missing dependencies only...
pip install easyocr Pillow fastapi uvicorn python-multipart
pip install streamlit plotly pandas
pip install pydantic loguru python-dotenv

echo.
echo [2/4] Creating output directories...
if not exist "output\clips"    mkdir "output\clips"
if not exist "output\challans" mkdir "output\challans"
if not exist "output\frames"   mkdir "output\frames"
if not exist "data\sample"     mkdir "data\sample"

echo.
echo [3/4] Verifying YOLO import...
python -c "from ultralytics import YOLO; print('[OK] ultralytics imported')"

echo.
echo [4/4] Verifying tracker (pure numpy — no filterpy needed)...
python -c "from tracking.kalman_numpy import KalmanFilter; print('[OK] KalmanFilter imported')"

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo  Run these commands in order:
echo.
echo  1. Test pipeline (no video):
echo     python test_pipeline.py --no-display
echo.
echo  2. Test with your video:
echo     python test_pipeline.py --video YOUR_VIDEO.mp4
echo.
echo  3. Launch full UI:
echo     streamlit run app/main.py
echo.
pause

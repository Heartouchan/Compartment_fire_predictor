@echo off
title Fire Curve Predictor Launcher
echo Checking Python environment...


REM 
cd /d %~dp0

echo Installing required packages (if not already installed)...
pip install -r requirements.txt

echo.
echo Running Program.py...
python Program.py

echo.
echo ------------------------------
echo   Program finished running.
echo   Press any key to exit...
pause > nul

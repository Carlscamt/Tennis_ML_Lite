@echo off
REM Tennis Prediction App - Daily Run Script
REM This script runs the batch job to forecast upcoming matches

echo ========================================================
echo   TENNIS PREDICTION ENGINE - DAILY BATCH RUN
echo   Date: %DATE% %TIME%
echo ========================================================

cd /d "%~dp0"

echo [1/3] activating environment (if venv exists)...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo [2/3] Running Batch Prediction (7 days ahead)...
python tennis.py batch-run --days 7 --force

echo [3/3] Displaying Top Value Bets...
python tennis.py show-predictions --limit 20 --min-odds 1.6

echo.
echo ========================================================
echo   DONE. Log closed.
echo ========================================================
pause

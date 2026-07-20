@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo ONNX GPU validation setup and run
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    echo Install 64-bit Python 3.12 and try again.
    pause
    exit /b 1
)

if not exist ".venv-gpu\Scripts\python.exe" (
    echo [1/4] Creating GPU virtual environment...
    python -m venv .venv-gpu
    if errorlevel 1 goto :failed
) else (
    echo [1/4] Using existing GPU virtual environment.
)

echo [2/4] Updating pip...
".venv-gpu\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo [3/4] Installing pinned GPU packages...
".venv-gpu\Scripts\python.exe" -m pip install -r requirements-gpu.txt
if errorlevel 1 goto :failed

echo [4/4] Running ONNX GPU profiling for all models...
".venv-gpu\Scripts\python.exe" profile_onnx_gpu.py
if errorlevel 1 goto :failed

echo.
echo [SUCCESS] Validation completed.
echo Summary: results_gpu_profile.csv
echo CPU nodes: cpu_nodes_gpu_profile.csv
echo.
pause
exit /b 0

:failed
echo.
echo [ERROR] The process failed. Review the message above.
echo If CSV files were created, they contain completed model results.
echo.
pause
exit /b 1

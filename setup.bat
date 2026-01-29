@echo off
setlocal enabledelayedexpansion

:: ========================================
::  Filharmonia AI - Ultimate Setup Script
::  Windows - One-Click Setup
:: ========================================

echo.
echo ========================================
echo   Filharmonia AI - Setup
echo ========================================
echo.
echo This will install everything needed.
echo Estimated time: 3-5 minutes
echo.

:: Check prerequisites
echo [1/7] Checking prerequisites...
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VER=%%i
echo [OK] Python %PYTHON_VER% found

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found!
    echo.
    echo Please install Node.js 18+ from:
    echo https://nodejs.org/
    echo.
    pause
    exit /b 1
)

for /f %%i in ('node --version') do set NODE_VER=%%i
echo [OK] Node.js %NODE_VER% found

:: Check/Install pnpm
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing pnpm...
    npm install -g pnpm --silent
)
for /f %%i in ('pnpm --version') do set PNPM_VER=%%i
echo [OK] pnpm %PNPM_VER% found

echo.

:: ========================================
:: Backend Setup
:: ========================================

echo [2/7] Setting up Python backend...
cd backend

:: Create venv
if not exist venv (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

:: Detect GPU
echo.
echo [3/7] Detecting hardware...
python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" >nul 2>&1
if errorlevel 1 (
    set DEVICE=CPU
    echo [INFO] No GPU detected - will use CPU-only PyTorch
) else (
    for /f %%i in ('python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"') do set DEVICE=%%i
    if "!DEVICE!"=="CUDA" (
        echo [INFO] NVIDIA GPU detected - will use CUDA acceleration
    ) else (
        echo [INFO] No GPU detected - will use CPU-only PyTorch
    )
)

:: Install PyTorch
echo.
echo [4/7] Installing PyTorch...
if "!DEVICE!"=="CUDA" (
    echo [INFO] Installing PyTorch with CUDA 12.1 support...
    pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo [INFO] Installing PyTorch CPU-only...
    pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --quiet
)

if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

:: Install dependencies
echo.
echo [5/7] Installing backend dependencies...
pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

:: Export ONNX model for CPU speedup
echo.
echo [6/7] Optimizing for CPU inference...
echo [INFO] Exporting ONNX INT8 model (takes ~30 seconds)...
python -m scripts.export_onnx >nul 2>&1
if errorlevel 1 (
    echo [WARN] ONNX export failed - will use PyTorch fallback
    echo [INFO] This is OK - app will work but CPU inference will be slower
) else (
    echo [OK] ONNX model exported - CPU inference 3x faster!
)

cd ..

:: ========================================
:: Frontend Setup
:: ========================================

echo.
echo [7/7] Setting up React frontend...
cd frontend

if not exist node_modules (
    echo [INFO] Installing frontend dependencies...
    pnpm install --silent
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies
        pause
        exit /b 1
    )
)

cd ..

:: ========================================
:: Verify Installation
:: ========================================

echo.
echo ========================================
echo   Verifying Installation
echo ========================================
echo.

cd backend
call venv\Scripts\activate.bat
python ..\verify_installation.py
if errorlevel 1 (
    echo.
    echo [ERROR] Installation verification failed!
    echo Please check the errors above and re-run setup.bat
    pause
    exit /b 1
)
cd ..

:: Create setup marker
echo %date% %time% > .setup_complete

:: ========================================
:: Environment Setup
:: ========================================

if not exist .env (
    echo.
    echo [INFO] Creating .env configuration file...
    copy .env.example .env >nul
    echo [OK] Created .env - you can edit it to configure data directories
)

:: ========================================
:: Success
:: ========================================

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Hardware: !DEVICE!
if "!DEVICE!"=="CUDA" (
    echo Acceleration: torch.compile ^(GPU^)
) else (
    if exist "backend\recognition_models\ast_active_int8.onnx" (
        echo Acceleration: ONNX INT8 ^(3x CPU speedup^)
    ) else (
        echo Acceleration: PyTorch CPU ^(fallback^)
    )
)
echo.
echo Next step:
echo   - Double-click start.bat to launch the app
echo.
echo The app will open at: http://localhost:5173
echo.
pause

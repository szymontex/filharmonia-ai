@echo off
echo ===================================
echo Filharmonia AI - Setup Script
echo ===================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.

REM Check pnpm
echo [2/5] Checking pnpm...
pnpm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: pnpm is not installed
    echo Installing pnpm via npm...
    npm install -g pnpm
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install pnpm. Please install Node.js first.
        pause
        exit /b 1
    )
)
echo.

REM Setup backend
echo [3/5] Setting up Python backend...
cd backend

if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing PyTorch with CUDA support...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo Installing remaining dependencies...
pip install -r requirements.txt --no-deps
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

cd ..
echo.

REM Setup frontend
echo [4/5] Setting up React frontend...
cd frontend

if not exist node_modules (
    echo Installing frontend dependencies...
    pnpm install
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install frontend dependencies
        pause
        exit /b 1
    )
)

cd ..
echo.

REM Setup environment
echo [5/5] Setting up environment...
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env and configure FILHARMONIA_BASE_DIR
    echo           if you want to use a custom data directory.
)
echo.

echo ===================================
echo Setup completed successfully!
echo ===================================
echo.
echo Verifying installation...
cd backend
call venv\Scripts\python ..\verify_installation.py
cd ..
echo.
echo Next steps:
echo 1. Edit .env file to configure your data directory
echo 2. Run start.bat to start both servers
echo.
pause

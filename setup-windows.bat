@echo off
:: Filharmonia AI - Windows Setup Script
:: Double-click to install and run

echo ================================================
echo  Filharmonia AI - Windows Setup
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found!
    echo.
    echo Please install Node.js 18+ from:
    echo https://nodejs.org/
    pause
    exit /b 1
)

:: Check pnpm
pnpm --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing pnpm...
    npm install -g pnpm
)

echo [OK] Prerequisites found
echo.

:: ================================================
:: Backend Setup
:: ================================================
echo ================================================
echo  Setting up Backend (Python)
echo ================================================
echo.

cd backend

:: Create virtual environment if doesn't exist
if not exist "venv\" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install PyTorch CPU version
echo [INFO] Installing PyTorch (CPU version)...
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --quiet

:: Install dependencies
echo [INFO] Installing backend dependencies...
pip install -r requirements.txt --quiet

:: Export ONNX model for CPU speedup
echo [INFO] Exporting ONNX model (this takes ~30 seconds)...
python -m scripts.export_onnx 2>nul
if errorlevel 1 (
    echo [WARN] ONNX export failed - will use PyTorch CPU fallback
) else (
    echo [OK] ONNX model exported - 3x+ CPU speedup enabled
)

cd ..

:: ================================================
:: Frontend Setup
:: ================================================
echo.
echo ================================================
echo  Setting up Frontend (Node.js)
echo ================================================
echo.

cd frontend

:: Install dependencies
echo [INFO] Installing frontend dependencies...
pnpm install --silent

cd ..

:: ================================================
:: Create run script
:: ================================================
echo.
echo [INFO] Creating run script...

(
echo @echo off
echo :: Filharmonia AI - Start Application
echo.
echo echo ================================================
echo echo  Starting Filharmonia AI
echo echo ================================================
echo echo.
echo echo Backend will start on: http://localhost:8000
echo echo Frontend will start on: http://localhost:5173
echo echo.
echo echo Press Ctrl+C to stop both services
echo echo ================================================
echo.
echo :: Start backend in background
echo cd backend
echo start /B cmd /c "venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo cd ..
echo.
echo :: Wait for backend to start
echo timeout /t 3 /nobreak ^>nul
echo.
echo :: Start frontend
echo cd frontend
echo pnpm dev
) > run-windows.bat

echo [OK] Setup complete!
echo.
echo ================================================
echo  Setup Complete!
echo ================================================
echo.
echo To start the application:
echo   1. Double-click "run-windows.bat"
echo   2. Open browser to http://localhost:5173
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Device detection at startup will show:
echo   - "Device: cpu (CPU)" for CPU-only
echo   - "Using ONNX INT8 backend (3.2x speedup)" if export succeeded
echo.
pause

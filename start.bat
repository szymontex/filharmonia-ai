@echo off
REM Filharmonia AI - Startup Script (Windows)
REM Starts both backend and frontend servers and opens browser

echo ================================
echo   Filharmonia AI - Starting...
echo ================================
echo.

REM Kill existing servers if running
echo [0/4] Stopping existing servers...

REM Kill backend (port 8000)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM Kill frontend (port 5173)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM Also kill by window title as backup
taskkill /F /FI "WINDOWTITLE eq Filharmonia Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Filharmonia Frontend*" >nul 2>&1

timeout /t 2 /nobreak >nul

REM Check if backend venv exists
if not exist "backend\venv\Scripts\activate.bat" (
    echo [ERROR] Backend virtual environment not found!
    echo Please run: cd backend ^&^& python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend dependencies not installed!
    echo Please run: cd frontend ^&^& pnpm install
    pause
    exit /b 1
)

echo [1/4] Starting backend server...
start "Filharmonia Backend" cmd /k "cd backend && venv\Scripts\activate && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul

echo [2/4] Starting frontend server...
start "Filharmonia Frontend" cmd /k "cd frontend && pnpm dev"
timeout /t 5 /nobreak >nul

echo [3/4] Waiting for servers to start...
timeout /t 3 /nobreak >nul

echo [4/4] Opening browser...
start http://localhost:5173

echo.
echo ================================
echo   Servers are running!
echo ================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C in server windows to stop
echo.
pause

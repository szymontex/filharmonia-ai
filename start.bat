@echo off
:: ========================================
::  Filharmonia AI - Start Application
::  Windows
:: ========================================

echo.
echo ========================================
echo   Filharmonia AI - Starting
echo ========================================
echo.

:: Check if setup was completed
if not exist .setup_complete (
    echo [ERROR] Setup not completed!
    echo.
    echo Please run setup.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

:: Verify backend setup
if not exist "backend\venv\Scripts\activate.bat" (
    echo [ERROR] Backend virtual environment not found!
    echo.
    echo Please run setup.bat to install dependencies.
    echo.
    pause
    exit /b 1
)

:: Verify frontend setup
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend dependencies not installed!
    echo.
    echo Please run setup.bat to install dependencies.
    echo.
    pause
    exit /b 1
)

:: Kill existing servers
echo [1/4] Stopping existing servers...

:: Kill by port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1

:: Kill by window title
taskkill /F /FI "WINDOWTITLE eq Filharmonia Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Filharmonia Frontend*" >nul 2>&1

timeout /t 2 /nobreak >nul

:: Start backend
echo [2/4] Starting backend server...
start "Filharmonia Backend" cmd /k "cd backend && venv\Scripts\activate && echo Backend starting on port 8000... && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul

:: Start frontend
echo [3/4] Starting frontend server...
start "Filharmonia Frontend" cmd /k "cd frontend && echo Frontend starting on port 5173... && pnpm dev"
timeout /t 5 /nobreak >nul

:: Open browser
echo [4/4] Opening browser...
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo ========================================
echo   Application Running
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Two command windows opened:
echo   - Filharmonia Backend  (port 8000)
echo   - Filharmonia Frontend (port 5173)
echo.
echo Press Ctrl+C in those windows to stop
echo Or run stop.bat to stop all servers
echo.
echo This window can be closed safely.
echo.
pause

@echo off
echo Killing all servers...

REM Kill all python processes
taskkill /F /IM python.exe >nul 2>&1

REM Kill all node processes
taskkill /F /IM node.exe >nul 2>&1

REM Kill by ports
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1

timeout /t 2 /nobreak >nul

echo Starting fresh...
call start.bat

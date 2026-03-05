@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title Filharmonia AI - Setup

REM Log all output to file for debugging
set "LOGFILE=%~dp0setup.log"
if "%~1"=="__logged__" goto :main
echo Setup started: %date% %time% > "!LOGFILE!"
powershell -NoProfile -Command "& cmd /c '\"%~f0\" __logged__' 2>&1 | Tee-Object -FilePath '!LOGFILE!' -Append"
echo.
echo  Log saved to: !LOGFILE!
pause
exit /b

:main
echo.
echo  =========================================
echo   Filharmonia AI - Setup Wizard
echo  =========================================
echo.

set "ERRORS=0"
set "SCRIPT_DIR=%~dp0"

REM ============================================================
REM  STEP 1: Check/Install Python
REM ============================================================
echo [1/6] Checking Python...

where python >nul 2>&1
if !errorlevel! equ 0 goto :python_in_path

echo   [!] Python not found in PATH.
echo   [!] Searching common locations...

set "PYTHON_FOUND="
for %%D in (
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "C:\Python313"
    "C:\Python312"
    "C:\Python311"
    "C:\Python310"
    "%ProgramFiles%\Python313"
    "%ProgramFiles%\Python312"
    "%ProgramFiles%\Python311"
) do (
    if not defined PYTHON_FOUND if exist "%%~D\python.exe" set "PYTHON_FOUND=%%~D"
)

if not defined PYTHON_FOUND (
    echo.
    echo   PYTHON NOT INSTALLED
    echo   1. Open: https://www.python.org/downloads/
    echo   2. Download Python 3.11 or newer
    echo   3. IMPORTANT: Check "Add Python to PATH" during install
    echo   4. Run setup.bat again
    echo.
    echo   Opening download page...
    start "" "https://www.python.org/downloads/"
    goto :fail
)

echo   [+] Found: !PYTHON_FOUND!\python.exe
set "PATH=!PYTHON_FOUND!;!PYTHON_FOUND!\Scripts;!PATH!"
echo   [+] Added to PATH for this session

:python_in_path
REM Test that python actually runs (not Microsoft Store alias)
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo   [ERROR] 'python' command does not work.
    echo   [ERROR] This may be the Microsoft Store alias - install Python from python.org
    goto :fail
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
if not defined PYVER (
    echo   [ERROR] Could not read Python version.
    goto :fail
)
echo   [OK] Python !PYVER!

set "PY_MAJOR="
set "PY_MINOR="
for /f "tokens=1,2 delims=." %%A in ("!PYVER!") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)
if not defined PY_MAJOR goto :python_version_bad
if not defined PY_MINOR goto :python_version_bad
if !PY_MAJOR! lss 3 goto :python_version_bad
if !PY_MAJOR! equ 3 if !PY_MINOR! lss 10 goto :python_version_bad
goto :python_version_ok

:python_version_bad
echo   [ERROR] Python 3.10+ required, found !PYVER!
goto :fail

:python_version_ok

REM ============================================================
REM  STEP 2: Check/Install Node.js + pnpm
REM ============================================================
echo.
echo [2/6] Checking Node.js and pnpm...

where node >nul 2>&1
if !errorlevel! equ 0 goto :node_ok

set "NODE_FOUND="
for %%D in (
    "%ProgramFiles%\nodejs"
    "%LOCALAPPDATA%\Programs\nodejs"
    "%ProgramFiles(x86)%\nodejs"
) do (
    if not defined NODE_FOUND if exist "%%~D\node.exe" set "NODE_FOUND=%%~D"
)

if defined NODE_FOUND (
    set "PATH=!NODE_FOUND!;!PATH!"
    echo   [+] Node.js found: !NODE_FOUND!
    goto :node_ok
)

echo   [!] Node.js not installed. Trying winget...
where winget >nul 2>&1
if !errorlevel! neq 0 goto :node_not_installed

echo   [*] Installing Node.js LTS via winget...
winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
if !errorlevel! equ 0 (
    set "PATH=%ProgramFiles%\nodejs;%APPDATA%\npm;!PATH!"
    echo   [OK] Node.js installed.
    goto :node_ok
)
echo   [!] winget could not install Node.js

:node_not_installed
echo.
echo   NODE.JS NOT INSTALLED
echo   1. Open: https://nodejs.org/
echo   2. Download LTS version
echo   3. Install with default options
echo   4. Run setup.bat again
echo.
start "" "https://nodejs.org/"
goto :fail

:node_ok
for /f "tokens=*" %%V in ('node --version 2^>^&1') do echo   [OK] Node.js %%V

set "PATH=%APPDATA%\npm;!PATH!"

REM Check pnpm
where pnpm >nul 2>&1
if !errorlevel! equ 0 goto :pnpm_ok

echo   [!] pnpm not installed. Installing...

where corepack >nul 2>&1
if !errorlevel! neq 0 goto :pnpm_via_npm
echo   [*] Enabling pnpm via corepack...
corepack enable >nul 2>&1
corepack prepare pnpm@latest --activate >nul 2>&1
where pnpm >nul 2>&1
if !errorlevel! equ 0 goto :pnpm_ok

:pnpm_via_npm
echo   [*] Installing pnpm via npm...
call npm install -g pnpm
if !errorlevel! neq 0 (
    echo   [ERROR] Failed to install pnpm
    set "ERRORS=1"
)

:pnpm_ok
where pnpm >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%V in ('pnpm --version 2^>^&1') do echo   [OK] pnpm %%V
) else (
    echo   [ERROR] pnpm still not available
    set "ERRORS=1"
)

REM ============================================================
REM  STEP 3: Check ffmpeg
REM ============================================================
echo.
echo [3/6] Checking ffmpeg...

where ffmpeg >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] ffmpeg installed
    goto :ffmpeg_done
)

echo   [!] ffmpeg not found in PATH.
set "FFMPEG_FOUND="
for %%D in (
    "C:\ffmpeg\bin"
    "C:\ffmpeg"
    "%ProgramFiles%\ffmpeg\bin"
    "%LOCALAPPDATA%\ffmpeg\bin"
) do (
    if not defined FFMPEG_FOUND if exist "%%~D\ffmpeg.exe" set "FFMPEG_FOUND=%%~D"
)

if defined FFMPEG_FOUND (
    echo   [+] ffmpeg found: !FFMPEG_FOUND!
) else (
    echo   [WARN] ffmpeg not installed - some audio formats may not work.
    echo   [WARN] You can install later: winget install Gyan.FFmpeg
)

:ffmpeg_done

REM ============================================================
REM  STEP 4: Setup Python backend
REM ============================================================
echo.
echo [4/6] Setting up Python backend...

if not exist "%SCRIPT_DIR%backend" (
    echo   [ERROR] backend folder not found!
    set "ERRORS=1"
    goto :skip_backend
)
cd /d "%SCRIPT_DIR%backend"

if exist venv (
    echo   [OK] virtualenv already exists
    goto :venv_ready
)

echo   [*] Creating virtualenv...
python -m venv venv
if !errorlevel! neq 0 (
    echo   [!] First attempt failed, trying with ensurepip...
    python -m ensurepip >nul 2>&1
    python -m venv venv
)
if !errorlevel! neq 0 (
    echo   [ERROR] Cannot create venv.
    set "ERRORS=1"
    goto :skip_backend
)
echo   [OK] virtualenv created

:venv_ready
if not exist "venv\Scripts\activate.bat" (
    echo   [ERROR] venv\Scripts\activate.bat missing - corrupted venv?
    echo   [*] Removing and recreating...
    rmdir /s /q venv >nul 2>&1
    python -m venv venv
    if !errorlevel! neq 0 (
        echo   [ERROR] Cannot recreate venv.
        set "ERRORS=1"
        goto :skip_backend
    )
)

echo   [*] Activating virtualenv...
call venv\Scripts\activate.bat

echo   [*] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Detect GPU
echo   [*] Detecting GPU...
set "CUDA_AVAILABLE=0"
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    set "CUDA_AVAILABLE=1"
    echo   [OK] NVIDIA GPU detected - installing with CUDA 12.1
) else (
    echo   [!] No NVIDIA GPU - installing CPU-only version
)

REM Install PyTorch
echo   [*] Installing PyTorch - this may take a few minutes...
if !CUDA_AVAILABLE! equ 1 (
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
)
if !errorlevel! neq 0 (
    echo   [WARN] Pinned versions failed, trying without version pins...
    if !CUDA_AVAILABLE! equ 1 (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
    if !errorlevel! neq 0 (
        echo   [ERROR] Failed to install PyTorch!
        set "ERRORS=1"
    )
)

REM Install remaining dependencies
echo   [*] Installing remaining dependencies...

if not exist requirements.txt (
    echo   [WARN] requirements.txt not found - skipping
    goto :skip_requirements
)

REM Write filter script to temp file to avoid batch escaping issues
set "FILTER_SCRIPT=%TEMP%\filharmonia_filter_req.py"
> "!FILTER_SCRIPT!" (
    echo import sys
    echo try:
    echo     lines = open('requirements.txt', encoding='utf-8'^).readlines(^)
    echo except Exception as e:
    echo     print(f'  [ERROR] Cannot read requirements.txt: {e}'^)
    echo     sys.exit(1^)
    echo skip = ['torch==','torchaudio==','torchvision==','nvidia-cublas','nvidia-cudnn','tensorflow','keras==','tensorboard']
    echo filtered = [l for l in lines if not any(s in l for s in skip^) and l.strip(^) and not l.strip(^).startswith('#'^)]
    echo open('_req_filtered.txt','w',encoding='utf-8'^).writelines(filtered^)
    echo print(f'  Filtered: {len(lines^)} -> {len(filtered^)} packages'^)
)
python "!FILTER_SCRIPT!"
del /f "!FILTER_SCRIPT!" >nul 2>&1

if not exist _req_filtered.txt (
    echo   [WARN] Filtering requirements.txt failed
    goto :skip_requirements
)

pip install -r _req_filtered.txt 2>&1 | findstr /V "already satisfied"
if !errorlevel! neq 0 (
    echo   [WARN] Some packages failed. Installing one by one...
    for /f "usebackq delims=" %%L in ("_req_filtered.txt") do (
        set "PKG=%%L"
        if not "!PKG!"=="" if not "!PKG:~0,1!"=="#" pip install "!PKG!" >nul 2>&1
    )
)
del /f _req_filtered.txt >nul 2>&1

:skip_requirements
echo   [*] Verifying critical dependencies...
pip install aiosqlite polars cachetools fastapi uvicorn pydantic python-dotenv python-multipart >nul 2>&1

echo   [OK] Backend configured

:skip_backend
cd /d "%SCRIPT_DIR%"

REM ============================================================
REM  STEP 5: Setup frontend
REM ============================================================
echo.
echo [5/6] Setting up React frontend...

if not exist "%SCRIPT_DIR%frontend" (
    echo   [ERROR] frontend folder not found!
    set "ERRORS=1"
    goto :skip_frontend
)
cd /d "%SCRIPT_DIR%frontend"

set "PATH=%APPDATA%\npm;%ProgramFiles%\nodejs;!PATH!"

if exist node_modules (
    echo   [OK] node_modules already exists
    echo   [*] Updating dependencies...
)

where pnpm >nul 2>&1
if !errorlevel! equ 0 (
    pnpm install
) else (
    echo   [!] pnpm not available, using npm...
    npm install
)
if !errorlevel! neq 0 (
    echo   [ERROR] Failed to install frontend dependencies!
    set "ERRORS=1"
)

echo   [*] Checking TypeScript...
call npx tsc --noEmit >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] TypeScript - no errors
) else (
    echo   [WARN] TypeScript errors found - may need fixes
)

:skip_frontend
cd /d "%SCRIPT_DIR%"

REM ============================================================
REM  STEP 6: Environment file
REM ============================================================
echo.
echo [6/6] Configuring .env file...

if exist .env (
    echo   [OK] .env already exists
    goto :env_done
)
if exist .env.example (
    copy .env.example .env >nul
    echo   [OK] Created .env from template
    echo   [!] IMPORTANT: Edit .env and set FILHARMONIA_BASE_DIR
    goto :env_done
)
> .env echo FILHARMONIA_BASE_DIR=D:\FILHARMONIA
>> .env echo HOST=0.0.0.0
>> .env echo PORT=8000
echo   [OK] Created minimal .env

:env_done

REM ============================================================
REM  VERIFICATION
REM ============================================================
echo.
echo  =========================================
echo   Verifying installation...
echo  =========================================
echo.

cd /d "%SCRIPT_DIR%backend"
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

REM Write verification script to temp file
set "VERIFY_SCRIPT=%TEMP%\filharmonia_verify.py"
> "!VERIFY_SCRIPT!" (
    echo import sys
    echo ok, fail = 0, 0
    echo tests = [
    echo     ('fastapi', 'FastAPI'^),
    echo     ('uvicorn', 'Uvicorn'^),
    echo     ('torch', 'PyTorch'^),
    echo     ('torchaudio', 'torchaudio'^),
    echo     ('transformers', 'Transformers'^),
    echo     ('librosa', 'librosa'^),
    echo     ('polars', 'Polars'^),
    echo     ('aiosqlite', 'aiosqlite'^),
    echo     ('pydantic', 'Pydantic'^),
    echo     ('dotenv', 'python-dotenv'^),
    echo     ('cachetools', 'cachetools'^),
    echo ]
    echo for mod, name in tests:
    echo     try:
    echo         __import__(mod^)
    echo         print(f'  [OK] {name}'^)
    echo         ok += 1
    echo     except ImportError:
    echo         print(f'  [!!] {name} - MISSING'^)
    echo         fail += 1
    echo print(^)
    echo try:
    echo     import torch
    echo     print(f'  PyTorch {torch.__version__}'^)
    echo     if torch.cuda.is_available(^):
    echo         print(f'  GPU: {torch.cuda.get_device_name(0^)}'^)
    echo     else:
    echo         print('  GPU: none (CPU mode^)'^)
    echo except Exception:
    echo     pass
    echo print(^)
    echo print(f'  Result: {ok}/{ok+fail} packages OK'^)
    echo if fail ^> 0:
    echo     print(f'  {fail} packages missing - manual install may be needed'^)
)
python "!VERIFY_SCRIPT!"
del /f "!VERIFY_SCRIPT!" >nul 2>&1

cd /d "%SCRIPT_DIR%"

echo.
if !ERRORS! equ 0 (
    echo  =========================================
    echo   SETUP COMPLETED SUCCESSFULLY!
    echo  =========================================
    echo.
    echo   Next steps:
    echo   1. Edit .env - set FILHARMONIA_BASE_DIR
    echo   2. Run start.bat
    echo   3. Open http://localhost:5173
) else (
    echo  =========================================
    echo   SETUP COMPLETED WITH WARNINGS
    echo  =========================================
    echo.
    echo   Some components need attention.
    echo   Check the messages above.
)
echo.
pause
exit /b !ERRORS!

:fail
echo.
echo   Setup aborted. Fix the issues above and run again.
echo.
pause
exit /b 1

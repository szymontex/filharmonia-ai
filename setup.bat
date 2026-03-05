@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title Filharmonia AI - Setup

echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║     Filharmonia AI - Setup Wizard         ║
echo  ║     Automatyczna instalacja               ║
echo  ╚═══════════════════════════════════════════╝
echo.

set "ERRORS=0"
set "SCRIPT_DIR=%~dp0"

REM ============================================================
REM  STEP 1: Check/Install Python
REM ============================================================
echo [1/6] Sprawdzanie Python...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo   [!] Python nie znaleziony w PATH.
    echo   [!] Szukam w typowych lokalizacjach...

    REM Check common install locations
    set "PYTHON_FOUND="
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
        "%ProgramFiles%\Python313\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
    ) do (
        if exist %%P (
            echo   [+] Znaleziony: %%P
            set "PYTHON_FOUND=%%~P"
            goto :python_found
        )
    )

    echo.
    echo   ╔══════════════════════════════════════════════════════════╗
    echo   ║  PYTHON NIE ZAINSTALOWANY                               ║
    echo   ║                                                          ║
    echo   ║  1. Otworz: https://www.python.org/downloads/           ║
    echo   ║  2. Pobierz Python 3.11 lub nowszy                      ║
    echo   ║  3. WAZNE: Zaznacz "Add Python to PATH" przy instalacji ║
    echo   ║  4. Uruchom setup.bat ponownie                          ║
    echo   ╚══════════════════════════════════════════════════════════╝
    echo.
    echo   Otwieram strone pobierania...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

:python_found
REM If python not in PATH but found on disk, add to PATH for this session
if defined PYTHON_FOUND (
    for %%F in ("%PYTHON_FOUND%") do set "PYTHON_DIR=%%~dpF"
    set "PATH=!PYTHON_DIR!;!PYTHON_DIR!Scripts;!PATH!"
    echo   [+] Dodano do PATH na czas instalacji: !PYTHON_DIR!
)

REM Verify Python version >= 3.10
for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
echo   [OK] Python %PYVER%

for /f "tokens=1,2 delims=." %%A in ("%PYVER%") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)
if %PY_MAJOR% lss 3 (
    echo   [BLAD] Wymagany Python 3.10+, znaleziony %PYVER%
    pause
    exit /b 1
)
if %PY_MINOR% lss 10 (
    echo   [BLAD] Wymagany Python 3.10+, znaleziony %PYVER%
    pause
    exit /b 1
)

REM ============================================================
REM  STEP 2: Check/Install Node.js + pnpm
REM ============================================================
echo.
echo [2/6] Sprawdzanie Node.js i pnpm...

where node >nul 2>&1
if %errorlevel% neq 0 (
    REM Check common locations
    set "NODE_FOUND="
    for %%N in (
        "%ProgramFiles%\nodejs\node.exe"
        "%ProgramFiles(x86)%\nodejs\node.exe"
        "%LOCALAPPDATA%\Programs\nodejs\node.exe"
    ) do (
        if exist %%N (
            for %%F in (%%N) do set "NODE_DIR=%%~dpF"
            set "PATH=!NODE_DIR!;!PATH!"
            set "NODE_FOUND=1"
            echo   [+] Node.js znaleziony: %%N
            goto :node_check_done
        )
    )

    echo   [!] Node.js nie zainstalowany. Probuje zainstalowac przez winget...

    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [*] Instaluje Node.js LTS przez winget...
        winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
        if !errorlevel! equ 0 (
            echo   [OK] Node.js zainstalowany. Odswiezam PATH...
            REM Refresh PATH to pick up newly installed Node
            set "PATH=%ProgramFiles%\nodejs;%APPDATA%\npm;!PATH!"
            goto :node_check_done
        ) else (
            echo   [!] winget nie mogl zainstalowac Node.js
        )
    )

    echo.
    echo   ╔══════════════════════════════════════════════════════════╗
    echo   ║  NODE.JS NIE ZAINSTALOWANY                              ║
    echo   ║                                                          ║
    echo   ║  1. Otworz: https://nodejs.org/                         ║
    echo   ║  2. Pobierz wersje LTS (22.x)                          ║
    echo   ║  3. Zainstaluj z domyslnymi opcjami                     ║
    echo   ║  4. Uruchom setup.bat ponownie                          ║
    echo   ╚══════════════════════════════════════════════════════════╝
    echo.
    start https://nodejs.org/
    pause
    exit /b 1
)

:node_check_done
for /f "tokens=*" %%V in ('node --version 2^>^&1') do echo   [OK] Node.js %%V

REM Ensure npm is available
set "PATH=%APPDATA%\npm;%PATH%"

REM Check pnpm
where pnpm >nul 2>&1
if %errorlevel% neq 0 (
    echo   [!] pnpm nie zainstalowany. Instaluje...

    REM Try corepack first (built into Node 16.13+)
    where corepack >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [*] Aktywuje pnpm przez corepack...
        corepack enable
        corepack prepare pnpm@latest --activate >nul 2>&1
    )

    REM Check again
    where pnpm >nul 2>&1
    if !errorlevel! neq 0 (
        echo   [*] Instaluje pnpm przez npm...
        call npm install -g pnpm
        if !errorlevel! neq 0 (
            echo   [BLAD] Nie udalo sie zainstalowac pnpm!
            set "ERRORS=1"
        )
    )
)

where pnpm >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%V in ('pnpm --version 2^>^&1') do echo   [OK] pnpm %%V
) else (
    echo   [BLAD] pnpm nadal niedostepny
    set "ERRORS=1"
)

REM ============================================================
REM  STEP 3: Check ffmpeg
REM ============================================================
echo.
echo [3/6] Sprawdzanie ffmpeg...

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo   [!] ffmpeg nie znaleziony w PATH.
    echo   [!] Szukam w typowych lokalizacjach...

    set "FFMPEG_FOUND="
    for %%F in (
        "C:\ffmpeg\bin\ffmpeg.exe"
        "C:\ffmpeg\ffmpeg.exe"
        "%ProgramFiles%\ffmpeg\bin\ffmpeg.exe"
        "%LOCALAPPDATA%\ffmpeg\bin\ffmpeg.exe"
    ) do (
        if exist %%F (
            echo   [+] Znaleziony: %%F
            set "FFMPEG_FOUND=1"
            goto :ffmpeg_done
        )
    )

    echo   [UWAGA] ffmpeg nie zainstalowany - niektorzy formaty audio moga nie dzialac.
    echo   [UWAGA] Mozesz zainstalowac pozniej: winget install Gyan.FFmpeg
    echo   [UWAGA] Kontynuuje bez ffmpeg...
) else (
    echo   [OK] ffmpeg zainstalowany
)
:ffmpeg_done

REM ============================================================
REM  STEP 4: Setup Python backend
REM ============================================================
echo.
echo [4/6] Konfiguracja backendu Python...

cd /d "%SCRIPT_DIR%backend"

if not exist venv (
    echo   [*] Tworzenie virtualenv...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo   [BLAD] Nie udalo sie utworzyc virtualenv!
        echo   [*] Probuje: python -m ensurepip...
        python -m ensurepip
        python -m venv venv
        if !errorlevel! neq 0 (
            echo   [BLAD] Definitywnie nie mozna utworzyc venv.
            set "ERRORS=1"
            goto :skip_backend
        )
    )
    echo   [OK] virtualenv utworzony
) else (
    echo   [OK] virtualenv juz istnieje
)

echo   [*] Aktywacja virtualenv...
call venv\Scripts\activate.bat

REM Upgrade pip first
echo   [*] Aktualizacja pip...
python -m pip install --upgrade pip >nul 2>&1

REM Detect GPU
echo   [*] Wykrywanie GPU...
set "CUDA_AVAILABLE=0"
python -c "import subprocess; r=subprocess.run(['nvidia-smi'],capture_output=True); exit(0 if r.returncode==0 else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    set "CUDA_AVAILABLE=1"
    echo   [OK] NVIDIA GPU wykryte - instalacja z CUDA 12.1
) else (
    echo   [!] Brak NVIDIA GPU - instalacja wersji CPU-only
)

REM Install PyTorch
echo   [*] Instalacja PyTorch (to moze potrwac kilka minut)...
if %CUDA_AVAILABLE% equ 1 (
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
)
if !errorlevel! neq 0 (
    echo   [UWAGA] PyTorch instalacja z pinami nie powiodla sie.
    echo   [*] Probuje bez pinow wersji...
    if %CUDA_AVAILABLE% equ 1 (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
    if !errorlevel! neq 0 (
        echo   [BLAD] Nie udalo sie zainstalowac PyTorch!
        set "ERRORS=1"
    )
)

REM Install remaining dependencies (skip torch lines - already installed)
echo   [*] Instalacja pozostalych zaleznosci...

REM Create a filtered requirements without torch/nvidia lines
python -c "
lines = open('requirements.txt').readlines()
skip = ['torch==', 'torchaudio==', 'torchvision==', 'nvidia-cublas', 'nvidia-cudnn', 'tensorflow', 'keras==', 'tensorboard']
filtered = [l for l in lines if not any(s in l for s in skip) and not l.strip().startswith('#')]
open('_requirements_filtered.txt','w').writelines(filtered)
print(f'  Filtered: {len(lines)} -> {len(filtered)} packages')
"

pip install -r _requirements_filtered.txt 2>&1 | findstr /V "already satisfied"
if !errorlevel! neq 0 (
    echo   [UWAGA] Niektorych pakietow nie udalo sie zainstalowac.
    echo   [*] Probuje instalowac po jednym (wolniej ale bezpieczniej)...
    for /f "usebackq delims=" %%L in ("_requirements_filtered.txt") do (
        set "PKG=%%L"
        if not "!PKG!"=="" if not "!PKG:~0,1!"=="#" (
            pip install "!PKG!" >nul 2>&1
        )
    )
)
del /f _requirements_filtered.txt >nul 2>&1

REM Ensure new critical deps are installed (even if requirements.txt failed)
echo   [*] Weryfikacja krytycznych zaleznosci...
pip install aiosqlite polars cachetools fastapi uvicorn pydantic python-dotenv python-multipart >nul 2>&1

echo   [OK] Backend skonfigurowany

:skip_backend
cd /d "%SCRIPT_DIR%"

REM ============================================================
REM  STEP 5: Setup frontend
REM ============================================================
echo.
echo [5/6] Konfiguracja frontendu React...

cd /d "%SCRIPT_DIR%frontend"

REM Ensure pnpm is available
set "PATH=%APPDATA%\npm;%ProgramFiles%\nodejs;%PATH%"

if exist node_modules (
    echo   [OK] node_modules juz istnieje
    echo   [*] Aktualizacja zaleznosci...
)
pnpm install
if !errorlevel! neq 0 (
    echo   [UWAGA] pnpm install nie powiodl sie, probuje npm...
    npm install
    if !errorlevel! neq 0 (
        echo   [BLAD] Nie udalo sie zainstalowac zaleznosci frontendu!
        set "ERRORS=1"
    )
)

REM Quick build test
echo   [*] Sprawdzanie TypeScript...
npx tsc --noEmit >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] TypeScript - brak bledow
) else (
    echo   [UWAGA] Sa bledy TypeScript - moze wymagac poprawek
)

cd /d "%SCRIPT_DIR%"

REM ============================================================
REM  STEP 6: Environment file
REM ============================================================
echo.
echo [6/6] Konfiguracja pliku .env...

if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo   [OK] Utworzono .env z szablonu
        echo   [!] WAZNE: Edytuj .env i ustaw FILHARMONIA_BASE_DIR
    ) else (
        echo   [UWAGA] Brak .env.example - tworze minimalny .env
        (
            echo FILHARMONIA_BASE_DIR=D:\FILHARMONIA
            echo HOST=0.0.0.0
            echo PORT=8000
        ) > .env
        echo   [OK] Utworzono minimalny .env
    )
) else (
    echo   [OK] .env juz istnieje
)

REM ============================================================
REM  VERIFICATION
REM ============================================================
echo.
echo  ========================================
echo   Weryfikacja instalacji...
echo  ========================================
echo.

cd /d "%SCRIPT_DIR%backend"
call venv\Scripts\activate.bat

python -c "
import sys
ok, fail = 0, 0
tests = [
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('torch', 'PyTorch'),
    ('torchaudio', 'torchaudio'),
    ('transformers', 'Transformers'),
    ('librosa', 'librosa'),
    ('polars', 'Polars'),
    ('aiosqlite', 'aiosqlite'),
    ('pydantic', 'Pydantic'),
    ('dotenv', 'python-dotenv'),
    ('cachetools', 'cachetools'),
]
for mod, name in tests:
    try:
        __import__(mod)
        print(f'  [OK] {name}')
        ok += 1
    except ImportError:
        print(f'  [!!] {name} - BRAK')
        fail += 1

print()
# PyTorch details
try:
    import torch
    print(f'  PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print(f'  GPU: brak (tryb CPU)')
except:
    pass

print()
print(f'  Wynik: {ok}/{ok+fail} pakietow OK')
if fail > 0:
    print(f'  {fail} pakietow brakuje - moze byc potrzebna reczna instalacja')
    sys.exit(1)
"

cd /d "%SCRIPT_DIR%"

echo.
if %ERRORS% equ 0 (
    echo  ╔═══════════════════════════════════════════╗
    echo  ║  SETUP ZAKONCZONY POMYSLNIE!              ║
    echo  ╠═══════════════════════════════════════════╣
    echo  ║                                           ║
    echo  ║  Nastepne kroki:                          ║
    echo  ║  1. Edytuj .env (ustaw FILHARMONIA_BASE_DIR) ║
    echo  ║  2. Uruchom start.bat                     ║
    echo  ║  3. Otworz http://localhost:5173           ║
    echo  ║                                           ║
    echo  ╚═══════════════════════════════════════════╝
) else (
    echo  ╔═══════════════════════════════════════════╗
    echo  ║  SETUP ZAKONCZONY Z UWAGAMI               ║
    echo  ╠═══════════════════════════════════════════╣
    echo  ║                                           ║
    echo  ║  Niektore elementy wymagaja uwagi.        ║
    echo  ║  Sprawdz powyzsze komunikaty.             ║
    echo  ║                                           ║
    echo  ╚═══════════════════════════════════════════╝
)

echo.
pause

#!/usr/bin/env bash
# ========================================
#  Filharmonia AI - Setup Script
#  macOS / Linux - One-Click Setup
# ========================================

# Do NOT use set -e — we handle errors ourselves
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ERRORS=0

# ========================================
# Helper functions
# ========================================

info()  { echo "[INFO] $*"; }
ok()    { echo "[OK]   $*"; }
warn()  { echo "[WARN] $*"; }
fail()  { echo "[FAIL] $*"; ERRORS=$((ERRORS + 1)); }
fatal() { echo "[FATAL] $*"; exit 1; }

detect_os() {
    case "$(uname -s)" in
        Darwin) OS="macos" ;;
        Linux)
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                case "$ID" in
                    ubuntu|debian|pop|linuxmint|elementary) OS="debian" ;;
                    fedora|rhel|centos|rocky|alma)          OS="fedora" ;;
                    arch|manjaro|endeavouros)                OS="arch" ;;
                    *)                                       OS="linux" ;;
                esac
            else
                OS="linux"
            fi
            ;;
        *) OS="unknown" ;;
    esac
    echo "$OS"
}

version_gte() {
    # Returns 0 if $1 >= $2 (version comparison)
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

version_lt() {
    ! version_gte "$1" "$2"
}

# Python 3.13+ removed stdlib modules (aifc, audioop) required by audio libraries.
# This project requires Python 3.10-3.12.
PYTHON_MIN="3.10.0"
PYTHON_MAX="3.13.0"  # exclusive upper bound

python_version_ok() {
    local ver="$1"
    version_gte "$ver" "$PYTHON_MIN" && version_lt "$ver" "$PYTHON_MAX"
}

find_compatible_python() {
    local candidate ver

    # 1. Check pyenv versions directory directly (most reliable on dev machines)
    if [ -d "$HOME/.pyenv/versions" ]; then
        local pyenv_dir
        for pyenv_dir in $(ls -d "$HOME/.pyenv/versions"/3.12.* \
                                  "$HOME/.pyenv/versions"/3.11.* \
                                  "$HOME/.pyenv/versions"/3.10.* 2>/dev/null | sort -rV); do
            candidate="$pyenv_dir/bin/python3"
            if [ -x "$candidate" ]; then
                ver="$("$candidate" --version 2>/dev/null | cut -d' ' -f2)"
                if python_version_ok "$ver"; then
                    echo "$candidate"
                    return 0
                fi
            fi
        done
    fi

    # 2. Check Homebrew installations directly (macOS — brew may not link to PATH)
    local brew_prefix
    for brew_prefix in /opt/homebrew/opt /usr/local/opt; do
        for suffix in 3.12 3.11 3.10; do
            candidate="$brew_prefix/python@$suffix/bin/python$suffix"
            if [ -x "$candidate" ]; then
                ver="$("$candidate" --version 2>/dev/null | cut -d' ' -f2)"
                if python_version_ok "$ver"; then
                    echo "$candidate"
                    return 0
                fi
            fi
        done
    done

    # 3. Check PATH for versioned executables (python3.12, python3.11, python3.10)
    local suffix
    for suffix in 3.12 3.11 3.10; do
        candidate="$(command -v "python$suffix" 2>/dev/null || true)"
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            ver="$("$candidate" --version 2>/dev/null | cut -d' ' -f2)"
            if python_version_ok "$ver"; then
                echo "$candidate"
                return 0
            fi
        fi
    done

    # 3. Check unversioned python3 (might be compatible)
    candidate="$(command -v python3 2>/dev/null || true)"
    if [ -n "$candidate" ]; then
        ver="$("$candidate" --version 2>/dev/null | cut -d' ' -f2)"
        if python_version_ok "$ver"; then
            echo "$candidate"
            return 0
        fi
    fi

    return 1
}

install_compatible_python() {
    # Install a compatible Python via pyenv

    # Ensure pyenv is available
    if ! command -v pyenv &>/dev/null; then
        case "$OS" in
            macos)
                if command -v brew &>/dev/null; then
                    info "Installing pyenv via Homebrew..."
                    brew install pyenv || return 1
                else
                    info "Installing pyenv via installer..."
                    curl -fsSL https://pyenv.run | bash || return 1
                fi
                ;;
            *)
                info "Installing pyenv..."
                curl -fsSL https://pyenv.run | bash || return 1
                ;;
        esac

        # Initialize pyenv for this session
        export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
        [ -d "$PYENV_ROOT/bin" ] && export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init - 2>/dev/null)" || true
    fi

    # Install build dependencies (Linux only — macOS has Xcode CLT)
    case "$OS" in
        debian)
            info "Installing Python build dependencies..."
            sudo apt-get update -qq 2>/dev/null
            sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
                libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev \
                xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev 2>/dev/null
            ;;
        fedora)
            info "Installing Python build dependencies..."
            sudo dnf install -y gcc make zlib-devel bzip2-devel readline-devel \
                sqlite-devel openssl-devel tk-devel libffi-devel xz-devel 2>/dev/null
            ;;
        arch)
            info "Installing Python build dependencies..."
            sudo pacman -S --noconfirm --needed base-devel openssl zlib 2>/dev/null
            ;;
    esac

    # Try multiple versions in preference order
    local target
    for target in 3.12.12 3.12.11 3.12.10 3.11.11 3.11.10 3.10.16; do
        info "Installing Python $target via pyenv..."
        if pyenv install -s "$target" 2>&1; then
            local installed="$HOME/.pyenv/versions/$target/bin/python3"
            if [ -x "$installed" ]; then
                ok "Python $target installed via pyenv"
                echo "$installed"
                return 0
            fi
        fi
        warn "Python $target installation failed, trying next version..."
    done

    return 1
}

# ========================================
# Start
# ========================================

echo ""
echo "========================================"
echo "  Filharmonia AI - Setup"
echo "========================================"
echo ""

OS=$(detect_os)
info "Detected platform: $OS ($(uname -s) $(uname -m))"
echo ""

# ========================================
# [1/7] Prerequisites
# ========================================

echo "[1/9] Checking prerequisites..."
echo ""

# --- Python (require 3.10-3.12 for audio library compatibility) ---
PYTHON_CMD="$(find_compatible_python || true)"

if [ -z "$PYTHON_CMD" ]; then
    # No compatible Python found — check what's available and try to fix it
    SYS_VER="$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "none")"

    if [ "$SYS_VER" != "none" ] && version_gte "$SYS_VER" "$PYTHON_MAX"; then
        warn "System Python $SYS_VER is too new — audio libraries require Python 3.10-3.12"

        # macOS: try Homebrew first (prebuilt binary, no compilation needed)
        if [ "$OS" = "macos" ] && command -v brew &>/dev/null; then
            info "Installing Python 3.12 via Homebrew (prebuilt, fast)..."
            brew install python@3.12 2>/dev/null || brew install python@3.11 2>/dev/null || true
            PYTHON_CMD="$(find_compatible_python || true)"
        fi

        # Fallback to pyenv (compiles from source — slower, needs build deps)
        if [ -z "$PYTHON_CMD" ]; then
            info "Attempting to install a compatible Python via pyenv..."
            PYTHON_CMD="$(install_compatible_python || true)"
        fi
    elif [ "$SYS_VER" = "none" ]; then
        info "Python not found — attempting to install..."
        case "$OS" in
            debian) sudo apt-get update -qq && sudo apt-get install -y python3 python3-venv python3-pip ;;
            fedora) sudo dnf install -y python3 python3-pip ;;
            arch)   sudo pacman -S --noconfirm python python-pip ;;
            macos)
                if command -v brew &>/dev/null; then
                    brew install python@3.12 || brew install python@3.11 || brew install python@3
                else
                    fatal "Python not found. Install Homebrew (https://brew.sh) then run: brew install python@3.12"
                fi
                ;;
            *) fatal "Python not found. Please install Python 3.10-3.12 manually." ;;
        esac
        PYTHON_CMD="$(find_compatible_python || true)"
    else
        fatal "Python $SYS_VER is too old. This project requires Python 3.10-3.12."
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    fatal "No compatible Python (3.10-3.12) found.
  Python 3.13+ removed stdlib modules (aifc, audioop) required by audio libraries.

  Install a compatible version:
    pyenv install 3.12.12    # any OS with pyenv
    brew install python@3.12 # macOS with Homebrew

  Then re-run: ./setup.sh"
fi

PYTHON_VER=$("$PYTHON_CMD" --version | cut -d' ' -f2)
PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f1-2)

ok "Python $PYTHON_VER ($PYTHON_CMD)"

# --- python3-venv (Debian/Ubuntu needs it as a separate package) ---
if [ "$OS" = "debian" ]; then
    if ! "$PYTHON_CMD" -m venv --help &>/dev/null; then
        info "Installing python3-venv..."
        sudo apt-get install -y python3-venv
    fi
fi

# --- System libraries (libsndfile for audio processing) ---
if [ "$OS" != "macos" ]; then
    # Check for libsndfile (required by soundfile/librosa)
    if ! ldconfig -p 2>/dev/null | grep -q libsndfile; then
        info "Installing libsndfile (required for audio processing)..."
        case "$OS" in
            debian) sudo apt-get install -y libsndfile1 ;;
            fedora) sudo dnf install -y libsndfile ;;
            arch)   sudo pacman -S --noconfirm libsndfile ;;
            *)      warn "Please install libsndfile manually" ;;
        esac
    fi
fi

# --- Node.js ---
if ! command -v node &>/dev/null; then
    info "Node.js not found — attempting to install..."
    case "$OS" in
        debian)
            if command -v curl &>/dev/null; then
                curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
                sudo apt-get install -y nodejs
            else
                sudo apt-get install -y nodejs npm
            fi
            ;;
        fedora) sudo dnf install -y nodejs npm ;;
        arch)   sudo pacman -S --noconfirm nodejs npm ;;
        macos)
            if command -v brew &>/dev/null; then
                brew install node
            else
                fatal "Node.js not found. Install via: brew install node"
            fi
            ;;
        *) fatal "Node.js not found. Please install Node.js 18+ manually." ;;
    esac
fi

if ! command -v node &>/dev/null; then
    fatal "Node.js installation failed. Please install manually."
fi

NODE_VER=$(node --version | tr -d 'v')
if ! version_gte "$NODE_VER" "18.0.0"; then
    fatal "Node.js >= 18 required, found v$NODE_VER"
fi
ok "Node.js v$NODE_VER"

# --- pnpm ---
if ! command -v pnpm &>/dev/null; then
    info "Installing pnpm..."
    # Try corepack first (built into Node 16+), fall back to npm
    if command -v corepack &>/dev/null; then
        corepack enable && corepack prepare pnpm@latest --activate 2>/dev/null \
            || npm install -g pnpm 2>/dev/null \
            || fatal "Could not install pnpm. Try manually: npm install -g pnpm"
    else
        npm install -g pnpm 2>/dev/null \
            || sudo npm install -g pnpm 2>/dev/null \
            || fatal "Could not install pnpm. Try manually: sudo npm install -g pnpm"
    fi
fi

if command -v pnpm &>/dev/null; then
    ok "pnpm $(pnpm --version)"
else
    fatal "pnpm installation failed"
fi

echo ""

# ========================================
# [2/7] Backend - Virtual Environment
# ========================================

echo "[2/9] Setting up Python backend..."

cd "$SCRIPT_DIR/backend"

# Check if existing venv uses a compatible Python
if [ -d "venv" ]; then
    VENV_PY_VER="$(venv/bin/python3 --version 2>/dev/null | cut -d' ' -f2 || echo "0.0.0")"
    if ! python_version_ok "$VENV_PY_VER"; then
        warn "Existing venv uses Python $VENV_PY_VER (incompatible) — recreating..."
        rm -rf venv
    fi
fi

if [ ! -d "venv" ]; then
    info "Creating Python virtual environment with Python $PYTHON_VER..."
    "$PYTHON_CMD" -m venv venv || fatal "Failed to create virtual environment"
fi

# Activate venv
source venv/bin/activate || fatal "Failed to activate virtual environment"

# Upgrade pip to avoid install issues
pip install --upgrade pip --quiet --disable-pip-version-check 2>/dev/null

echo ""

# ========================================
# [3/7] Detect Hardware (GPU/CPU)
# ========================================

echo "[3/9] Detecting hardware..."

DEVICE="CPU"
CUDA_VER=""

# Detect GPU via nvidia-smi BEFORE torch is installed
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        DEVICE="CUDA"
        # Extract CUDA version from nvidia-smi
        CUDA_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p')
        info "NVIDIA GPU detected (CUDA $CUDA_VER)"
    fi
fi

if [ "$DEVICE" = "CPU" ]; then
    info "No NVIDIA GPU detected — will use CPU-only PyTorch"
fi

echo ""

# ========================================
# [4/7] Install PyTorch
# ========================================

echo "[4/9] Installing PyTorch..."

if [ "$DEVICE" = "CUDA" ]; then
    # Pick the right CUDA wheel based on detected CUDA version
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    case "$CUDA_MAJOR" in
        12) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        11) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        *)  TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    esac
    info "Installing PyTorch with CUDA support ($TORCH_INDEX)..."
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    info "Installing PyTorch CPU-only..."
fi

# PyTorch wheels live on their own index — try preferred index, then fallback indexes
# IMPORTANT: torchvision/torchaudio MUST come from a PyTorch index, NOT vanilla PyPI
TORCH_INDEXES=(
    "$TORCH_INDEX"
    "https://download.pytorch.org/whl/cu124"
    "https://download.pytorch.org/whl/cu121"
    "https://download.pytorch.org/whl/cpu"
)

TORCH_INSTALLED=false
for idx in "${TORCH_INDEXES[@]}"; do
    if pip install torch torchaudio torchvision \
        --index-url "$idx" --quiet --disable-pip-version-check 2>/dev/null; then
        TORCH_INSTALLED=true
        ok "PyTorch installed from $idx"
        break
    fi
done

if [ "$TORCH_INSTALLED" = false ]; then
    fail "PyTorch installation failed from all indexes"
fi

echo ""

# ========================================
# [5/7] Install Backend Dependencies
# ========================================

echo "[5/9] Installing backend dependencies..."

# Filter out torch lines (handled in step 4) and comments
FILTERED_REQ=$(mktemp)
grep -v -E '^(torch|torchaudio|torchvision|nvidia-)' \
    requirements.txt \
    | grep -v '^\s*#' \
    | grep -v '^\s*$' \
    > "$FILTERED_REQ"

pip install -r "$FILTERED_REQ" --quiet --disable-pip-version-check \
    || { fail "Some backend dependencies failed to install"; }
rm -f "$FILTERED_REQ"

echo ""

# ========================================
# [6/7] ONNX Optimization (optional)
# ========================================

echo "[6/9] Optimizing for CPU inference..."

if [ "$DEVICE" = "CPU" ]; then
    info "Exporting ONNX INT8 model (takes ~30 seconds)..."
    if python -m scripts.export_onnx &>/dev/null; then
        ok "ONNX model exported — CPU inference 3x faster!"
    else
        warn "ONNX export failed — will use PyTorch fallback (still works, just slower)"
    fi
else
    info "GPU detected — skipping ONNX export (not needed)"
fi

cd "$SCRIPT_DIR"
echo ""

# ========================================
# [7/7] Frontend Setup
# ========================================

echo "[7/9] Setting up React frontend..."

cd "$SCRIPT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    info "Installing frontend dependencies..."
    pnpm install --silent || pnpm install || fail "Frontend dependency installation failed"
else
    info "node_modules already exists — skipping (run 'pnpm install' in frontend/ to update)"
fi

cd "$SCRIPT_DIR"
echo ""

# ========================================
# [8/9] Data Directory Structure
# ========================================

echo "[8/9] Setting up data directory structure..."

# Determine base directory from .env or use default
if [ -f "$SCRIPT_DIR/.env" ]; then
    FILHARMONIA_BASE=$(grep -E '^FILHARMONIA_BASE_DIR=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
fi
if [ -z "$FILHARMONIA_BASE" ]; then
    FILHARMONIA_BASE="$SCRIPT_DIR/FILHARMONIA_DATA"
fi

# Create required directories
for dir in SORTED NAGRANIA_KONCERTOW TRAINING_DATA/DATA RECOGNITION_MODELS ML_EXPERIMENTS/datasets; do
    mkdir -p "$FILHARMONIA_BASE/$dir"
done

# Create training data class subdirectories
for cls in APPLAUSE MUSIC PUBLIC SPEECH TUNING; do
    mkdir -p "$FILHARMONIA_BASE/TRAINING_DATA/DATA/$cls"
done

ok "Data directories ready at $FILHARMONIA_BASE"
echo ""

# ========================================
# [9/9] Download Pre-trained Model
# ========================================

echo "[9/9] Downloading pre-trained model..."

MODEL_DIR="$FILHARMONIA_BASE/RECOGNITION_MODELS"
MODEL_FILE="$MODEL_DIR/ast_active.pth"

if [ -f "$MODEL_FILE" ]; then
    info "Model already exists at $MODEL_FILE — skipping download"
else
    HF_MODEL_URL="https://huggingface.co/szymontex/filharmonia-ast/resolve/main/ast_20251009_222204.pth"
    info "Downloading AST model from HuggingFace (~1GB)..."

    # Try wget, then curl
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$MODEL_FILE" "$HF_MODEL_URL" \
            || { rm -f "$MODEL_FILE"; fail "Model download failed"; }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$MODEL_FILE" "$HF_MODEL_URL" \
            || { rm -f "$MODEL_FILE"; fail "Model download failed"; }
    else
        fail "Neither wget nor curl found — cannot download model"
    fi

    if [ -f "$MODEL_FILE" ]; then
        ok "Model downloaded to $MODEL_FILE"
    fi
fi

echo ""

# ========================================
# Verify Installation
# ========================================

echo "========================================"
echo "  Verifying Installation"
echo "========================================"
echo ""

cd "$SCRIPT_DIR/backend"
source venv/bin/activate
python "$SCRIPT_DIR/verify_installation.py" || warn "Some verification checks failed (see above)"
cd "$SCRIPT_DIR"

# ========================================
# Environment Setup
# ========================================

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo ""
    info "Creating .env configuration file..."
    cp .env.example .env
    ok "Created .env — edit it to configure data directories"
fi

# Create setup marker
date > .setup_complete

# ========================================
# Summary
# ========================================

echo ""
echo "========================================"
if [ "$ERRORS" -eq 0 ]; then
    echo "  Setup Complete!"
else
    echo "  Setup Complete (with $ERRORS warnings)"
fi
echo "========================================"
echo ""
echo "Platform: $OS ($(uname -m))"
echo "Hardware: $DEVICE"

if [ "$DEVICE" = "CUDA" ]; then
    echo "CUDA:     $CUDA_VER"
    echo "Accel:    torch.compile (GPU)"
else
    if [ -f "backend/recognition_models/ast_active_int8.onnx" ]; then
        echo "Accel:    ONNX INT8 (3x CPU speedup)"
    else
        echo "Accel:    PyTorch CPU (fallback)"
    fi
fi

echo ""
echo "Next step:"
echo "  ./start.sh"
echo ""
echo "The app will open at: http://localhost:5173"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    exit 1
fi

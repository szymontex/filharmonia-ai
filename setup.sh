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

# --- Python (prefer 3.11 for maximum compatibility) ---
PYTHON_CMD=""

# Check for python3.11 first (pyenv, system, brew)
for candidate in \
    "$HOME/.pyenv/versions/3.11.11/bin/python3.11" \
    "$HOME/.pyenv/versions/3.11.*/bin/python3.11" \
    "$(command -v python3.11 2>/dev/null)"; do
    if [ -x "$candidate" ] 2>/dev/null; then
        PYTHON_CMD="$candidate"
        break
    fi
done

# Fall back to python3 if no 3.11 found
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    fi
fi

# Still no Python — try to install
if [ -z "$PYTHON_CMD" ]; then
    info "Python not found — attempting to install..."
    case "$OS" in
        debian) sudo apt-get update -qq && sudo apt-get install -y python3 python3-venv python3-pip ;;
        fedora) sudo dnf install -y python3 python3-pip ;;
        arch)   sudo pacman -S --noconfirm python python-pip ;;
        macos)
            if command -v brew &>/dev/null; then
                brew install python@3.11 || brew install python@3
            else
                fatal "Python not found. Install Homebrew (https://brew.sh) then run: brew install python@3.11"
            fi
            ;;
        *) fatal "Python not found. Please install Python 3.10-3.12 manually." ;;
    esac
    PYTHON_CMD="python3"
fi

if ! "$PYTHON_CMD" --version &>/dev/null; then
    fatal "Python installation failed. Please install Python 3.10-3.12 manually."
fi

PYTHON_VER=$("$PYTHON_CMD" --version | cut -d' ' -f2)
PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f1-2)

if ! version_gte "$PYTHON_VER" "3.10.0"; then
    fatal "Python >= 3.10 required, found $PYTHON_VER"
fi

# Warn about Python 3.13+ compatibility issues
if version_gte "$PYTHON_VER" "3.13.0"; then
    warn "Python $PYTHON_VER detected — some packages may have compatibility issues"
    warn "Recommended: Python 3.10-3.12 (install via pyenv: pyenv install 3.11.11)"
fi

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

if [ ! -d "venv" ]; then
    info "Creating Python virtual environment..."
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
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
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

# Filter out platform-specific packages from requirements.txt
# These are handled separately or are not needed on all platforms
FILTERED_REQ=$(mktemp)
grep -v -E '^(torch==|torchaudio==|torchvision==|nvidia-|tensorflow|keras==|tensorboard|tensorflow-intel|tensorflow-estimator|tensorflow-io-gcs-filesystem|libclang)' \
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

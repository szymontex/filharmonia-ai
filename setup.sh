#!/usr/bin/env bash
# ========================================
#  Filharmonia AI - Ultimate Setup Script
#  macOS / Linux - One-Click Setup
# ========================================

set -e  # Exit on error

echo ""
echo "========================================"
echo "  Filharmonia AI - Setup"
echo "========================================"
echo ""
echo "This will install everything needed."
echo "Estimated time: 3-5 minutes"
echo ""

# ========================================
# Check Prerequisites
# ========================================

echo "[1/7] Checking prerequisites..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found!"
    echo ""
    echo "Install Python 3.10+:"
    echo "  macOS:  brew install python@3.10"
    echo "  Ubuntu: sudo apt install python3.10 python3.10-venv python3-pip"
    exit 1
fi

PYTHON_VER=$(python3 --version | cut -d' ' -f2)
echo "[OK] Python $PYTHON_VER found"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js not found!"
    echo ""
    echo "Install Node.js 18+:"
    echo "  macOS:  brew install node"
    echo "  Ubuntu: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs"
    exit 1
fi

NODE_VER=$(node --version)
echo "[OK] Node.js $NODE_VER found"

# Check/Install pnpm
if ! command -v pnpm &> /dev/null; then
    echo "[INFO] Installing pnpm..."
    npm install -g pnpm --silent
fi

PNPM_VER=$(pnpm --version)
echo "[OK] pnpm $PNPM_VER found"

echo ""

# ========================================
# Backend Setup
# ========================================

echo "[2/7] Setting up Python backend..."
cd backend

# Create venv
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Detect GPU
echo ""
echo "[3/7] Detecting hardware..."

# Try to detect CUDA
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="CUDA"
    echo "[INFO] NVIDIA GPU detected - will use CUDA acceleration"
else
    DEVICE="CPU"
    echo "[INFO] No GPU detected - will use CPU-only PyTorch"
fi

# Install PyTorch
echo ""
echo "[4/7] Installing PyTorch..."

if [ "$DEVICE" == "CUDA" ]; then
    echo "[INFO] Installing PyTorch with CUDA 12.1 support..."
    pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --quiet --disable-pip-version-check
else
    echo "[INFO] Installing PyTorch CPU-only..."
    pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --quiet --disable-pip-version-check
fi

# Install dependencies
echo ""
echo "[5/7] Installing backend dependencies..."
pip install -r requirements.txt --quiet --disable-pip-version-check

# Export ONNX model for CPU speedup
echo ""
echo "[6/7] Optimizing for CPU inference..."
echo "[INFO] Exporting ONNX INT8 model (takes ~30 seconds)..."

if python -m scripts.export_onnx &>/dev/null; then
    echo "[OK] ONNX model exported - CPU inference 3x faster!"
else
    echo "[WARN] ONNX export failed - will use PyTorch fallback"
    echo "[INFO] This is OK - app will work but CPU inference will be slower"
fi

cd ..

# ========================================
# Frontend Setup
# ========================================

echo ""
echo "[7/7] Setting up React frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "[INFO] Installing frontend dependencies..."
    pnpm install --silent
fi

cd ..

# ========================================
# Verify Installation
# ========================================

echo ""
echo "========================================"
echo "  Verifying Installation"
echo "========================================"
echo ""

cd backend
source venv/bin/activate
python ../verify_installation.py
cd ..

# Create setup marker
date > .setup_complete

# ========================================
# Environment Setup
# ========================================

if [ ! -f ".env" ]; then
    echo ""
    echo "[INFO] Creating .env configuration file..."
    cp .env.example .env
    echo "[OK] Created .env - you can edit it to configure data directories"
fi

# ========================================
# Success
# ========================================

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Hardware: $DEVICE"

if [ "$DEVICE" == "CUDA" ]; then
    echo "Acceleration: torch.compile (GPU)"
else
    if [ -f "backend/recognition_models/ast_active_int8.onnx" ]; then
        echo "Acceleration: ONNX INT8 (3x CPU speedup)"
    else
        echo "Acceleration: PyTorch CPU (fallback)"
    fi
fi

echo ""
echo "Next step:"
echo "  - Run: ./start.sh"
echo ""
echo "The app will open at: http://localhost:5173"
echo ""

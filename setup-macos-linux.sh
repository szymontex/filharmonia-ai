#!/usr/bin/env bash
# Filharmonia AI - macOS/Linux Setup Script
# Run: chmod +x setup-macos-linux.sh && ./setup-macos-linux.sh

set -e  # Exit on error

echo "================================================"
echo " Filharmonia AI - macOS/Linux Setup"
echo "================================================"
echo ""

# ================================================
# Check Prerequisites
# ================================================

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found!"
    echo ""
    echo "Install Python 3.10+:"
    echo "  macOS: brew install python@3.10"
    echo "  Ubuntu: sudo apt install python3.10 python3.10-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "[OK] Python found: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js not found!"
    echo ""
    echo "Install Node.js 18+:"
    echo "  macOS: brew install node"
    echo "  Ubuntu: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs"
    exit 1
fi

echo "[OK] Node.js found: $(node --version)"

# Check/Install pnpm
if ! command -v pnpm &> /dev/null; then
    echo "[INFO] Installing pnpm..."
    npm install -g pnpm
fi

echo "[OK] pnpm found: $(pnpm --version)"
echo ""

# ================================================
# Backend Setup
# ================================================

echo "================================================"
echo " Setting up Backend (Python)"
echo "================================================"
echo ""

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install PyTorch CPU version
echo "[INFO] Installing PyTorch (CPU version)..."
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --quiet --disable-pip-version-check

# Install dependencies
echo "[INFO] Installing backend dependencies..."
pip install -r requirements.txt --quiet --disable-pip-version-check

# Export ONNX model
echo "[INFO] Exporting ONNX model (this takes ~30 seconds)..."
if python -m scripts.export_onnx 2>/dev/null; then
    echo "[OK] ONNX model exported - 3x+ CPU speedup enabled"
else
    echo "[WARN] ONNX export failed - will use PyTorch CPU fallback"
fi

cd ..

# ================================================
# Frontend Setup
# ================================================

echo ""
echo "================================================"
echo " Setting up Frontend (Node.js)"
echo "================================================"
echo ""

cd frontend

echo "[INFO] Installing frontend dependencies..."
pnpm install --silent

cd ..

# ================================================
# Create run script
# ================================================

echo ""
echo "[INFO] Creating run script..."

cat > run-macos-linux.sh << 'EOF'
#!/usr/bin/env bash
# Filharmonia AI - Start Application

echo "================================================"
echo " Starting Filharmonia AI"
echo "================================================"
echo ""
echo "Backend will start on: http://localhost:8000"
echo "Frontend will start on: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"
echo "================================================"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend in background
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend (foreground)
cd frontend
pnpm dev &
FRONTEND_PID=$!

# Wait for both processes
wait
EOF

chmod +x run-macos-linux.sh

echo "[OK] Setup complete!"
echo ""
echo "================================================"
echo " Setup Complete!"
echo "================================================"
echo ""
echo "To start the application:"
echo "  1. Run: ./run-macos-linux.sh"
echo "  2. Open browser to http://localhost:5173"
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Device detection at startup will show:"
echo "  - 'Device: cpu (CPU)' for CPU-only"
echo "  - 'Using ONNX INT8 backend (3.2x speedup)' if export succeeded"
echo ""

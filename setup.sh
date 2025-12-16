#!/bin/bash

echo "==================================="
echo "Filharmonia AI - Setup Script"
echo "==================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11+ from https://www.python.org/downloads/"
    exit 1
fi
python3 --version
echo ""

# Check pnpm
echo "[2/5] Checking pnpm..."
if ! command -v pnpm &> /dev/null; then
    echo "WARNING: pnpm is not installed"
    echo "Installing pnpm via npm..."
    npm install -g pnpm
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install pnpm. Please install Node.js first."
        exit 1
    fi
fi
echo ""

# Setup backend
echo "[3/5] Setting up Python backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing PyTorch with CUDA support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi

echo "Installing remaining dependencies..."
pip install -r requirements.txt --no-deps
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    exit 1
fi

cd ..
echo ""

# Setup frontend
echo "[4/5] Setting up React frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    pnpm install
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install frontend dependencies"
        exit 1
    fi
fi

cd ..
echo ""

# Setup environment
echo "[5/5] Setting up environment..."
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Please edit .env and configure FILHARMONIA_BASE_DIR"
    echo "           if you want to use a custom data directory."
fi
echo ""

echo "==================================="
echo "Setup completed successfully!"
echo "==================================="
echo ""
echo "Verifying installation..."
cd backend
source venv/bin/activate
python ../verify_installation.py
cd ..
echo ""
echo "Next steps:"
echo "1. Edit .env file to configure your data directory"
echo "2. Run ./start.sh to start both servers"
echo ""

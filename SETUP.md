# Filharmonia AI - Local Setup Guide

Quick setup scripts to run Filharmonia AI on your local machine with CPU-only support (no GPU required).

## Prerequisites

Install these first:

### Windows
- **Python 3.10+**: https://www.python.org/downloads/
  - ✅ Check "Add Python to PATH" during installation
- **Node.js 18+**: https://nodejs.org/

### macOS
```bash
brew install python@3.10 node
```

### Ubuntu/Linux
```bash
# Python
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

---

## Setup & Run

### Windows

**1. Run Setup (one time):**
```
Double-click: setup-windows.bat
```

This will:
- Create Python virtual environment
- Install PyTorch CPU version
- Install all dependencies
- Export ONNX model for 3x CPU speedup
- Install frontend dependencies

**2. Start Application:**
```
Double-click: run-windows.bat
```

### macOS / Linux

**1. Run Setup (one time):**
```bash
chmod +x setup-macos-linux.sh
./setup-macos-linux.sh
```

**2. Start Application:**
```bash
./run-macos-linux.sh
```

---

## After Starting

The setup creates two services:

- **Backend (API)**: http://localhost:8000
- **Frontend (UI)**: http://localhost:5173

**Open your browser to:** http://localhost:5173

---

## What Gets Installed

### Backend (Python)
- **PyTorch 2.5.1 CPU**: No CUDA required
- **ONNX Runtime**: 3x+ faster CPU inference
- **FastAPI**: REST API server
- **Polars**: Fast CSV parsing

### Frontend (Node.js)
- **React 19**: Modern UI framework
- **Vite**: Fast dev server
- **TypeScript**: Type safety

### Models
- **AST Active Model**: 5-class audio classifier (MUSIC, APPLAUSE, SPEECH, PUBLIC, TUNING)
- **ONNX INT8 Model**: Quantized version for CPU speedup

---

## Device Detection

At startup, backend logs will show:

**CPU-only (ONNX optimized):**
```
INFO: Device: cpu (CPU)
INFO: Using ONNX INT8 backend (3.2x speedup vs PyTorch CPU)
```

**CPU-only (PyTorch fallback):**
```
INFO: Device: cpu (CPU)
WARNING: ONNX model not found, using PyTorch CPU
```

**NVIDIA GPU (if detected):**
```
INFO: GPU detected: NVIDIA CUDA 12.1 — NVIDIA GeForce RTX 3090 (1 device(s))
INFO: Using PyTorch GPU backend with torch.compile
```

**AMD GPU (if detected):**
```
INFO: GPU detected: AMD ROCm 6.2 — AMD Radeon RX 7900 XTX (1 device(s))
INFO: Using PyTorch GPU backend with torch.compile
```

---

## Manual Setup (Advanced)

If you prefer manual control:

### Backend

```bash
cd backend

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install PyTorch CPU
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Export ONNX model (optional, for 3x speedup)
python -m scripts.export_onnx

# Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend

# Install pnpm (if not installed)
npm install -g pnpm

# Install dependencies
pnpm install

# Start frontend
pnpm dev
```

---

## GPU Support (Optional)

If you have a GPU, you can use GPU-accelerated inference:

### NVIDIA (CUDA 12.1)

Replace PyTorch installation in backend:
```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### AMD (ROCm 6.2)

Replace PyTorch installation in backend:
```bash
pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
```

See `docs/ROCM_SETUP.md` for detailed AMD GPU setup.

---

## Troubleshooting

### Backend won't start

**"Module not found" errors:**
```bash
# Make sure virtual environment is activated
cd backend
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate.bat  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**Port 8000 already in use:**
```bash
# Find process using port 8000
# Windows
netstat -ano | findstr :8000

# macOS/Linux
lsof -i :8000

# Kill the process or change port:
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Frontend won't start

**"EADDRINUSE: address already in use":**
```bash
# Port 5173 already in use, Vite will suggest another port
# Press 'y' to use the suggested port
```

**"pnpm: command not found":**
```bash
npm install -g pnpm
```

### ONNX export fails

Not critical - app will use PyTorch CPU fallback (slower but works).

To retry:
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
python -m scripts.export_onnx
```

### Slow inference on CPU

Make sure ONNX model exported successfully:
```bash
ls backend/recognition_models/ast_active_int8.onnx
```

If file exists but still slow, check startup logs for:
```
INFO: Using ONNX INT8 backend (3.2x speedup vs PyTorch CPU)
```

If you see "Using PyTorch CPU backend" instead, re-run export script.

---

## Directories Created

```
filharmonia-ai/
├── backend/
│   ├── venv/               # Python virtual environment
│   └── recognition_models/
│       └── ast_active_int8.onnx  # Exported ONNX model
├── frontend/
│   └── node_modules/       # Node.js packages
├── setup-windows.bat       # Windows setup
├── run-windows.bat         # Windows run
├── setup-macos-linux.sh    # macOS/Linux setup
└── run-macos-linux.sh      # macOS/Linux run
```

---

## Configuration

### Backend Port
Edit `backend/app/config.py`:
```python
# Default: 8000
PORT = 8000
```

### Frontend API Endpoint
Edit `frontend/vite.config.ts`:
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8000',  // Backend URL
    changeOrigin: true,
  },
}
```

### Audio Processing
Edit `backend/app/config.py`:
```python
# Sample rate (DO NOT CHANGE - model trained on 48kHz)
SAMPLE_RATE = 48000

# Frame duration (DO NOT CHANGE - model trained on 2.97s)
FRAME_DURATION = 2.97
```

---

## Features

### Core Workflow
1. **Upload MP3** → Browse local files or use calendar browser
2. **Automatic Analysis** → AST model classifies audio segments
3. **Review Results** → View waveform + predictions
4. **Edit Classifications** → Correct mistakes (undo/redo support)
5. **Export CSV** → Timestamped segment list for documentation

### Keyboard Shortcuts
- `Spacebar`: Play/pause audio
- `1-5`: Change segment classification
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Ctrl+S`: Save (autosave already enabled)
- `?`: Show keyboard help

### Performance Optimizations
- **Polars CSV parsing**: 5-30x faster than pandas
- **ONNX INT8 inference**: 3x+ faster than PyTorch CPU
- **Waveform caching**: <500ms repeat load (was 8s)
- **torch.compile on GPU**: ~2x speedup vs eager mode

### Device Support
- ✅ CPU-only (ONNX optimized)
- ✅ NVIDIA CUDA (torch.compile acceleration)
- ✅ AMD ROCm (with silent fallback detection)

---

## Next Steps

After setup works:

1. **Test Upload & Analysis**
   - Upload a short MP3 file
   - Wait for analysis to complete
   - Verify CSV displays with waveform

2. **Test Editing**
   - Click a segment to select
   - Press `1-5` to change classification
   - Press `Ctrl+Z` to undo
   - Verify autosave works

3. **Check Performance**
   - Backend startup logs show device type
   - ONNX should show 3x+ speedup on CPU
   - GPU should show torch.compile speedup

4. **Explore Features**
   - Calendar browser for recordings by date
   - Uncertainty review for low-confidence predictions
   - Training data export for model retraining

---

## Support

**Issues:** https://github.com/anthropics/claude-code/issues (for GSD workflow issues)

**Logs:**
- Backend: Terminal where `run-*.bat/.sh` was executed
- Frontend: Browser console (F12 → Console tab)

**Config Files:**
- Backend: `backend/app/config.py`
- Frontend: `frontend/vite.config.ts`

---

## License

See LICENSE file in repository root.

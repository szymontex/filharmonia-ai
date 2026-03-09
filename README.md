# 🎵 Filharmonia AI

**AI-powered concert audio analysis** using PyTorch Audio Spectrogram Transformer (AST) for automatic classification and segmentation of philharmonic concert recordings.

---

## 🚀 Quick Start (Choose One)

### Option 1: Docker (Easiest - No Setup Required)

```bash
docker-compose up
```

Open http://localhost:5173

Docker handles everything automatically.

---

### Option 2: Local Setup (Windows)

1. **Install Prerequisites:**
   - [Python 3.11](https://python.org) ✅ Check "Add Python to PATH"
   - [Node.js 18+](https://nodejs.org)

2. **Run Setup (one time):**
   ```
   Double-click: setup.bat
   ```
   Setup auto-detects GPU vs CPU, downloads the pre-trained model from HuggingFace, and creates the data directory structure.

3. **Start App:**
   ```
   Double-click: start.bat
   ```
   Opens http://localhost:5173 automatically.

---

### Option 3: Local Setup (macOS / Linux)

No manual prerequisite installation needed — `setup.sh` handles everything automatically.

```bash
chmod +x setup.sh
./setup.sh
./start.sh
```

The setup script will:
- Auto-detect your OS (Debian/Fedora/Arch/macOS) and install system packages (Python 3.11, Node.js, pnpm, ffmpeg, etc.)
- Install Python 3.11 via pyenv if not available from system packages
- Create a Python virtual environment and install all dependencies
- Auto-detect GPU and install the correct PyTorch version (CUDA 12.4 → 12.1 → CPU fallback)
- Download the pre-trained AST model (~1GB) from HuggingFace
- Create the data directory structure
- Install frontend dependencies and build

Opens http://localhost:5173 automatically.

---

## 📸 Screenshots

### Main Dashboard
![Main Dashboard](docs/images/1.png)

### File Browser & Sorting
![File Browser](docs/images/browse.png)
![Sort Recordings](docs/images/sort.png)

### Waveform Editor
![CSV Waveform Editor](docs/images/csv1.png)
![CSV Editor - Detailed View](docs/images/csv2.png)

### Model Management & Uncertainty Review
![Model Management](docs/images/model.png)
![Uncertainty Review](docs/images/uncertain.png)

---

## 📋 Features

### 🎼 Audio Classification
Automatically classifies audio into 5 categories:
- 🎵 **MUSIC** - orchestral music
- 👏 **APPLAUSE** - audience applause
- 🗣️ **SPEECH** - announcements, speeches
- 👥 **PUBLIC** - audience noise, intermission
- 🎻 **TUNING** - instrument tuning

### 🎨 Visual Waveform Editor
DAW-style interface for reviewing and correcting predictions with color-coded segments.

### 🤖 Self-Improving ML Loop
Export corrected segments → retrain model → improved accuracy over time.

### ⚡ Performance Optimization
- **GPU Acceleration**: torch.compile for NVIDIA CUDA (~2x speedup)
- **CPU Optimization**: ONNX INT8 quantization (3x+ speedup vs PyTorch)
- **AMD GPU Support**: ROCm 6.4 with silent fallback detection
- **Auto-Detection**: No manual configuration needed

### 🎹 User Experience
- **Keyboard Shortcuts**: Spacebar (play/pause), 1-5 (classifications), Ctrl+Z/Y (undo/redo), ? (help)
- **Undo/Redo**: Full history for all edits
- **Autosave**: Automatic saving of edits
- **Calendar Browser**: Navigate recordings by date
- **Uncertainty Review**: Filter and review low-confidence predictions

---

## 🛠️ What Gets Installed

### Backend (Python 3.11)
- PyTorch (auto-detects CUDA 12.4/12.1 vs CPU)
- ONNX Runtime for CPU speedup
- FastAPI REST API
- Polars for fast CSV parsing (5-30x faster than pandas)

### Frontend (Node.js)
- React 19 with TypeScript
- Vite dev server
- Modern component architecture

### Models
- **AST Active Model**: Pre-trained 5-class audio classifier (auto-downloaded from [HuggingFace](https://huggingface.co/szymontex/filharmonia-ast))
- **Base Architecture**: `MIT/ast-finetuned-audioset-10-10-0.4593`
- **ONNX INT8 Model**: Quantized version for CPU (3x+ faster)

**Total Size:** ~2GB (including model download)

> **Note:** Python 3.11 is required. Python 3.13+ has breaking stdlib removals (`aifc`, `audioop`) that affect audio processing libraries.

---

## 📊 Performance

### Device Detection
At startup, backend logs show what was detected:

**CPU-only (ONNX optimized):**
```
INFO: Device: cpu (CPU)
INFO: Using ONNX INT8 backend (3.2x speedup vs PyTorch CPU)
```

**NVIDIA GPU:**
```
INFO: GPU detected: NVIDIA CUDA 12.4 — GeForce RTX 3090
INFO: Using PyTorch GPU backend with torch.compile
```

**AMD GPU:**
```
INFO: GPU detected: AMD ROCm 6.2 — Radeon RX 7900 XTX
INFO: Using PyTorch GPU backend with torch.compile
```

> **Note:** `torch.compile` requires NVIDIA GPUs with CUDA capability >= 7.0 (Volta+). Older GPUs (e.g. GTX 1060) will automatically fall back to eager mode.

### Speed Improvements
- **Polars CSV parsing**: 5-30x faster than pandas
- **ONNX INT8 CPU**: 3x+ faster than PyTorch
- **torch.compile GPU**: ~2x speedup vs eager mode
- **Waveform caching**: <500ms repeat load (was 8s)

---

## 🔧 Troubleshooting

### Setup Issues

**"Python not found" (Windows):**
- Install Python 3.11 with "Add Python to PATH" checked
- Restart terminal after installation

**"Python not found" (Linux/macOS):**
- `setup.sh` installs Python 3.11 automatically via system packages or pyenv
- If it fails, install pyenv manually: `curl https://pyenv.run | bash` then re-run setup

**"Node.js not found":**
- On Linux/macOS, `setup.sh` installs it automatically
- On Windows, install from https://nodejs.org
- Restart terminal

**ONNX export fails:**
- Not critical - app uses PyTorch CPU fallback
- Slightly slower but works fine
- Re-run setup to retry

**`No module named 'aifc'` or `audioop`:**
- You're running Python 3.13+ which removed these modules
- Switch to Python 3.11 (recommended version for this project)

### Runtime Issues

**Backend won't start:**
```bash
# Windows
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --reload

# macOS/Linux
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload
```

**Frontend won't start:**
```bash
cd frontend
pnpm install
pnpm dev
```

**Port already in use:**
- Windows: `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F`
- macOS/Linux: `lsof -ti:8000 | xargs kill -9`

---

## 📁 Project Structure

```
filharmonia-ai/
├── backend/           # FastAPI + PyTorch backend
│   ├── app/          # API routes, services, models
│   ├── scripts/      # ONNX export, utilities
│   └── venv/         # Python virtual environment
├── frontend/         # React 19 frontend
│   ├── src/          # Components, hooks, pages
│   └── node_modules/ # Node.js packages
├── data/             # Working data directory (auto-created)
│   ├── SORTED/       # Analyzed recordings
│   └── NAGRANIA_KONCERTOW/  # Raw concert recordings
├── docker/           # Docker configuration
├── setup.bat         # Windows setup script
├── setup.sh          # macOS/Linux setup script (auto-installs everything)
├── start.bat         # Windows start script
├── start.sh          # macOS/Linux start script (proper process cleanup on Ctrl+C)
├── stop.sh           # macOS/Linux stop script
└── docker-compose.yml # Docker setup
```

---

## 🐳 Docker Usage

### Development (with hot reload):
```bash
docker-compose up
```

### Production:
```bash
docker build -t filharmonia-ai .
docker run -p 80:80 -v /path/to/audio:/data filharmonia-ai
```

Open http://localhost

---

## 🎯 Usage Workflow

1. **Upload MP3** → Browse local files or use calendar browser
2. **Automatic Analysis** → AST model classifies audio segments
3. **Review Results** → View waveform + predictions
4. **Edit Classifications** → Correct mistakes (undo/redo support)
5. **Export CSV** → Timestamped segment list

### Keyboard Shortcuts
- `Spacebar`: Play/pause audio
- `1-5`: Change segment classification
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Ctrl+S`: Save (autosave already enabled)
- `?`: Show keyboard help

---

## 🌐 API

**Backend:** http://localhost:8000
**Frontend:** http://localhost:5173
**API Docs:** http://localhost:8000/docs

---

## 📚 Documentation

- **docs/ROCM_SETUP.md** - AMD GPU setup guide
- **.planning/** - Development milestones and planning docs

---

## 🏗️ Built With

- **PyTorch** - Deep learning framework
- **Audio Spectrogram Transformer (AST)** - Audio classification model
- **FastAPI** - High-performance Python API
- **React 19** - Modern UI framework
- **Polars** - Fast dataframe library
- **ONNX Runtime** - Cross-platform inference
- **Vite** - Fast frontend tooling

---

## ✨ Recent Updates

### v0.10 - Cross-Platform & Reliability (2026-03-10)
- ✅ Python 3.11 standardization (3.13+ breaks audio libs)
- ✅ `setup.sh` auto-installs all prerequisites (OS-aware)
- ✅ Pre-trained model auto-download from HuggingFace
- ✅ Multi-index PyTorch install fallback (cu124 → cu121 → CPU)
- ✅ `torch.compile` CUDA capability check (graceful fallback on older GPUs)
- ✅ Cross-platform path display fixes (Linux + Windows)
- ✅ Proper process cleanup on Ctrl+C (`setsid` + process group kills)
- ✅ Data directory structure auto-creation

### v0.9 - Polish & Stability (2026-01-29)
- ✅ Unified device detection (NVIDIA/AMD/CPU)
- ✅ ONNX INT8 CPU optimization (3x speedup)
- ✅ torch.compile GPU acceleration
- ✅ ROCm 6.4 support for AMD GPUs
- ✅ React 19 upgrade
- ✅ Confidence threshold auto-tuning
- ✅ Cross-platform paths
- ✅ Component refactoring (30% code reduction)
- ✅ Performance improvements (Polars migration)

---

## 📝 License

See LICENSE file in repository root.

---

## 🆘 Support

**Setup Issues:**
1. Check logs: `backend.log` and `frontend.log`
2. Re-run setup script
3. Check troubleshooting section above

**Feature Requests:**
- Create an issue on GitHub

**Need Help:**
- Check documentation in `docs/` folder
- Review API docs at http://localhost:8000/docs

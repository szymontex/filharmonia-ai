# 🎵 Filharmonia AI

AI-powered concert audio analysis system using **PyTorch Audio Spectrogram Transformer (AST)** for automatic classification and segmentation of philharmonic concert recordings.

## 📋 Features

- **🎼 Audio Classification**: Automatically classifies audio into 5 categories:
  - 🎵 MUSIC - orchestral music
  - 👏 APPLAUSE - audience applause
  - 🗣️ SPEECH - announcements, speeches
  - 👥 PUBLIC - audience noise, intermission
  - 🎻 TUNING - instrument tuning

- **🎨 Visual Waveform Editor**: DAW-style interface for reviewing and correcting predictions
- **🤖 Self-Improving ML Loop**: Export corrected segments → retrain model → improved accuracy
- **📊 Model Management**: Train, compare, and switch between models with measured accuracy
- **📈 Uncertainty Review**: Filter low-confidence predictions for manual review
- **⚡ GPU Accelerated**: CUDA support for fast training and inference

## 🏗️ Architecture

```
filharmonia-ai/
├── backend/              # FastAPI + PyTorch backend
│   ├── app/
│   │   ├── api/v1/      # REST API endpoints
│   │   ├── services/    # Core business logic
│   │   │   ├── ast_training.py    # Model training service
│   │   │   ├── ast_inference.py   # Model inference service
│   │   │   └── analyze.py         # Audio analysis pipeline
│   │   └── config.py    # Settings and paths
│   ├── pytorch_dataset.py         # Custom PyTorch dataset
│   └── requirements.txt
│
├── frontend/            # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── pages/       # Page views
│   │   └── api/         # API client
│   └── package.json
│
└── .claude/             # Project documentation
    ├── PROJECT_OVERVIEW.md
    ├── ARCHITECTURE.md
    └── QUICK_START.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (with pnpm)
- **NVIDIA GPU** (optional but recommended for training)
- **CUDA 12.x** (if using GPU)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Run dev server
pnpm dev
```

Frontend will be available at: `http://localhost:5173`

## 📊 Model Training

The system uses **Audio Spectrogram Transformer (AST)** from MIT:
- Pre-trained on AudioSet-10M
- Fine-tuned on concert recordings
- ~86M parameters
- Training time: ~4h on RTX 3080 Ti

**Training new model:**
1. Prepare training data in `TRAINING DATA/DATA/` folder (5 class subfolders)
2. Open web UI → "Training" tab
3. Click "Start Training"
4. Monitor progress in real-time
5. Click "📊 Measure" to evaluate accuracy
6. Click "Activate" to deploy new model

## 🎯 Performance

**Current best model (ast_20251009_222204.pth):**
- Test Accuracy: **97.75%**
- Per-class accuracy:
  - APPLAUSE: 100%
  - MUSIC: 100%
  - PUBLIC: 96.2%
  - SPEECH: 100%
  - TUNING: 85.7%

## 🔧 Configuration

Edit `backend/app/config.py` to configure:
- Training data paths
- Model save location
- Sample rate & duration
- GPU/CPU device selection

## 📝 Workflow

1. **Sort Recordings**: Organize MP3 files by date using ID3 tags
2. **Analyze**: Process concerts through AST model (~5 min per 1h concert)
3. **Review**: Visual waveform editor for corrections
4. **Export**: Generate tracklists for clients
5. **Train**: Export corrected segments → retrain model → improved accuracy

## 🛠️ Tech Stack

**Backend:**
- FastAPI (REST API)
- PyTorch + torchaudio (ML)
- HuggingFace Transformers (AST model)
- scikit-learn (dataset splitting)

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- TanStack Query (data fetching)
- Recharts (visualizations)
- Tailwind CSS (styling)

## 📚 Documentation

Detailed documentation available in `.claude/` folder:
- `PROJECT_OVERVIEW.md` - Project goals and architecture
- `ARCHITECTURE.md` - Technical architecture details
- `QUICK_START.md` - Development setup guide
- `CLAUDE.md` - Claude Code assistant guide

## 🤝 Contributing

This is a private project for Filharmonia workflow automation. For questions or collaboration, contact the project maintainer.

## 📄 License

Private project - all rights reserved.

## 🎉 Achievements

- ✅ MVP completed (Oct 2025)
- ✅ Migrated from Keras CNN to PyTorch AST
- ✅ Achieved 97.75% test accuracy
- ✅ Reduced monthly processing time from 4-6h to ~30 min
- ✅ Implemented self-improving ML loop

---

**Last Updated:** October 2025
**Status:** 🚀 Production Ready (MVP)

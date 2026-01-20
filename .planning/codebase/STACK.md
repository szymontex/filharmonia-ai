# Technology Stack

**Analysis Date:** 2026-01-20

## Languages

**Primary:**
- TypeScript 5.4.x - Frontend React application (`frontend/src/`)
- Python 3.12 - Backend FastAPI application (`backend/app/`)

**Secondary:**
- JavaScript - Build configuration files (`vite.config.ts`, `tailwind.config.js`)
- Bash/Batch - Startup and setup scripts (`setup.sh`, `setup.bat`, `start.sh`, `start.bat`)

## Runtime

**Backend:**
- Python 3.11+ (tested with 3.12)
- Virtual environment: `backend/venv/`

**Frontend:**
- Node.js 18+
- Package manager: pnpm
- Lockfile: `frontend/pnpm-lock.yaml` (present)

## Frameworks

**Backend Core:**
- FastAPI 0.115.0 - REST API framework (`backend/app/main.py`)
- Uvicorn 0.30.0 - ASGI server
- Pydantic 2.11.9 - Data validation and serialization

**ML/AI:**
- PyTorch 2.5.1+cu121 - Primary ML framework (GPU-accelerated)
- torchaudio 2.5.1+cu121 - Audio processing transforms
- transformers 4.58.0 - Hugging Face AST model loading
- scikit-learn 1.7.2 - Dataset splitting, utilities

**Audio Processing:**
- librosa 0.10.2 - Audio loading and feature extraction (inference)
- soundfile 0.12.1 - Audio file I/O
- eyed3 0.9.7 - MP3 ID3 tag reading

**Frontend Core:**
- React 18.3.1 - UI framework (`frontend/src/`)
- Vite 5.1.5 - Build tool and dev server
- TypeScript 5.4.2 - Type safety

**Frontend State/Data:**
- @tanstack/react-query 5.28.0 - Server state management
- zustand 4.5.2 - Client state management
- axios 1.6.7 - HTTP client

**Frontend UI:**
- Tailwind CSS 3.4.1 - Styling framework
- lucide-react 0.344.0 - Icon library
- recharts 2.12.0 - Charts and visualizations
- howler 2.2.4 - Audio playback
- waveform-data 4.5.2 - Waveform visualization

**Testing:**
- pytest 8.0.0 - Python test runner
- pytest-asyncio 0.23.0 - Async test support

## Key Dependencies

**Critical ML:**
- `transformers` - Loads MIT/ast-finetuned-audioset model from Hugging Face
- `torch` - Model training and inference (CUDA 12.1 support)
- `torchaudio` - Audio transforms (MelSpectrogram, MFCC, resampling)

**Data Processing:**
- `numpy` 1.26.4 - Numerical arrays
- `pandas` 2.2.0 - Data manipulation (not heavily used)
- `scikit-learn` - train_test_split, dataset utilities

**Audio Stack:**
- `librosa` - MP3 loading, audio analysis (uses 48kHz sample rate)
- `soundfile` - WAV file reading (training data)
- `eyed3` - ID3 tag extraction for date/time metadata

**API/Server:**
- `python-dotenv` 1.1.1 - Environment variable loading
- `python-multipart` 0.0.9 - Multipart form handling

## Configuration

**Environment Variables:**
- `FILHARMONIA_BASE_DIR` - Root directory for all data (optional, defaults to `FILHARMONIA_DATA/`)
- `SORTED_FOLDER_NAME` - Sorted recordings folder name (default: `SORTED`)
- `NAGRANIA_FOLDER_NAME` - Raw recordings folder name (default: `NAGRANIA_KONCERTOW`)
- `TRAINING_DATA_FOLDER_NAME` - Training samples folder (default: `TRAINING_DATA`)
- `MODELS_FOLDER_NAME` - Model storage folder (default: `RECOGNITION_MODELS`)
- `ML_EXPERIMENTS_FOLDER_NAME` - Experiments folder (default: `ML_EXPERIMENTS`)
- `DATABASE_URL` - SQLite database path (optional)
- `CORS_ORIGINS` - Allowed origins for CORS (default: `http://localhost:5173,http://localhost:3000`)
- `VITE_ALLOWED_HOSTS` - Additional allowed hosts for Vite dev server

**Configuration Files:**
- `backend/app/config.py` - Centralized settings singleton
- `frontend/vite.config.ts` - Vite build config with proxy setup
- `frontend/tsconfig.json` - TypeScript strict mode, path aliases (`@/*`)
- `frontend/tailwind.config.js` - Tailwind content paths
- `.env` - Environment overrides (root level, loaded by backend)
- `.env.example` - Template with documentation

**Build Configuration:**
- TypeScript target: ES2020
- Module: ESNext with bundler resolution
- Strict mode enabled with unused variable checks
- Path alias: `@/*` maps to `./src/*`

## Platform Requirements

**Development:**
- Python 3.11+ with pip
- Node.js 18+ with pnpm
- NVIDIA GPU with CUDA 12.x (optional, falls back to CPU)
- 16GB+ RAM recommended for training

**Production:**
- Same as development
- Backend: Uvicorn on port 8000
- Frontend: Vite dev server on port 5173 (or static build)
- GPU strongly recommended for training (~4h on RTX 3080 Ti)

**GPU Support:**
- PyTorch built with CUDA 12.1 (`torch==2.5.1+cu121`)
- NVIDIA drivers required for GPU inference/training
- CPU fallback available (significantly slower)

## Audio Processing Settings

**Fixed Constants (DO NOT CHANGE):**
- Sample rate: 48000 Hz
- Frame duration: 2.97 seconds (training consistency)
- Classification labels: APPLAUSE, MUSIC, PUBLIC, SPEECH, TUNING (alphabetical order)

**Mel-Spectrogram Parameters:**
- n_fft: 2048
- hop_length: 512
- n_mels: 128

---

*Stack analysis: 2026-01-20*

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Filharmonia AI** is a concert audio analysis system using deep learning (PyTorch AST) to automatically categorize and manage philharmonic concert recordings.

### What It Does
- Automatically classifies audio into 5 categories: MUSIC, APPLAUSE, SPEECH, PUBLIC, TUNING
- Provides a web-based waveform editor for visual correction of AI predictions
- Exports corrected segments as training data for model improvement
- Implements active learning workflow via uncertainty review

### Tech Stack
- **Backend:** FastAPI + PyTorch + librosa + torchaudio
- **Frontend:** React 18 + TypeScript + Vite + Tailwind CSS
- **ML Model:** Audio Spectrogram Transformer (AST) - Vision Transformer architecture
- **Audio Processing:** 48kHz sample rate, 2.97s segments, mel-spectrogram → Fbank features

## Development Commands

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Backend Run
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Frontend Setup
```bash
cd frontend
pnpm install  # or npm install
```

### Frontend Run
```bash
cd frontend
pnpm dev  # or npm run dev
# Opens: http://localhost:5173
```

### Model Training
```bash
cd backend
venv\Scripts\activate

# Prepare dataset (creates train/val/test splits)
python prepare_full_balanced_dataset.py

# Train AST model
python train_ast_full.py

# Evaluate model
python evaluate_model.py

# Compare models
python compare_all_models.py
```

### Testing Individual Models
```bash
# Test specific architecture before full training
python test_ast.py         # Audio Spectrogram Transformer
python test_efficientnet.py
python test_resnet.py
python test_panns.py
```

## Architecture Overview

### Hybrid Code-Data Structure
**CRITICAL:** Code is stored locally (`C:\IT\code\filharmonia-ai\`) but data lives on network drive (`Y:\!_FILHARMONIA\`). All file paths in `backend/app/config.py` point to the network drive.

### Data Flow
```
!NAGRANIA KONCERTÓW/         →  SORTED/YYYY/MM/DD/  →  ANALYSIS_RESULTS/predictions_*.csv
(unsorted MP3 uploads)          (organized by date)      (AI predictions + edits)
                                                     ↓
                                            TRAINING DATA/DATA/{CLASS}/
                                            (exported WAV segments for retraining)
```

### Model Architecture (PyTorch AST)
The system uses **Audio Spectrogram Transformer** (MIT/PSLA pretrained):
- **Preprocessing:** Fbank features (filter bank), NOT Mel-spectrogram
- **Input:** 1024 time frames × 128 frequency bins
- **Batch size:** 32 segments processed simultaneously
- **Accuracy:** ~97% (vs previous Keras CNN ~85%)

**CRITICAL BUG FIX (v1.5.0):** Train-inference preprocessing must match EXACTLY. Use `torchaudio.transforms` for both training and inference, NOT librosa for inference.

### Background Processing
Analysis jobs run in **daemon threads** (not BackgroundTasks) to allow immediate backend restart:
- `app/api/v1/analyze.py` - Single file analysis
- `app/api/v1/batch.py` - Batch analysis with progress tracking
- Jobs survive in memory but NOT across backend restarts

### CSV Format Evolution
**Old format (pre-v1.5.0):**
```csv
segment_time,predicted_class
00:00:00,MUSIC
```

**Current format (v1.5.0+):**
```csv
segment_time,predicted_class,confidence
00:00:00,MUSIC,0.9823
```

Parser is **backward compatible** - old CSVs work, but uncertainty review requires confidence scores.

## Key Implementation Details

### Virtual Chunking (Training Performance)
`backend/pytorch_dataset.py` uses **virtual chunking** to utilize 100% of training data:
- Long audio files (e.g., 8-minute MUSIC file) are split into multiple 2.97s chunks
- Creates virtual samples WITHOUT copying files (uses hardlinks)
- Training sees ~10x more samples from same data
- Validation/test: chunking disabled for deterministic evaluation

### Class Imbalance Handling
Training data is heavily imbalanced (MUSIC: 874 min vs TUNING: 25 min = 35x difference):
- `balance_strength` parameter (0.0-1.0) controls sampling strategy
- 0.0 = natural distribution, 1.0 = fully balanced
- Recommended: 0.75 (good trade-off found via overnight experiments)
- See `EXPERIMENT_README.md` for balance strength comparison

### Waveform Rendering
Frontend uses **Canvas 2D API** (not wavesurfer.js) for custom rendering:
- Downsamples audio to ~2000 points for performance
- Renders colored regions per class (defined in `frontend/src/constants/colors.ts`)
- Real-time playhead sync with HTML5 Audio API
- Amplitude zoom (1x-10x) for quiet recordings

### Export Tracking
Two separate tracking systems:
1. **Manual exports** (`exported_segments.csv`) - from CSV editor
2. **Uncertainty review** (`reviewed_segments.csv`) - from active learning quiz

Both prevent duplicate exports and track which segments have been manually verified.

### Model Versioning System
- Active model: `RECOGNITION_MODELS/ast_active.pth`
- Source models: `ast_YYYYMMDD_HHMMSS.pth`
- Metadata: `models_metadata.json` tracks accuracy, epochs, notes
- CSV predictions tagged with `model_version` to link results to specific model
- Activation: copies source model to `ast_active.pth` + updates metadata

## Important Configuration

### Audio Processing Constants
**DO NOT CHANGE** these values without retraining the model:
- `SAMPLE_RATE = 48000` (for loading MP3s)
- `FRAME_DURATION_SEC = 2.97` (segment length)
- AST preprocessing uses **16kHz** internally (auto-resampled)

### Paths Configuration
All paths defined in `backend/app/config.py`:
- `FILHARMONIA_BASE = Y:\!_FILHARMONIA`
- `SORTED_FOLDER` - organized recordings
- `TRAINING_DATA_FOLDER` - exported WAV segments
- `RECOGNITION_MODELS_FOLDER` - trained models

### CORS Origins
Backend allows only: `http://localhost:5173` and `http://localhost:3000`

## Common Gotchas

### 1. GPU Detection
If TensorFlow/PyTorch doesn't detect GPU:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall with CUDA support
pip install tensorflow[and-cuda]==2.15.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Segment Count Mismatch
Old bug: Analysis didn't process entire file (used `//` instead of ceil division).
**Fixed in v1.1.0:** Use `(len(y) + frame_length - 1) // frame_length` to ensure last segment is included.

### 3. CSV Parser Edge Cases
- **Commas in titles:** Parser uses `quoting=1` and `escape_csv_field()` function
- **Duplicate timestamps:** Fixed by changing gap filler from `<=` to `<`
- **Column name detection:** Parser auto-detects old vs new format

### 4. Autosave False Positives
`check-autosave` endpoint compares file **contents**, not just timestamps. If identical, autosave is auto-deleted.

### 5. Preprocessing Train-Inference Mismatch
**CRITICAL:** AST model requires **Fbank features** (via `ASTFeatureExtractor`), not raw Mel-spectrograms. Using librosa for inference while training used torchaudio caused 50% accuracy drop.

## Workflow Entry Points

### Analysis Pipeline
1. **Sort:** `app/api/v1/sort.py` - Reads ID3 tags, organizes by date
2. **Analyze:** `app/api/v1/batch.py` - Batch processing with progress tracking
3. **Edit:** Frontend `pages/CsvViewer.tsx` - Waveform editor with autosave
4. **Export:** `app/api/v1/export.py` - Segments to training data (stereo WAV, 44.1kHz)

### Training Pipeline
1. **Prepare dataset:** `prepare_full_balanced_dataset.py` - 80/10/10 split with virtual chunking
2. **Train:** `train_ast_full.py` - AST fine-tuning with class weighting
3. **Evaluate:** `compare_models_fair.py` - Per-class accuracy breakdown
4. **Deploy:** Model versioning system in `app/services/ast_training.py`

### Active Learning Loop
1. **Find uncertain:** `app/api/v1/uncertainty.py` - confidence < 0.7 threshold
2. **User review:** Frontend `pages/UncertaintyReview.tsx` - Visual quiz with waveform
3. **Export verified:** User-selected range → training data
4. **Track reviewed:** `reviewed_segments.csv` prevents re-showing

## Documentation Structure

For detailed information, see `Y:\!_FILHARMONIA\.claude\`:
- `PROJECT_OVERVIEW.md` - Goals, metrics, success criteria
- `ARCHITECTURE.md` - Planned architecture and API design
- `CURRENT_STATE.md` - Actual implementation status (v1.5.0)
- `AST_MIGRATION_SUMMARY.md` - Keras CNN → PyTorch AST migration
- `PYTORCH_RESEARCH_RESULTS.md` - Model architecture comparison
- `MODEL_VERSIONING_IMPLEMENTATION.md` - Versioning system details
- `QUICK_START.md` - Step-by-step setup guide
- `CHANGELOG.md` - Version history

## Testing Notes

When testing analysis:
1. Use short files first (~10 min) to verify segmentation
2. Check CSV has correct number of rows: `duration / 2.97`
3. Verify confidence scores are present (for uncertainty review)
4. GPU utilization should be 60-80% during batch inference

When testing exports:
1. Verify WAV files are **stereo** at **44.1kHz** (matches existing training data)
2. Check tracking CSV for duplicate prevention
3. Confirm exported segments show visual indicators (📦 badge + purple background)

## Current Status (2025-10-09)

✅ **MVP Complete** - All core features implemented
- Web UI (React + FastAPI)
- Analysis with PyTorch AST (97% accuracy)
- Waveform editor with visual feedback
- Background batch processing
- Export to training data
- Uncertainty review (active learning)
- Model versioning system

⚠️ **Known Limitations:**
- No database (file-based CSV storage)
- No authentication (single-user local deployment)
- Job tracking in-memory (lost on restart)
- No Docker deployment yet

🚀 **Next Priorities:**
- Database migration (SQLite)
- Model retraining with verified uncertainty data
- Batch export improvements
- Drag-to-adjust segment boundaries

# Development Guide

## Project Structure

```
filharmonia-ai/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/v1/      # REST API endpoints
│   │   ├── services/    # Business logic
│   │   └── config.py    # Configuration
│   └── pytorch_dataset.py
├── frontend/            # React frontend
│   └── src/
│       ├── components/
│       ├── pages/
│       └── api/
└── docs/               # Documentation
```

## Development Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API documentation: http://localhost:8000/docs

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Frontend: http://localhost:5173

## Configuration

All configuration in `backend/app/config.py`:

```python
FILHARMONIA_BASE: Path  # Base data directory
SAMPLE_RATE: int = 48000
FRAME_DURATION_SEC: float = 2.97
LABELS: List[str] = ["APPLAUSE", "MUSIC", "PUBLIC", "SPEECH", "TUNING"]
```

### Environment Variables

Create `.env` file:

```bash
FILHARMONIA_BASE_DIR=/path/to/data
SORTED_FOLDER_NAME=SORTED
NAGRANIA_FOLDER_NAME=NAGRANIA_KONCERTOW
TRAINING_DATA_FOLDER_NAME=TRAINING_DATA
MODELS_FOLDER_NAME=RECOGNITION_MODELS
ML_EXPERIMENTS_FOLDER_NAME=ML_EXPERIMENTS
```

## Data Flow

```
Raw MP3s → Sort by date → Analysis → CSV predictions → Manual review → Export to training data
```

### Directory Structure

```
FILHARMONIA_BASE_DIR/
├── SORTED/YYYY/MM/DD/          # Organized recordings
├── NAGRANIA_KONCERTOW/         # Unsorted uploads
├── TRAINING_DATA/DATA/         # Training samples
│   ├── APPLAUSE/
│   ├── MUSIC/
│   ├── PUBLIC/
│   ├── SPEECH/
│   └── TUNING/
├── RECOGNITION_MODELS/         # Trained models
│   ├── ast_active.pth          # Active model
│   ├── ast_YYYYMMDD_HHMMSS.pth
│   └── models_metadata.json
└── ML_EXPERIMENTS/datasets/    # Dataset splits
```

## Model Training

### Using Web UI

1. Open http://localhost:5173
2. Navigate to "Training" tab
3. Check training data statistics
4. Click "Start Training"
5. Monitor progress in real-time
6. Measure accuracy when complete
7. Activate best model

### Training Pipeline (Internal)

The system uses **virtual chunking** to maximize training data utilization:

- Long audio files split into 2.97s segments
- Uses hardlinks (zero disk space overhead)
- ~10x more training samples from same data
- Validation/test: no chunking for deterministic results

### Class Imbalance

Training data can be heavily imbalanced. The system uses:

- `balance_strength` parameter (0.0-1.0)
- 0.0 = natural distribution
- 1.0 = fully balanced sampling
- Recommended: 0.75

## Audio Processing

### Constants (DO NOT CHANGE without retraining)

```python
SAMPLE_RATE = 48000        # MP3 loading
FRAME_DURATION_SEC = 2.97  # Segment length
```

AST model internally uses 16kHz (auto-resampled).

### Processing Pipeline

1. Load MP3 at 48kHz
2. Segment into 2.97s chunks
3. Resample to 16kHz
4. Extract Fbank features (NOT mel-spectrogram)
5. Feed to AST model (1024 time frames × 128 freq bins)

**Critical**: Always use `torchaudio.transforms` for preprocessing to match training.

## CSV Format

### Current Format (v1.5.0+)

```csv
segment_time,predicted_class,confidence
00:00:00,MUSIC,0.9823
00:00:02.97,APPLAUSE,0.7521
```

### Legacy Format (backward compatible)

```csv
segment_time,predicted_class
00:00:00,MUSIC
```

Parser auto-detects format. Confidence scores required for uncertainty review.

## Background Processing

Analysis jobs run in daemon threads (survive immediate backend restart):

- Single file: `app/api/v1/analyze.py`
- Batch analysis: `app/api/v1/batch.py`
- Job tracking: in-memory (lost on full restart)

## Export System

Two tracking mechanisms:

1. **Manual exports**: `exported_segments.csv`
   - From CSV editor
   - Prevents duplicate exports

2. **Uncertainty review**: `reviewed_segments.csv`
   - From active learning workflow
   - Tracks manually verified segments

Export format: Stereo WAV, 44.1kHz

## Model Versioning

- Active model: `ast_active.pth` (symlink/copy)
- Timestamped models: `ast_YYYYMMDD_HHMMSS.pth`
- Metadata: `models_metadata.json` tracks accuracy, epochs, notes
- CSV predictions tagged with `model_version`

Activation: copies source model to `ast_active.pth` + updates metadata.

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Segment Count Mismatch

Ensure analysis processes entire file:

```python
num_segments = (len(audio) + frame_length - 1) // frame_length
```

### Low Model Accuracy

Check preprocessing consistency:
- Training uses `torchaudio.transforms.MelSpectrogram`
- Inference MUST use same preprocessing
- Never mix librosa/torchaudio between train/inference

## Testing

### Analysis Testing

1. Use short files (~10 min) first
2. Verify CSV row count: `duration / 2.97`
3. Check confidence scores present
4. GPU utilization: 60-80% during batch inference

### Export Testing

1. Verify WAV format: stereo, 44.1kHz
2. Check tracking CSVs updated
3. Confirm visual indicators (📦 badge, purple background)

## API Endpoints

See [API.md](API.md) for full endpoint documentation.

## Frontend Components

### Waveform Editor

- Custom Canvas 2D renderer (not wavesurfer.js)
- Downsamples to ~2000 points for performance
- Color-coded segments per class
- Real-time playhead sync with HTML5 Audio
- Amplitude zoom: 1x-10x

### Active Learning

- Filters predictions with confidence < 0.7
- Visual quiz interface
- Export verified segments to training data
- Prevents re-showing reviewed segments

## Known Limitations

- File-based storage (no database yet)
- Single-user (no authentication)
- In-memory job tracking
- No Docker deployment

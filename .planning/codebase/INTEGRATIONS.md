# External Integrations

**Analysis Date:** 2026-01-20

## APIs & External Services

**Hugging Face Hub:**
- Service: Model download for pre-trained AST weights
- SDK/Client: `transformers` library (ASTForAudioClassification.from_pretrained)
- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Auth: None required (public model)
- Files: `backend/app/services/ast_inference.py`, `backend/app/services/ast_training.py`

**No Other External APIs:**
- Application is fully self-contained
- No cloud storage integrations
- No external authentication providers
- No third-party analytics or monitoring

## Data Storage

**Databases:**
- SQLite (local file-based)
  - Connection: `FILHARMONIA_BASE/.claude/filharmonia.db`
  - ORM: SQLAlchemy 2.0.25 (configured in `backend/app/config.py`)
  - Usage: Minimal/optional - most data stored as files

**File Storage:**
- Local filesystem only
- Directory structure managed by application:
  ```
  FILHARMONIA_BASE_DIR/
  ├── SORTED/                    # Organized concert recordings (YYYY/MM/DD/)
  │   └── ANALYSIS_RESULTS/      # CSV prediction files
  ├── NAGRANIA_KONCERTOW/        # Raw unsorted recordings (input)
  ├── TRAINING_DATA/DATA/        # Training samples (5 class subfolders)
  ├── RECOGNITION_MODELS/        # Trained .pth model files
  │   └── models_metadata.json   # Model registry
  └── ML_EXPERIMENTS/datasets/   # Pre-split train/val/test datasets
  ```

**Caching:**
- None (no Redis, Memcached, etc.)
- Models loaded as singletons in memory (`_service` pattern)

## Authentication & Identity

**Auth Provider:**
- None - application runs without authentication
- Designed for single-user local deployment
- CORS configured for localhost origins by default

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Bugsnag, etc.)
- Errors logged to console/stdout

**Logs:**
- Console output via print statements
- Backend logs to stdout (captured in `backend.log` when running via scripts)
- Frontend logs to browser console
- Training progress logged with timestamps and batch details

**Metrics:**
- None external
- Training metrics stored in `models_metadata.json`:
  - test_accuracy, val_accuracy, loss, val_loss
  - per_class_acc, measured_train_acc
  - epochs_trained, training_samples

## CI/CD & Deployment

**Hosting:**
- Local deployment only
- No containerization (Docker) configured
- No cloud deployment targets

**CI Pipeline:**
- None configured
- No GitHub Actions, GitLab CI, etc.

**Startup Scripts:**
- `setup.sh` / `setup.bat` - First-time installation
- `start.sh` / `start.bat` - Launch both servers
- `stop.sh` - Kill running processes (Linux/Mac)
- `restart.bat` - Windows restart script

## Environment Configuration

**Required env vars:**
- None strictly required (sensible defaults exist)

**Optional env vars:**
- `FILHARMONIA_BASE_DIR` - Data root directory
- `CORS_ORIGINS` - Comma-separated allowed origins
- `DATABASE_URL` - Custom SQLite path
- Folder name overrides (SORTED_FOLDER_NAME, etc.)

**Secrets location:**
- `.env` file in project root
- No sensitive API keys required
- No external service credentials needed

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Frontend-Backend Communication

**API Proxy:**
- Vite dev server proxies `/api` and `/health` to `http://localhost:8000`
- Configured in `frontend/vite.config.ts`

**API Structure:**
- All endpoints under `/api/v1/` prefix
- REST API with JSON payloads
- Routers:
  - `/api/v1/files` - File listing and management
  - `/api/v1/analyze` - Single file analysis
  - `/api/v1/batch` - Batch analysis jobs
  - `/api/v1/csv_parser` - CSV reading/writing
  - `/api/v1/audio` - Audio streaming
  - `/api/v1/waveform` - Waveform data generation
  - `/api/v1/sort` - File sorting operations
  - `/api/v1/export` - Tracklist export
  - `/api/v1/training` - Model training management
  - `/api/v1/uncertainty` - Low-confidence segment review

**Health Endpoint:**
- `GET /health` - Returns `{"status": "healthy"}`
- `GET /api/v1/info` - Returns version and GPU status

## GPU Integration

**CUDA Support:**
- PyTorch with CUDA 12.1 support
- Automatic device detection: `torch.cuda.is_available()`
- Falls back to CPU if no GPU detected
- Thread limiting for CPU inference: `torch.set_num_threads(2)`

**GPU Usage:**
- Training: Full GPU utilization
- Inference: Batch processing (32 segments at once)
- Memory management handled by PyTorch automatic garbage collection

## Model Registry

**Local Registry:**
- File: `RECOGNITION_MODELS/models_metadata.json`
- Tracks all trained models with metadata
- Active model copied to `ast_active.pth`
- Model versioning via timestamps (e.g., `ast_20251009_222204.pth`)
- Implementation: `backend/app/services/model_registry.py`

**Pre-trained Model Source:**
- Hugging Face: `huggingface.co/szymontex/filharmonia-ast`
- Base model: MIT AST (AudioSet pre-trained)
- Size: ~1.03 GB per model file

---

*Integration audit: 2026-01-20*

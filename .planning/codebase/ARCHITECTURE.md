# Architecture

**Analysis Date:** 2026-01-20

## Pattern Overview

**Overall:** Monolithic Full-Stack Application with Background Workers

**Key Characteristics:**
- React SPA frontend communicating with FastAPI backend via REST
- Background processing for CPU-intensive ML tasks (subprocess/multiprocessing)
- File-based job status tracking for inter-process communication
- Singleton service pattern for ML model management
- No external database - file system and JSON for persistence

## Layers

**Frontend (React SPA):**
- Purpose: User interface for audio analysis workflow
- Location: `frontend/src/`
- Contains: Pages, components, constants
- Depends on: Backend REST API via axios
- Used by: End users via browser

**API Layer (FastAPI Routers):**
- Purpose: HTTP endpoints exposing backend functionality
- Location: `backend/app/api/v1/`
- Contains: Route handlers, request/response models (Pydantic)
- Depends on: Service layer, config
- Used by: Frontend

**Service Layer:**
- Purpose: Core business logic, ML inference, training
- Location: `backend/app/services/`
- Contains: Analysis, training, inference, model registry
- Depends on: Config, PyTorch/transformers, librosa
- Used by: API layer, Workers

**Workers:**
- Purpose: Isolated CPU-intensive background processing
- Location: `backend/app/workers/`
- Contains: analyze_worker.py (subprocess for single file analysis)
- Depends on: Services (imported inside worker process)
- Used by: API layer spawns as subprocess

**Configuration:**
- Purpose: Environment and path configuration
- Location: `backend/app/config.py`
- Contains: Settings class with env var support
- Depends on: .env file, environment variables
- Used by: All backend code

## Data Flow

**Audio Analysis Flow:**

1. Frontend calls POST `/api/v1/analyze/` with MP3 path
2. API creates job ID, writes initial status to `/tmp/filharmonia_jobs/{job_id}.json`
3. API spawns subprocess running `analyze_worker.py` with job_id and path
4. Worker imports services, loads AST model, processes audio in batches
5. Worker writes progress updates to JSON file during processing
6. Worker generates CSV with predictions to `SORTED/ANALYSIS_RESULTS/`
7. Worker writes final status (completed/failed) to JSON file
8. Frontend polls GET `/api/v1/analyze/status/{job_id}` for updates

**Batch Analysis Flow:**

1. Frontend calls POST `/api/v1/analyze/batch` with year/month or paths
2. API spawns `multiprocessing.Process` for batch worker
3. Batch worker iterates files, calling `AnalyzeService.analyze_file()` for each
4. Progress written to JSON file after each file/batch
5. Frontend polls `/api/v1/analyze/batch/{job_id}` for status

**Model Training Flow:**

1. Frontend calls POST `/api/v1/training/start`
2. API spawns background thread for AST training
3. Thread loads WAV files from `TRAINING_DATA/DATA/{CLASS}/`
4. PyTorch training loop with custom callback updating job status
5. Model saved to `RECOGNITION_MODELS/ast_{timestamp}.pth`
6. Metadata updated in `models_metadata.json`

**State Management:**
- Job state: JSON files in `/tmp/filharmonia_jobs/`
- Model registry: `RECOGNITION_MODELS/models_metadata.json`
- Edited CSVs tracking: `.claude/edited_csvs.txt`
- Exported segments: `TRAINING_DATA/exported_segments.csv`

## Key Abstractions

**AnalyzeService:**
- Purpose: Audio analysis orchestration
- Location: `backend/app/services/analyze.py`
- Pattern: Singleton via `get_analyze_service()`

**ASTInferenceService:**
- Purpose: PyTorch AST model inference
- Location: `backend/app/services/ast_inference.py`
- Pattern: Singleton via `get_ast_inference_service()`
- Key method: `predict_batch(audio_segments)` returns `[(class_name, confidence), ...]`

**ASTTrainingService:**
- Purpose: Model retraining with progress tracking
- Location: `backend/app/services/ast_training.py`
- Pattern: Singleton via `get_ast_training_service()`

**Model Registry:**
- Purpose: Track trained models and active model
- Location: `backend/app/services/model_registry.py`
- Functions: `get_active_model_id()`, `set_active_model()`, `is_csv_edited()`

**Settings:**
- Purpose: Centralized configuration
- Location: `backend/app/config.py`
- Pattern: Module-level singleton `settings`
- Key paths: `SORTED_FOLDER`, `TRAINING_DATA_FOLDER`, `AST_MODEL_PATH`

## Entry Points

**Backend Main:**
- Location: `backend/app/main.py`
- Triggers: `uvicorn` or direct `python -m uvicorn`
- Responsibilities: FastAPI app creation, CORS, router registration, lifespan management

**Frontend Main:**
- Location: `frontend/src/main.tsx`
- Triggers: Vite dev server or build
- Responsibilities: React root render, StrictMode wrapper

**Analysis Worker:**
- Location: `backend/app/workers/analyze_worker.py`
- Triggers: Spawned by analyze API as subprocess
- Responsibilities: Isolated analysis process with CPU thread limits

## Error Handling

**Strategy:**
- Try/catch with status updates for background jobs
- HTTPException for API errors
- Job status JSON records error messages

**Patterns:**
- Workers catch all exceptions, write to job status file as `failed`
- Services raise exceptions, let API layer handle
- Frontend displays error toasts from API error responses

## Cross-Cutting Concerns

**Logging:**
- `print()` statements in services/workers
- Worker stdout captured to `.log` files in `/tmp/filharmonia_jobs/`

**Validation:**
- Pydantic models for request/response validation
- Path existence checks before processing

**Authentication:**
- None - local application, no auth required

**CPU Resource Management:**
- Thread limits set via environment variables before torch import
- `torch.set_num_threads(2)` to prevent CPU monopolization
- Subprocess isolation for analysis to keep server responsive

---

*Architecture analysis: 2026-01-20*

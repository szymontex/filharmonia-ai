# Codebase Structure

**Analysis Date:** 2026-01-20

## Directory Layout

```
filharmonia-ai/
├── backend/                    # Python FastAPI backend
│   ├── app/                    # Main application package
│   │   ├── api/                # API layer
│   │   │   ├── v1/             # Versioned API routes
│   │   │   │   ├── analyze.py  # Single file analysis
│   │   │   │   ├── batch.py    # Batch analysis
│   │   │   │   ├── audio.py    # Audio streaming
│   │   │   │   ├── csv_parser.py  # CSV parsing/saving
│   │   │   │   ├── export.py   # Training data export
│   │   │   │   ├── files.py    # File browser
│   │   │   │   ├── sort.py     # Recording sorting
│   │   │   │   ├── training.py # Model training
│   │   │   │   ├── uncertainty.py # Uncertainty review
│   │   │   │   └── waveform.py # Waveform generation
│   │   │   └── __init__.py
│   │   ├── services/           # Business logic
│   │   │   ├── analyze.py      # Analysis orchestration
│   │   │   ├── ast_inference.py # PyTorch AST inference
│   │   │   ├── ast_training.py # AST model training
│   │   │   ├── model_registry.py # Model management
│   │   │   ├── sort.py         # File sorting service
│   │   │   └── training.py     # Legacy Keras training
│   │   ├── workers/            # Background workers
│   │   │   └── analyze_worker.py # Subprocess analysis
│   │   ├── config.py           # Configuration
│   │   ├── main.py             # FastAPI app entry
│   │   └── __init__.py
│   ├── scripts/                # Utility scripts
│   │   ├── delete_unedited_csvs.py
│   │   ├── measure_train_acc.py
│   │   ├── migrate_metadata.py
│   │   └── set_active_model.py
│   └── pytorch_dataset.py      # PyTorch dataset class
├── frontend/                   # React TypeScript frontend
│   ├── src/
│   │   ├── pages/              # Page components
│   │   │   ├── AnalysisMonitor.tsx
│   │   │   ├── CalendarBrowser.tsx
│   │   │   ├── CsvViewer.tsx   # Main editor
│   │   │   ├── SortManager.tsx
│   │   │   ├── TrainingManager.tsx
│   │   │   ├── UncertaintyReview.tsx
│   │   │   └── WaveformEditor.tsx
│   │   ├── components/         # Reusable components
│   │   │   ├── StickyPlayer.tsx # Audio player + waveform
│   │   │   └── Toast.tsx       # Notification toasts
│   │   ├── constants/
│   │   │   └── colors.ts       # Class color definitions
│   │   ├── App.tsx             # Root app with routing
│   │   ├── main.tsx            # Entry point
│   │   ├── index.css           # Tailwind CSS
│   │   └── vite-env.d.ts
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── docs/                       # Documentation
│   └── images/                 # Documentation images
├── .planning/                  # Planning documents
│   └── codebase/               # Architecture docs
├── .env                        # Environment config (not committed)
├── .env.example                # Environment template
├── setup.sh / setup.bat        # Setup scripts
├── start.sh / start.bat        # Start scripts
├── stop.sh                     # Stop script
├── restart.bat                 # Restart script
├── verify_installation.py      # Installation verification
├── README.md
└── LICENSE
```

## Directory Purposes

**`backend/app/api/v1/`:**
- Purpose: REST API endpoints organized by feature
- Contains: FastAPI routers with Pydantic models
- Key files: `analyze.py`, `batch.py`, `training.py`, `csv_parser.py`

**`backend/app/services/`:**
- Purpose: Core business logic, ML operations
- Contains: Singleton services for analysis, inference, training
- Key files: `ast_inference.py` (inference), `ast_training.py` (training), `model_registry.py`

**`backend/app/workers/`:**
- Purpose: Isolated background processing
- Contains: Standalone Python scripts run as subprocesses
- Key files: `analyze_worker.py`

**`frontend/src/pages/`:**
- Purpose: Full-page components for each feature
- Contains: React components with local state
- Key files: `CsvViewer.tsx` (main editor), `CalendarBrowser.tsx`

**`frontend/src/components/`:**
- Purpose: Reusable UI components
- Contains: Shared components used across pages
- Key files: `StickyPlayer.tsx` (audio + waveform), `Toast.tsx`

**`frontend/src/constants/`:**
- Purpose: Shared constant values
- Contains: Color definitions, type exports
- Key files: `colors.ts` (CLASS_COLORS)

**`backend/scripts/`:**
- Purpose: Utility/maintenance scripts
- Contains: One-off scripts for data migration, measurement
- Key files: `set_active_model.py`, `measure_train_acc.py`

## Key File Locations

**Entry Points:**
- `backend/app/main.py`: FastAPI application
- `frontend/src/main.tsx`: React application
- `backend/app/workers/analyze_worker.py`: Analysis subprocess

**Configuration:**
- `backend/app/config.py`: Backend settings with env var support
- `.env`: Environment variables (FILHARMONIA_BASE_DIR, CORS_ORIGINS)
- `frontend/vite.config.ts`: Vite build config with API proxy

**Core Logic:**
- `backend/app/services/ast_inference.py`: AST model inference
- `backend/app/services/analyze.py`: Analysis orchestration
- `backend/app/services/ast_training.py`: Model training

**API Routes:**
- `backend/app/api/v1/analyze.py`: Single file analysis endpoints
- `backend/app/api/v1/batch.py`: Batch analysis + job listing
- `backend/app/api/v1/training.py`: Training management

**Frontend Pages:**
- `frontend/src/pages/CsvViewer.tsx`: Main CSV editor (1268 lines)
- `frontend/src/pages/CalendarBrowser.tsx`: Recording browser
- `frontend/src/components/StickyPlayer.tsx`: Waveform + audio player

**Testing:**
- No test files currently present

## Naming Conventions

**Files:**
- Python: `snake_case.py` (e.g., `ast_inference.py`, `model_registry.py`)
- TypeScript: `PascalCase.tsx` for components (e.g., `CsvViewer.tsx`)
- TypeScript: `camelCase.ts` for non-components (e.g., `colors.ts`)

**Directories:**
- All lowercase with underscores where needed
- `api/v1/` for versioned API
- `services/`, `workers/`, `pages/`, `components/`

**Functions:**
- Python: `snake_case` (e.g., `get_analyze_service()`, `analyze_file()`)
- TypeScript: `camelCase` (e.g., `loadCsvList()`, `handlePlayRecording()`)

**Classes:**
- Python: `PascalCase` (e.g., `AnalyzeService`, `ASTInferenceService`)
- TypeScript: `PascalCase` interfaces (e.g., `Track`, `Recording`)

**Constants:**
- Python: `UPPER_SNAKE_CASE` (e.g., `LABELS`, `SAMPLE_RATE`)
- TypeScript: `UPPER_SNAKE_CASE` for constants (e.g., `CLASS_COLORS`)

**API Routes:**
- Prefix: `/api/v1/{resource}`
- RESTful: GET for reads, POST for actions, DELETE for removal
- Examples: `/api/v1/analyze/`, `/api/v1/files/sorted`, `/api/v1/training/start`

## Where to Add New Code

**New API Endpoint:**
- Create/modify router in `backend/app/api/v1/{feature}.py`
- Add Pydantic models for request/response in same file
- Register router in `backend/app/main.py` if new file

**New Service:**
- Create `backend/app/services/{name}.py`
- Use singleton pattern with `get_{name}_service()` function
- Import in API layer as needed

**New Background Worker:**
- Create `backend/app/workers/{name}_worker.py`
- Set CPU thread limits before torch imports
- Write status to JSON in `/tmp/filharmonia_jobs/`

**New Frontend Page:**
- Create `frontend/src/pages/{PageName}.tsx`
- Add navigation in `frontend/src/App.tsx` (page state + component render)
- Add button in HomePage component

**New Frontend Component:**
- Create `frontend/src/components/{ComponentName}.tsx`
- Import in pages that need it

**New Constant:**
- Backend: Add to `backend/app/config.py` Settings class
- Frontend: Create/modify in `frontend/src/constants/`

**New Utility Function:**
- Backend service: Add to relevant service file
- Standalone script: Add to `backend/scripts/`

## Special Directories

**`/tmp/filharmonia_jobs/`:**
- Purpose: Job status files for IPC between API and workers
- Generated: Yes, at runtime
- Committed: No (temp directory)

**`FILHARMONIA_DATA/` (external):**
- Purpose: Audio files, analysis results, training data, models
- Generated: Yes, by application
- Committed: No (configured via FILHARMONIA_BASE_DIR env var)
- Structure:
  - `SORTED/` - Organized recordings by date
  - `SORTED/ANALYSIS_RESULTS/` - CSV analysis outputs
  - `NAGRANIA_KONCERTOW/` - Raw recordings to sort
  - `TRAINING_DATA/DATA/{CLASS}/` - Training WAV files
  - `RECOGNITION_MODELS/` - Trained model files

**`.planning/codebase/`:**
- Purpose: Architecture documentation
- Generated: By analysis commands
- Committed: Yes

---

*Structure analysis: 2026-01-20*

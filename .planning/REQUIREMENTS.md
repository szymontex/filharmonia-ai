# Requirements: Filharmonia AI

**Defined:** 2026-01-20
**Updated:** 2026-01-21 for v0.9 milestone
**Core Value:** Zamiast reczenie sluchac ~6-8h nagran/tyg, AI robi to za ciebie.

## v0.9 Requirements

Based on deep code audit and technology review. Each requirement references specific code location.

### CRITICAL — Error Handling & Security

- [ ] **CRIT-01**: Replace bare `except:` at `main.py:84` with `except ImportError`
- [ ] **CRIT-02**: Replace bare `except:` at `analyze.py:36,49` with specific exceptions + logging
- [ ] **CRIT-03**: Replace bare `except:` at `batch.py:39` with `except json.JSONDecodeError` + log corrupted file
- [ ] **CRIT-04**: Replace bare `except:` at `files.py:49,66` with specific exceptions
- [ ] **CRIT-05**: Replace bare `except:` at `ast_inference.py:189-190` with specific exceptions
- [ ] **CRIT-06**: Replace bare `except:` at `ast_training.py:214` with specific exceptions
- [ ] **CRIT-07**: Add path traversal prevention in `files.py:104` — `Path.resolve().relative_to(SORTED_FOLDER)`
- [ ] **CRIT-08**: Add path traversal prevention in `csv_parser.py:181`
- [ ] **CRIT-09**: Add path traversal prevention in `waveform.py`
- [ ] **CRIT-10**: Add global FastAPI exception handler with error IDs

### CRITICAL — Memory & Process Management

- [ ] **CRIT-11**: Fix memory leak in `analyze.py:48` — add TTL cleanup for `_single_jobs` dict
- [ ] **CRIT-12**: Fix memory leak in `batch.py:48-49` — add TTL cleanup for `_jobs` dict
- [ ] **CRIT-13**: Fix race condition in `batch.py:138,164` — atomic writes for job status
- [ ] **CRIT-14**: Fix zombie processes in `analyze.py:90` — store process handle, cleanup on shutdown
- [ ] **CRIT-15**: Fix blocking I/O in `csv_parser.py:273-276` — use `asyncio.to_thread()`

### HIGH — Performance Fixes

- [ ] **PERF-01**: Fix CSV double-read in `batch.py:335-341` — single `pd.read_csv` with `nrows=100`
- [ ] **PERF-02**: Fix N+1 query in `uncertainty.py:273-293` — read full CSV once, extract header and data
- [ ] **PERF-03**: Fix regex recompilation in `files.py:86` — compile pattern at module level
- [ ] **PERF-04**: Add waveform caching — pre-generate during analysis, cache to filesystem
- [ ] **PERF-05**: Replace polling (2s interval) with exponential backoff — `CsvViewer.tsx:62-80`

### HIGH — Technology Migration: pandas to Polars

- [ ] **TECH-01**: Install polars: `pip install polars`
- [ ] **TECH-02**: Migrate `csv_parser.py` from pandas to polars (5-30x faster CSV)
- [ ] **TECH-03**: Migrate `uncertainty.py` from pandas to polars
- [ ] **TECH-04**: Migrate `batch.py` from pandas to polars
- [ ] **TECH-05**: Remove pandas from requirements after migration complete

### HIGH — Cross-Platform Paths

- [ ] **PATH-01**: Remove hardcoded `Y:\\!_FILHARMONIA\\SORTED\\` from `CsvViewer.tsx:206`
- [ ] **PATH-02**: Remove hardcoded path from `CalendarBrowser.tsx:268-269`
- [ ] **PATH-03**: Add backend API endpoint for MP3 path resolution
- [ ] **PATH-04**: Use `tempfile.gettempdir()` instead of `/tmp/` in `analyze.py:15-16`
- [ ] **PATH-05**: Use `tempfile.gettempdir()` in `batch.py:16-17`
- [ ] **PATH-06**: Use `tempfile.gettempdir()` in `analyze_worker.py:22`

### HIGH — Frontend Component Refactor

- [ ] **COMP-01**: Split `CsvViewer.tsx` (1268 lines) into:
  - `TrackTable.tsx` (~300 lines) — table rendering
  - `CsvSelector.tsx` (~150 lines) — file selection
  - `PlayerControls.tsx` (~200 lines) — audio player integration
  - `CsvViewer.tsx` (~200 lines) — orchestration only
- [ ] **COMP-02**: Extract `useTrackEditor` hook from CsvViewer
- [ ] **COMP-03**: Extract `useAudioPlayer` hook
- [ ] **COMP-04**: Extract `useAutosave` hook
- [ ] **COMP-05**: Fix duplicated time calculation in `CsvViewer.tsx:230-284` — extract utility

### HIGH — Type Safety

- [ ] **TYPE-01**: Add return type hints to `csv_parser.py:52` (`get_duration`)
- [ ] **TYPE-02**: Add return type hints to `csv_parser.py:70` (`parse_segment_time`)
- [ ] **TYPE-03**: Add return type hints to `csv_parser.py:198,203,208,215,222`
- [ ] **TYPE-04**: Fix time parsing in `uncertainty.py:21-22` — use `float(parts[2])` for fractional seconds

### MEDIUM — Code Cleanup

- [ ] **CLEAN-01**: Delete unused `backend/app/services/training.py` (legacy Keras, 535 lines)
- [ ] **CLEAN-02**: Remove unused `howler` from `frontend/package.json`
- [ ] **CLEAN-03**: Remove `@types/howler` if exists
- [ ] **CLEAN-04**: Implement `handlePlayRecording` in `CalendarBrowser.tsx:108` or remove button
- [ ] **CLEAN-05**: Fix autosave atomicity in `CsvViewer.tsx:671` — write to .tmp then rename
- [ ] **CLEAN-06**: Sanitize filenames in `export.py:169` — `re.sub(r'[<>:"|?*]', '_', filename)`

### MEDIUM — UX Improvements

- [ ] **UX-01**: Add keyboard shortcuts — spacebar play/pause
- [ ] **UX-02**: Add Ctrl+S explicit save
- [ ] **UX-03**: Add Ctrl+Z undo (single step first)
- [ ] **UX-04**: Add number keys 1-5 for class cycling
- [ ] **UX-05**: Add debounce abort in `CsvViewer.tsx:145-158` — cancel in-flight requests
- [ ] **UX-06**: Standardize error response format — always `{"status": "error", "message": "...", "code": "..."}`
- [ ] **UX-07**: Add progress indicators — "Loading... Analyzing... Saving..."

### MEDIUM — Infrastructure

- [ ] **INFRA-01**: SQLite job registry zamiast JSON files
- [ ] **INFRA-02**: Add request timeout for long operations
- [ ] **INFRA-03**: Configure SQLAlchemy connection pooling in `config.py:52`
- [ ] **INFRA-04**: Add startup validation for audio backends
- [ ] **INFRA-05**: Pin exact PyTorch versions with platform suffix

### LOW — GPU Support (Later in milestone)

- [ ] **GPU-01**: Unified device detection (distinguish NVIDIA from AMD)
- [ ] **GPU-02**: torch.compile for GPU acceleration
- [ ] **GPU-03**: ONNX export for CPU optimization
- [ ] **GPU-04**: INT8 quantization for CPU inference
- [ ] **GPU-05**: ROCm 6.4 support on Linux
- [ ] **GPU-06**: ROCm Windows preview support (with warnings)

### LOW — Frontend Upgrades (Optional)

- [ ] **FRONT-01**: Upgrade React 18 to React 19
- [ ] **FRONT-02**: Consider wavesurfer.js migration for better waveform UX
- [ ] **FRONT-03**: Add confidence threshold auto-tuning (dynamic instead of hardcoded 0.7)

## v2 Requirements (Next Milestone)

- ZAIKS export automation
- Copyright checking AI
- WebSocket for real-time job status
- Spectrogram view
- Minimap for long recordings

## Out of Scope

| Feature | Reason |
|---------|--------|
| FastAPI to Litestar migration | Bottleneck is audio I/O, not API layer |
| AST to FastAST migration | Requires retraining, current accuracy is fine |
| PostgreSQL migration | SQLite sufficient for single-user |
| Authentication | Local tool, trusted network |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CRIT-01 | 1 | Pending |
| CRIT-02 | 1 | Pending |
| CRIT-03 | 1 | Pending |
| CRIT-04 | 1 | Pending |
| CRIT-05 | 1 | Pending |
| CRIT-06 | 1 | Pending |
| CRIT-07 | 1 | Pending |
| CRIT-08 | 1 | Pending |
| CRIT-09 | 1 | Pending |
| CRIT-10 | 1 | Pending |
| CRIT-11 | 3 | Pending |
| CRIT-12 | 3 | Pending |
| CRIT-13 | 3 | Pending |
| CRIT-14 | 3 | Pending |
| CRIT-15 | 3 | Pending |
| PERF-01 | 4 | Pending |
| PERF-02 | 4 | Pending |
| PERF-03 | 4 | Pending |
| PERF-04 | 4 | Pending |
| PERF-05 | 3 | Pending |
| TECH-01 | 4 | Pending |
| TECH-02 | 4 | Pending |
| TECH-03 | 4 | Pending |
| TECH-04 | 4 | Pending |
| TECH-05 | 4 | Pending |
| PATH-01 | 1 | Pending |
| PATH-02 | 1 | Pending |
| PATH-03 | 1 | Pending |
| PATH-04 | 1 | Pending |
| PATH-05 | 1 | Pending |
| PATH-06 | 1 | Pending |
| COMP-01 | 5 | Pending |
| COMP-02 | 5 | Pending |
| COMP-03 | 5 | Pending |
| COMP-04 | 5 | Pending |
| COMP-05 | 5 | Pending |
| TYPE-01 | 1 | Pending |
| TYPE-02 | 1 | Pending |
| TYPE-03 | 1 | Pending |
| TYPE-04 | 1 | Pending |
| CLEAN-01 | 5 | Pending |
| CLEAN-02 | 5 | Pending |
| CLEAN-03 | 5 | Pending |
| CLEAN-04 | 2 | Pending |
| CLEAN-05 | 2 | Pending |
| CLEAN-06 | 5 | Pending |
| UX-01 | 2 | Pending |
| UX-02 | 2 | Pending |
| UX-03 | 2 | Pending |
| UX-04 | 2 | Pending |
| UX-05 | 2 | Pending |
| UX-06 | 2 | Pending |
| UX-07 | 2 | Pending |
| INFRA-01 | 3 | Pending |
| INFRA-02 | 3 | Pending |
| INFRA-03 | 3 | Pending |
| INFRA-04 | 1 | Pending |
| INFRA-05 | 1 | Pending |
| GPU-01 | 6 | Pending |
| GPU-02 | 6 | Pending |
| GPU-03 | 6 | Pending |
| GPU-04 | 6 | Pending |
| GPU-05 | 6 | Pending |
| GPU-06 | 6 | Pending |
| FRONT-01 | 6 | Pending |
| FRONT-02 | 6 | Pending |
| FRONT-03 | 6 | Pending |

**Coverage:** 62/62 requirements mapped to phases

---
*Requirements defined: 2026-01-20 after deep code audit*
*Traceability added: 2026-01-21 with roadmap creation*

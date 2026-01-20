# Requirements: Filharmonia AI

**Defined:** 2026-01-20
**Core Value:** Zamiast ręcznie słuchać ~6-8h nagrań/tyg, AI robi to za ciebie.

## v1 Requirements

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

### HIGH — Technology Migration: pandas → Polars

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

- [ ] **FRONT-01**: Upgrade React 18 → React 19
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
| FastAPI → Litestar migration | Bottleneck is audio I/O, not API layer |
| AST → FastAST migration | Requires retraining, current accuracy is fine |
| PostgreSQL migration | SQLite sufficient for single-user |
| Authentication | Local tool, trusted network |

## Traceability

| Category | Requirements | Priority |
|----------|--------------|----------|
| CRITICAL (security, memory, crashes) | 15 | Must fix |
| HIGH (performance, paths, components) | 21 | Should fix |
| MEDIUM (cleanup, UX, infra) | 17 | Nice to fix |
| LOW (GPU, frontend) | 9 | Optional |
| **Total** | **62** | |

---
*Requirements defined: 2026-01-20 after deep code audit*

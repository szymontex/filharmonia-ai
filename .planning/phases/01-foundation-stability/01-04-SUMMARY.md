---
phase: 01-foundation-stability
plan: 04
subsystem: backend/paths
tags: [cross-platform, tempfile, windows, macos, linux]
dependency-graph:
  requires: []
  provides: [cross-platform-temp-directories]
  affects: [01-05, 01-06]
tech-stack:
  added: []
  patterns: [tempfile.gettempdir()]
key-files:
  created: []
  modified:
    - backend/app/api/v1/analyze.py
    - backend/app/api/v1/batch.py
    - backend/app/workers/analyze_worker.py
decisions: []
metrics:
  duration: 2m 8s
  completed: 2026-01-21
---

# Phase 01 Plan 04: Cross-Platform Temp Directory Summary

**One-liner:** Replace hardcoded `/tmp/` paths with `tempfile.gettempdir()` for Windows/macOS/Linux compatibility.

## What Was Done

Replaced all hardcoded `/tmp/filharmonia_jobs` paths with cross-platform `tempfile.gettempdir()` calls in three backend files:

1. **analyze.py** - Single file analysis API endpoint
2. **batch.py** - Batch analysis API endpoint
3. **analyze_worker.py** - Standalone worker process

### Changes Made

| File | Before | After |
|------|--------|-------|
| analyze.py | `Path("/tmp/filharmonia_jobs")` | `Path(tempfile.gettempdir()) / "filharmonia_jobs"` |
| batch.py | `Path("/tmp/filharmonia_jobs")` | `Path(tempfile.gettempdir()) / "filharmonia_jobs"` |
| analyze_worker.py | `Path("/tmp/filharmonia_jobs")` | `Path(tempfile.gettempdir()) / "filharmonia_jobs"` |

### Platform Behavior

- **Linux:** Uses `/tmp/filharmonia_jobs` (same as before)
- **Windows:** Uses `C:\Users\<user>\AppData\Local\Temp\filharmonia_jobs`
- **macOS:** Uses `/var/folders/.../filharmonia_jobs` or `/tmp/filharmonia_jobs`
- **Custom:** Respects `TMPDIR`/`TEMP` environment variables if set

## Requirements Completed

- [x] PATH-04: analyze.py uses `tempfile.gettempdir()` instead of `/tmp/`
- [x] PATH-05: batch.py uses `tempfile.gettempdir()` instead of `/tmp/`
- [x] PATH-06: analyze_worker.py uses `tempfile.gettempdir()` instead of `/tmp/`

## Commits

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Fix temp directory in analyze.py | 3b20818 | backend/app/api/v1/analyze.py |
| 2 | Fix temp directory in batch.py | ec1e4b1 | backend/app/api/v1/batch.py |
| 3 | Fix temp directory in analyze_worker.py | 94094fb | backend/app/workers/analyze_worker.py |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

1. All three files use `tempfile.gettempdir()`: **PASS** (3/3)
2. No hardcoded `/tmp/` paths remain in backend: **PASS** (0 occurrences)
3. Backend imports successfully: **PASS**

## Next Phase Readiness

Plan 01-04 complete. No blockers for subsequent plans.

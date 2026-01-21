---
phase: 01-foundation-stability
plan: 05
subsystem: backend/files-api
tags: [mp3-resolution, csv-parsing, cross-platform, path-handling]
dependency-graph:
  requires: []
  provides: [mp3-path-resolution-endpoint, csv-metadata-endpoint]
  affects: [01-06]
tech-stack:
  added: []
  patterns: [regex-parsing, pathlib, pydantic-response-models]
key-files:
  created: []
  modified:
    - backend/app/api/v1/files.py
decisions: []
metrics:
  duration: 3m 15s
  completed: 2026-01-21
---

# Phase 01 Plan 05: MP3 Path Resolution Endpoint Summary

**One-liner:** Backend API endpoint resolves MP3 paths from CSV filenames using regex parsing and SORTED_FOLDER setting.

## What Was Done

Created two new API endpoints in `files.py` that resolve MP3 file paths from CSV prediction filenames, enabling the frontend to use server-side path logic instead of hardcoded Windows paths.

### Endpoints Added

| Endpoint | Purpose | Response Model |
|----------|---------|----------------|
| `GET /api/v1/files/mp3-for-csv` | Resolve MP3 path from CSV path | `Mp3PathResponse` |
| `GET /api/v1/files/csv-metadata` | Get full CSV metadata including MP3 path | `CsvMetadataResponse` |

### CSV Filename Pattern Supported

Format: `predictions_{songName}_{YYYY-MM-DD}[_{HH-MM}].csv`

Variations handled:
- `predictions_SONG042_2025-09-27.csv` (basic)
- `predictions_SONG042_2025-09-27_autosave.csv` (autosave suffix)
- `predictions_SONG042_2025-09-27_14-30.csv` (time suffix)

### Response Structure

**Mp3PathResponse:**
```json
{
  "mp3_path": "/path/to/SORTED/2025/09/27/SONG042.MP3",
  "recording_date": "2025-09-27",
  "exists": true
}
```

**CsvMetadataResponse:**
```json
{
  "csv_path": "/path/to/predictions_SONG042_2025-09-27.csv",
  "mp3_path": "/path/to/SORTED/2025/09/27/SONG042.MP3",
  "recording_date": "2025-09-27",
  "song_name": "SONG042",
  "mp3_exists": true,
  "csv_exists": true
}
```

## Requirements Completed

- [x] PATH-03: Backend API endpoint `/api/v1/files/mp3-for-csv` exists
- [x] Endpoint parses CSV filename pattern correctly
- [x] Endpoint handles `_autosave` suffix
- [x] Endpoint handles optional time suffix `_HH-MM`
- [x] Endpoint returns MP3 path using server's SORTED_FOLDER setting
- [x] Frontend can call this endpoint instead of constructing paths

## Commits

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add MP3 path resolution endpoint | a17e914 | backend/app/api/v1/files.py |
| 2 | Add CSV metadata endpoint | 5138e8c | backend/app/api/v1/files.py |

Note: Task 2 was committed as part of concurrent plan 01-07's commit due to parallel execution.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Add logging import and bare except handling**
- **Found during:** Code modifications by linter
- **Issue:** File lacked structured logging, bare `except:` clause
- **Fix:** Added `import logging` and logger initialization, improved exception handling
- **Files modified:** backend/app/api/v1/files.py
- **Commit:** Included in subsequent commits

**2. [Rule 3 - Blocking] Re-import cleanup**
- **Found during:** Task 1
- **Issue:** Redundant `import re` inside function when added at top level
- **Fix:** Removed redundant import from `list_analysis_results`
- **Files modified:** backend/app/api/v1/files.py
- **Commit:** a17e914

## Verification Results

1. Endpoint exists and responds: **PASS**
2. Handles basic CSV format: **PASS**
3. Handles autosave suffix: **PASS**
4. Handles time suffix: **PASS**
5. Returns proper structure: **PASS**
6. Uses SORTED_FOLDER setting: **PASS**

## Next Phase Readiness

Plan 01-05 complete. Frontend can now migrate path logic in Plan 01-06 using these endpoints.

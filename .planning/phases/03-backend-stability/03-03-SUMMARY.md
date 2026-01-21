---
phase: 03-backend-stability
plan: 03
subsystem: resource-management
tags: [process-cleanup, async-io, timeout-middleware, job-registry]
dependency-graph:
  requires: [03-01]
  provides: [graceful-shutdown, non-blocking-io, request-timeout, job-persistence]
  affects: [frontend-reliability, server-stability]
tech-stack:
  added: []
  patterns:
    - signal-handler-shutdown
    - asyncio-to-thread-io
    - starlette-middleware
    - sqlite-job-persistence
key-files:
  created:
    - backend/app/core/middleware.py
  modified:
    - backend/app/main.py
    - backend/app/api/v1/csv_parser.py
    - backend/app/api/v1/analyze.py
    - backend/app/api/v1/batch.py
decisions:
  - key: process-termination-timeout
    choice: 5 second timeout before force kill
    reason: Balance between graceful shutdown and not hanging on unresponsive workers
  - key: request-timeout-exclusion
    choice: Exclude /analyze endpoints from timeout
    reason: Analysis endpoints return immediately with job_id; heavy work is in subprocess
  - key: job-registry-fallback
    choice: File -> Cache -> SQLite lookup order
    reason: Files most current during active processing; SQLite for restart recovery
metrics:
  duration: 3 minutes 12 seconds
  completed: 2026-01-21
---

# Phase 03 Plan 03: Resource Cleanup Summary

**One-liner:** Process termination on shutdown, async file I/O, 60s request timeout, and SQLite job persistence integration

## Commits

| Hash | Type | Message |
|------|------|---------|
| 6cdc584 | feat | Add process cleanup on server shutdown (CRIT-14) |
| 13d4fb1 | feat | Wrap blocking file I/O with asyncio.to_thread (CRIT-15) |
| 7ba548d | feat | Add TimeoutMiddleware for request protection (INFRA-02) |
| c217e90 | feat | Wire JobRegistry into analyze.py and batch.py |

## What Was Done

### Task 1: Process Cleanup on Server Shutdown (CRIT-14)

**File:** `backend/app/main.py`

**Changes:**
- Added `terminate_all_workers()` function that:
  - Collects processes from both `analyze._processes` and `batch._processes`
  - Handles both `subprocess.Popen` (single analysis) and `multiprocessing.Process` (batch)
  - Uses 5s timeout then force kill for unresponsive workers
  - Clears process dicts after termination
- Registered SIGTERM/SIGINT signal handlers for graceful shutdown
- Updated lifespan shutdown to call `terminate_all_workers()` and cleanup old SQLite jobs

### Task 2: Async File I/O (CRIT-15)

**File:** `backend/app/api/v1/csv_parser.py`

**Changes:**
- Added `import asyncio`
- Wrapped blocking operations with `asyncio.to_thread()`:
  - `parse_csv`: CSV reading via pandas
  - `check_autosave`: File content comparison
  - `autosave_csv`: Autosave file writing
  - `save_csv`: Original file writing

This prevents event loop blocking during file I/O, allowing concurrent request handling.

### Task 3: Timeout Middleware (INFRA-02)

**File:** `backend/app/core/middleware.py` (new)

**Changes:**
- Created `TimeoutMiddleware` class extending `BaseHTTPMiddleware`
- 60 second default timeout for all endpoints
- Excludes `/analyze` endpoints (they return immediately with job_id)
- Returns structured 504 error on timeout

**File:** `backend/app/main.py`

**Changes:**
- Imported and registered `TimeoutMiddleware` after CORS middleware

### Task 4: JobRegistry Integration

**Files:** `backend/app/api/v1/analyze.py`, `backend/app/api/v1/batch.py`

**Changes:**
- Imported `get_job_registry` from job_registry service
- On job creation: persist to SQLite with `started_at` timestamp
- On status poll: update SQLite to keep persistent store in sync
- On job not found: fallback to SQLite for restart recovery

**Lookup order:**
1. Temp JSON file (most current during active processing)
2. In-memory TTLCache (fast access)
3. SQLite via JobRegistry (restart recovery)

## Technical Details

### Process Termination Flow

```
Server Shutdown
    |
    v
signal_handler() <-- SIGTERM/SIGINT
    |
    v
terminate_all_workers()
    |
    +-- For each subprocess.Popen:
    |       proc.terminate()
    |       proc.wait(timeout=5)  -- or kill() if timeout
    |
    +-- For each multiprocessing.Process:
    |       proc.terminate()
    |       proc.join(timeout=5)  -- or kill() if timeout
    |
    v
Clear process dicts
    |
    v
Cleanup old SQLite jobs (>7 days)
```

### Async I/O Pattern

```python
# Before (blocking):
df = pd.read_csv(csv_path)

# After (non-blocking):
def _read_csv():
    return pd.read_csv(csv_path)
df = await asyncio.to_thread(_read_csv)
```

### Job Persistence Flow

```
Job Start
    |
    +-- write_job_status() -> temp JSON file
    +-- _jobs[job_id] = status -> TTLCache
    +-- registry.create_job() -> SQLite
    |
    v
Status Poll
    |
    +-- read_job_status() -> temp JSON (if exists)
    +-- registry.update_job() -> SQLite (sync)
    |
    v
After Server Restart
    |
    +-- temp files may be gone
    +-- TTLCache is empty
    +-- registry.get_job() -> SQLite (recovery)
```

## Issues Fixed

| Issue ID | Description | Fix |
|----------|-------------|-----|
| CRIT-14 | Worker processes not terminated on shutdown | terminate_all_workers() with signal handlers |
| CRIT-15 | Blocking file I/O in csv_parser.py | asyncio.to_thread() wrapping |
| INFRA-02 | No request timeout protection | TimeoutMiddleware with 60s timeout |
| - | Jobs lost on server restart | SQLite persistence via JobRegistry |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

| Check | Status |
|-------|--------|
| terminate_all_workers function exists | Pass |
| signal.SIGTERM handler registered | Pass |
| asyncio.to_thread in csv_parser.py (4 occurrences) | Pass |
| TimeoutMiddleware import and registration | Pass |
| get_job_registry in analyze.py (4 occurrences) | Pass |
| get_job_registry in batch.py (4 occurrences) | Pass |

## Next Phase Readiness

Phase 03 Backend Stability is now complete:
- 03-01: SQLite job registry (complete)
- 03-02: Memory leak and race condition fixes (complete)
- 03-03: Resource cleanup (this plan, complete)
- 03-04: Frontend exponential backoff (complete)

Ready for Phase 02 (Core UX Polish) or Phase 04 (pending roadmap update).

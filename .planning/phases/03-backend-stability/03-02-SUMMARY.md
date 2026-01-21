---
phase: 03-backend-stability
plan: 02
subsystem: job-management
tags: [memory-leak, ttl-cache, atomic-write, cachetools]
dependency-graph:
  requires: []
  provides: [bounded-job-cache, atomic-file-writes]
  affects: [03-01, 03-03]
tech-stack:
  added: []
  patterns: [ttl-cache, atomic-write-pattern]
key-files:
  created: []
  modified:
    - backend/app/api/v1/analyze.py
    - backend/app/api/v1/batch.py
decisions:
  - key: ttl-single-jobs
    choice: 1 hour TTL, 100 max entries
    reason: Single file jobs are short-lived
  - key: ttl-batch-jobs
    choice: 4 hour TTL, 50 max entries
    reason: Batch jobs can be long-running
  - key: atomic-write-pattern
    choice: temp file + os.replace
    reason: Works on both Unix and Windows, prevents corruption on crash
metrics:
  duration: 2 minutes
  completed: 2026-01-21
---

# Phase 03 Plan 02: Memory Leak & Race Condition Fixes Summary

**One-liner:** TTLCache replaces unbounded dicts, atomic temp+replace for crash-safe writes

## What Was Done

### Task 1: Fix memory leak in analyze.py (CRIT-11)

**Changes:**
- Added `from cachetools import TTLCache` import
- Added `import os` for atomic file operations
- Replaced `_single_jobs = {}` with `TTLCache(maxsize=100, ttl=3600)` (1 hour TTL)
- Rewrote `write_job_status()` to use atomic temp file + os.replace pattern:
  - Creates temp file in same directory (same filesystem)
  - Writes JSON with fsync to ensure data on disk
  - Uses `os.replace()` for atomic rename
  - Cleans up temp file on error

**Commit:** c03d39f

### Task 2: Fix memory leak and race condition in batch.py (CRIT-12, CRIT-13)

**Changes:**
- Added `from cachetools import TTLCache` import
- Added `import os` for atomic file operations
- Replaced `_jobs = {}` with `TTLCache(maxsize=50, ttl=14400)` (4 hour TTL)
- Rewrote `write_job_status()` to use same atomic temp + replace pattern

**Commit:** 185dc73

## Technical Details

### TTLCache Configuration

| Variable | File | maxsize | ttl (seconds) | Rationale |
|----------|------|---------|---------------|-----------|
| `_single_jobs` | analyze.py | 100 | 3600 (1h) | Single file jobs complete quickly |
| `_jobs` | batch.py | 50 | 14400 (4h) | Batch jobs can take hours |

### Atomic Write Pattern

```python
def write_job_status(job_id: str, status: dict):
    job_file = get_job_file(job_id)
    fd, tmp_path = tempfile.mkstemp(suffix='.tmp', prefix=f'{job_id}_', dir=JOBS_DIR)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(status, f)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is on disk
        os.replace(tmp_path, job_file)  # Atomic
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
```

**Why this pattern:**
- `tempfile.mkstemp` in same dir guarantees same filesystem (required for atomic rename)
- `os.fsync()` ensures data is written to disk before rename
- `os.replace()` is atomic on POSIX (rename) and Windows (MoveFileEx with REPLACE_EXISTING)
- Cleanup on error prevents orphan temp files

## Issues Fixed

| Issue ID | Description | Fix |
|----------|-------------|-----|
| CRIT-11 | `_single_jobs` dict grows unbounded | TTLCache with automatic eviction |
| CRIT-12 | `_jobs` dict grows unbounded | TTLCache with automatic eviction |
| CRIT-13 | `write_job_status` can corrupt on crash | Atomic temp+replace pattern |

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

1. Both files import TTLCache from cachetools
2. `_single_jobs` and `_jobs` are TTLCache instances
3. `write_job_status` in both files uses temp file + os.replace pattern
4. cachetools already in requirements.txt (line 13)

## Next Phase Readiness

- Memory leaks in job tracking are fixed
- File corruption race condition is fixed
- These fixes are independent and do not block other plans
- Plan 03-01 (SQLite job registry) can proceed independently

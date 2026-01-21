---
phase: 01-foundation-stability
plan: 03
subsystem: api-error-handling
tags: [exception-handlers, type-hints, error-messages, debugging]

dependency-graph:
  requires: []
  provides: [global-exception-handlers, type-hinted-functions, error-id-tracking]
  affects: [all-api-endpoints, error-responses]

tech-stack:
  added: []
  patterns:
    - pattern: global-exception-handler-chain
      location: backend/app/main.py
    - pattern: error-id-correlation
      location: backend/app/main.py

key-files:
  created: []
  modified:
    - backend/app/main.py
    - backend/app/api/v1/csv_parser.py
    - backend/app/api/v1/uncertainty.py

decisions:
  - id: three-handler-chain
    choice: "StarletteHTTPException, RequestValidationError, Exception handlers"
    rationale: "Most specific to least specific order for proper exception hierarchy"
  - id: error-id-format
    choice: "8-character UUID prefix as error_id"
    rationale: "Short enough for user to report, unique enough to find in logs"

metrics:
  tasks: 3
  commits: 1
  duration: ~5 minutes
  completed: 2026-01-21
---

# Phase 01 Plan 03: Exception Handlers & Type Hints Summary

**One-liner:** Global exception handlers returning JSON with error IDs for all API errors, plus complete return type hints in csv_parser.py.

## What Was Done

### Task 1: Add global exception handlers to main.py

Exception handlers were added during Plan 01-07 execution (audio backend validation). The implementation includes:

- `http_exception_handler` - Handles 404, 403, and other HTTP errors from FastAPI/Starlette
- `validation_exception_handler` - Handles Pydantic validation errors (422) with details
- `global_exception_handler` - Catch-all for unexpected 500 errors with:
  - 8-character UUID error_id for correlation
  - Full traceback logged server-side
  - Generic message to client (no sensitive details leaked)

All handlers return consistent JSON format:
```json
{
  "status": "error",
  "message": "...",
  "type": "http_error|validation_error|server_error",
  "error_id": "abc12345"  // only for 500 errors
}
```

### Task 2: Add return type hints to csv_parser.py

Return type hint for `mark_csv_as_edited()` was added during Plan 01-02 execution. All functions now have explicit return types:

| Function | Return Type |
|----------|-------------|
| `get_duration(start, stop)` | `-> str` |
| `extract_tracks(df, threshold)` | `-> List[Track]` |
| `get_autosave_path(original_path)` | `-> str` |
| `time_to_seconds(time_str)` | `-> int` |
| `seconds_to_time(seconds)` | `-> str` |
| `escape_csv_field(field)` | `-> str` |
| `tracks_to_csv_content(tracks)` | `-> str` |
| `mark_csv_as_edited(csv_path)` | `-> None` |

### Task 3: Fix time parsing in uncertainty.py

The `time_to_seconds()` function already correctly used `float(parts[2])` for fractional seconds support. Updated docstring to clarify format support:

```python
def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS or HH:MM:SS.mmm to seconds"""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
```

Verified working with:
- `01:30:45` -> `5445.0`
- `00:00:05.5` -> `5.5`
- `00:00:00` -> `0.0`

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 49b1694 | docs | Clarify time_to_seconds handles fractional seconds |

Note: Tasks 1 and 2 were completed during earlier plan executions (01-07 and 01-02).

## Requirements Completed

| ID | Description | Status |
|----|-------------|--------|
| CRIT-10 | Global exception handlers in main.py | Done |
| TYPE-01 | get_duration return type hint | Done |
| TYPE-02 | extract_tracks return type hint | Done |
| TYPE-03 | csv_parser utility functions return type hints | Done |
| TYPE-04 | uncertainty.py fractional seconds parsing | Done |

## Deviations from Plan

**Pre-completed Work:**
- Task 1 (exception handlers) was implemented during Plan 01-07 alongside startup validation
- Task 2 (mark_csv_as_edited return type) was implemented during Plan 01-02 security updates

Both implementations met the plan requirements, so no rework was needed.

## Verification Results

```
[OK] 404 returns: {'status': 'error', 'message': 'Not Found', 'type': 'http_error'}
[OK] 422 returns: status=error, type=validation_error
[OK] Fractional seconds: 00:00:05.5 -> 5.5
[OK] All csv_parser.py functions have return type hints (0 missing)
```

## Next Phase Readiness

**Blockers:** None

**Notes:**
- Any new endpoints automatically benefit from global exception handlers
- New functions in csv_parser.py should follow the return type hint pattern
- Error IDs can be searched in server logs using format `[{error_id}]`

---
*Summary generated: 2026-01-21*

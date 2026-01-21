---
phase: 01-foundation-stability
plan: 01
subsystem: error-handling
tags: [python, exceptions, logging, debugging]

dependency-graph:
  requires: []
  provides:
    - specific-exception-handling
    - debug-logging
    - ctrl-c-termination
  affects:
    - all-error-handling-patterns

tech-stack:
  added: []
  patterns:
    - "except SpecificException as e: with logging"
    - "logger = logging.getLogger(__name__)"

file-tracking:
  created: []
  modified:
    - backend/app/main.py
    - backend/app/api/v1/analyze.py
    - backend/app/api/v1/batch.py
    - backend/app/api/v1/files.py
    - backend/app/services/ast_training.py
    - backend/app/services/analyze.py
    - backend/app/api/v1/csv_parser.py
    - backend/app/services/training.py

decisions:
  - id: DEC-01-01-001
    description: "Use debug-level logging for expected failures (ID3 tags, hardlinks)"
    rationale: "Avoid log noise while maintaining traceability"
  - id: DEC-01-01-002
    description: "Use warning-level for parsing failures that indicate data issues"
    rationale: "JSON decode errors may indicate corrupt job files"
  - id: DEC-01-01-003
    description: "Use most specific exception type available (OSError, JSONDecodeError, ValueError)"
    rationale: "More specific types enable better debugging and error handling"

metrics:
  duration: "~15 minutes"
  completed: "2026-01-21"
---

# Phase 01 Plan 01: Exception Handling Cleanup Summary

Replaced all bare `except:` clauses with specific exception types and added structured logging.

## One-Liner

Eliminated 10+ bare `except:` clauses across 8 backend files, adding specific exception types and debug logging.

## What Was Built

### Changes Made

1. **main.py** - `check_gpu()` function
   - `except:` -> `except ImportError:`
   - No logging needed (expected case when PyTorch not installed)

2. **analyze.py** (API) - `read_job_status()` function
   - `except:` -> `except json.JSONDecodeError as e:`
   - Added warning-level logging for parse failures

3. **batch.py** - Two locations
   - `read_job_status()`: `except json.JSONDecodeError as e:` with warning
   - `get_outdated_csvs()`: `except Exception as e:` with debug logging

4. **files.py** - ID3 tag parsing
   - `except:` -> `except Exception as e:`
   - Debug-level logging (many files lack ID3 tags)

5. **ast_training.py** - Two locations
   - Hardlink fallback: `except OSError as e:` (expected on some filesystems)
   - Duration reading: `except Exception as e:` (fallback to default)

6. **analyze.py** (service) - ID3 tag extraction
   - `except:` -> `except Exception as e:`
   - Debug-level logging

7. **csv_parser.py** - Duration calculation
   - `except:` -> `except (ValueError, IndexError) as e:`
   - Debug-level logging

8. **training.py** - Duration reading
   - `except:` -> `except Exception as e:`
   - Debug-level logging

### Pattern Applied

```python
# Before (BAD)
try:
    data = json.loads(file.read_text())
except:
    pass

# After (GOOD)
try:
    data = json.loads(file.read_text())
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse JSON from {file}: {e}")
    return None
```

## Commits

| Hash | Message |
|------|---------|
| bc841f6 | fix(01-01): replace bare except clauses in ast_training.py |
| d6bc00c | fix(01-01): replace remaining bare except clauses across backend |

Note: Task 1 changes (main.py, analyze.py, batch.py, files.py) were already committed in prior work.

## Success Criteria Met

- [x] CRIT-01: main.py uses `except ImportError:` in check_gpu()
- [x] CRIT-02: analyze.py uses `except json.JSONDecodeError:` in read_job_status()
- [x] CRIT-03: batch.py uses `except json.JSONDecodeError:` in read_job_status()
- [x] CRIT-04: files.py uses `except Exception as e:` with logging
- [x] CRIT-05: ast_inference.py already had no bare excepts (FileNotFoundError handling)
- [x] CRIT-06: ast_training.py uses `except OSError:` for hardlink fallback
- [x] All modified files have logging imports and logger instances
- [x] `grep -rn "except:" backend/app/` returns no bare except clauses

## Verification

```bash
# Verify no bare except clauses remain
grep -rn "except:" backend/app/ --include="*.py"
# Returns: No matches found
```

## Deviations from Plan

### Additional Files Fixed

**[Rule 2 - Missing Critical]** Found and fixed 3 additional bare `except:` clauses not listed in the original plan:

1. `backend/app/services/analyze.py:66` - ID3 tag extraction
2. `backend/app/api/v1/csv_parser.py:70` - Duration calculation
3. `backend/app/services/training.py:514` - Duration reading

These were discovered during Task 3's comprehensive verification scan and fixed using the same pattern.

## Next Phase Readiness

All bare `except:` clauses have been eliminated. The application now:
- Allows Ctrl+C to terminate immediately (KeyboardInterrupt not caught)
- Logs specific error context for debugging
- Uses appropriate log levels (debug for expected failures, warning for potential issues)

No blockers for subsequent plans.

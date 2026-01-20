# Phase 1: Foundation Stability - Research

**Researched:** 2026-01-21
**Domain:** Python exception handling, path security, cross-platform compatibility, FastAPI error handling
**Confidence:** HIGH

## Summary

This phase addresses fundamental stability issues in the Filharmonia AI codebase: bare except clauses that swallow errors silently, hardcoded Windows paths that break cross-platform compatibility, and missing global error handling in the FastAPI application.

The research identified that all 10 bare except clauses should be replaced with specific exception types (typically `Exception` for catch-all logging, or more specific types like `json.JSONDecodeError`, `FileNotFoundError`). Path traversal prevention requires `Path.resolve()` combined with parent directory validation. Cross-platform paths should use `pathlib` exclusively with `tempfile.gettempdir()` for temporary directories. The global FastAPI exception handler should catch both `Exception` (for unexpected errors) and `StarletteHTTPException` (for HTTP errors from internal components).

**Primary recommendation:** Replace bare `except:` with `except Exception:` and add logging; use `Path.resolve()` with ancestry checks for path security; use `tempfile.gettempdir()` for temp directories; add `@app.exception_handler(Exception)` as catch-all.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pathlib | stdlib | Cross-platform path handling | Built-in, object-oriented, handles Windows/Unix differences automatically |
| tempfile | stdlib | Cross-platform temporary files | Built-in, respects OS conventions, secure random naming |
| logging | stdlib | Error logging | Built-in, configurable, standard across Python ecosystem |
| fastapi | 0.115.0 | API framework | Already in project, has robust exception handling |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| starlette.exceptions | (via fastapi) | HTTPException base class | For global exception handlers |
| typing | stdlib | Type hints | Return type annotations for clarity |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `os.path` | `pathlib` | pathlib is more Pythonic and handles cross-platform automatically |
| Custom temp dirs | `tempfile` | tempfile handles OS-specific locations and security |

**No installation required** - all tools are either stdlib or already in requirements.txt.

## Architecture Patterns

### Recommended Exception Handling Pattern
```python
# Before (BAD - bare except)
try:
    data = json.loads(file.read_text())
except:
    pass

# After (GOOD - specific exception with logging)
try:
    data = json.loads(file.read_text())
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse JSON from {file}: {e}")
    return None
except FileNotFoundError:
    logger.debug(f"File not found: {file}")
    return None
except Exception as e:
    logger.error(f"Unexpected error reading {file}: {e}")
    raise  # Re-raise unexpected errors
```

### Recommended Path Security Pattern
```python
from pathlib import Path

def safe_path_within_directory(user_input: str, allowed_root: Path) -> Path:
    """
    Validate that a path stays within allowed directory.

    Source: https://salvatoresecurity.com/preventing-directory-traversal-vulnerabilities-in-python/
    """
    # Resolve both to absolute, canonical paths
    root = allowed_root.resolve()
    requested = (root / user_input).resolve()

    # Check that requested path is within root
    if root not in requested.parents and requested != root:
        raise ValueError(f"Path traversal detected: {user_input}")

    return requested
```

### Recommended Global Exception Handler Pattern
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions from FastAPI and Starlette internals."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### Recommended Cross-Platform Temp Directory Pattern
```python
import tempfile
from pathlib import Path

# Instead of hardcoded /tmp/filharmonia_jobs
JOBS_DIR = Path(tempfile.gettempdir()) / "filharmonia_jobs"
JOBS_DIR.mkdir(exist_ok=True)
```

### Anti-Patterns to Avoid
- **Bare `except:`:** Catches SystemExit, KeyboardInterrupt, GeneratorExit - making apps hard to stop and debug
- **String-based path prefix matching:** `str(path).startswith("/allowed")` fails with paths like `/allowed-secrets`
- **Hardcoded temp paths:** `/tmp` is Unix-only; Windows uses `C:\Users\...\AppData\Local\Temp`
- **Hardcoded Windows paths:** `Y:\!_FILHARMONIA\...` breaks on any non-Windows system or different drive mapping

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-platform temp dir | `if os.name == 'nt': ... else: ...` | `tempfile.gettempdir()` | Handles TMPDIR/TEMP env vars, OS fallbacks |
| Path traversal check | String manipulation | `Path.resolve()` + ancestry check | Handles symlinks, `..`, encoded chars |
| Cross-platform path join | `f"{base}\\{file}"` | `Path(base) / file` | Handles separators automatically |
| Exception logging | `print(e)` | `logger.error(msg, exc_info=True)` | Captures stack trace, configurable output |

**Key insight:** Path and tempfile handle edge cases (symlinks, encoding, env vars, OS differences) that custom string manipulation misses.

## Common Pitfalls

### Pitfall 1: Bare Except Swallowing Ctrl+C
**What goes wrong:** `except:` catches `KeyboardInterrupt` - users cannot stop runaway processes
**Why it happens:** Developers want "catch everything" but don't realize what "everything" includes
**How to avoid:** Use `except Exception:` which excludes BaseException subclasses (SystemExit, KeyboardInterrupt, GeneratorExit)
**Warning signs:** App doesn't respond to Ctrl+C, hangs indefinitely on errors

### Pitfall 2: Path Traversal via Symlinks
**What goes wrong:** Checking `..` in string but not following symlinks allows `/allowed/link -> /etc/passwd`
**Why it happens:** String-based checks don't understand filesystem semantics
**How to avoid:** Always use `Path.resolve()` which canonicalizes paths including symlink resolution
**Warning signs:** Path checks that use `str.replace("..", "")` or similar

### Pitfall 3: StarletteHTTPException vs FastAPI HTTPException
**What goes wrong:** Global handler for FastAPI's HTTPException misses errors from Starlette internals
**Why it happens:** FastAPI's HTTPException inherits from Starlette's but they're different classes
**How to avoid:** Register handler for `starlette.exceptions.HTTPException`, not `fastapi.HTTPException`
**Warning signs:** Some 404s return JSON error, others return default HTML

### Pitfall 4: Windows vs Unix Path Separators
**What goes wrong:** Code like `path.split('/')` fails on Windows paths with `\`
**Why it happens:** String-based path manipulation assumes Unix conventions
**How to avoid:** Use `pathlib.Path` which handles separators automatically
**Warning signs:** Paths in frontend have `\\` (Windows) that backend can't parse

### Pitfall 5: Frontend Hardcoded Backend Paths
**What goes wrong:** Frontend constructs paths like `Y:\\!_FILHARMONIA\\...` assuming specific server config
**Why it happens:** Quick hack during development, never refactored
**How to avoid:** Backend returns full paths via API; frontend never constructs filesystem paths
**Warning signs:** Path construction logic duplicated between frontend and backend

## Code Examples

Verified patterns from official sources:

### Exception Hierarchy Awareness
```python
# Source: https://docs.python.org/3/library/exceptions.html
# BaseException
#  +-- SystemExit
#  +-- KeyboardInterrupt
#  +-- GeneratorExit
#  +-- Exception (everything else)
#       +-- StopIteration
#       +-- OSError (includes FileNotFoundError, etc.)
#       +-- ValueError
#       +-- TypeError
#       +-- json.JSONDecodeError
#       ...

# WRONG - catches KeyboardInterrupt
try:
    process_data()
except:
    pass

# WRONG - equivalent to above
try:
    process_data()
except BaseException:
    pass

# CORRECT - allows Ctrl+C to work
try:
    process_data()
except Exception:
    pass

# BEST - catch specific exceptions when you know what to expect
try:
    data = json.loads(text)
except json.JSONDecodeError:
    return default_value
```

### Path Traversal Prevention (Production Pattern)
```python
# Source: https://salvatoresecurity.com/preventing-directory-traversal-vulnerabilities-in-python/
from pathlib import Path

def validate_path_within_root(user_path: str, root_dir: Path) -> Path:
    """
    Securely validate that user-provided path stays within root directory.

    Handles:
    - Relative paths with ..
    - Absolute paths
    - Symlinks
    - URL-encoded characters (handled by caller before this function)
    """
    root = root_dir.resolve()

    # Join user path with root and resolve to canonical form
    requested = (root / user_path).resolve()

    # Check ancestry using Path.parents (more reliable than string prefix)
    if root not in requested.parents and requested != root:
        raise ValueError(f"Path outside allowed directory: {user_path}")

    if not requested.exists():
        raise FileNotFoundError(f"Path does not exist: {requested}")

    return requested
```

### FastAPI Global Exception Handler (Complete)
```python
# Source: https://fastapi.tiangolo.com/tutorial/handling-errors/
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle all HTTP exceptions consistently."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with useful details."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "type": "validation_error"
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unexpected errors.
    Logs full traceback but returns generic message to client.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )
```

### Cross-Platform Temp Directory
```python
# Source: https://docs.python.org/3/library/tempfile.html
import tempfile
from pathlib import Path

# Returns OS-appropriate temp directory:
# - Unix: /tmp or TMPDIR env var
# - Windows: C:\Users\...\AppData\Local\Temp or TEMP env var
TEMP_BASE = Path(tempfile.gettempdir())

# Create app-specific subdirectory
JOBS_DIR = TEMP_BASE / "filharmonia_jobs"
JOBS_DIR.mkdir(exist_ok=True)
```

### Type Hints for Functions
```python
# Examples for csv_parser.py functions that need return types
from typing import Optional

def get_duration(start: str, stop: str) -> str:
    """Calculate duration as M'S" format."""
    ...

def get_autosave_path(original_path: str) -> str:
    """Get autosave path for a given CSV path."""
    ...

def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to total seconds."""
    ...

def seconds_to_time(seconds: int) -> str:
    """Convert total seconds to HH:MM:SS."""
    ...

def escape_csv_field(field: str) -> str:
    """Escape CSV field - quote if contains comma, quote, or newline."""
    ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `os.path.join()` | `pathlib.Path / child` | Python 3.4+ | Cleaner syntax, cross-platform by default |
| `/tmp` hardcoded | `tempfile.gettempdir()` | Always recommended | Works on Windows/Mac/Linux |
| `except:` bare | `except Exception:` | PEP 760 discussion (2024) | Better debugging, Ctrl+C works |
| Print errors | `logging` module | Always recommended | Configurable, includes stack traces |

**Deprecated/outdated:**
- `os.path` string manipulation: Use `pathlib` instead
- Print-based error handling: Use `logging` module
- PEP 760 (ban bare except): Withdrawn, but best practice remains to avoid them

## Open Questions

Things that couldn't be fully resolved:

1. **MP3 path resolution API design**
   - What we know: Frontend needs to get MP3 path from CSV path
   - What's unclear: Should this be a dedicated endpoint or part of CSV metadata?
   - Recommendation: Create `/api/v1/files/mp3-for-csv?csv_path=...` endpoint that returns the full path

2. **Audio backend validation scope**
   - What we know: librosa falls back to audioread if soundfile fails
   - What's unclear: Should startup fail if ffmpeg missing, or just warn?
   - Recommendation: Log warning at startup if MP3 support is via slower audioread fallback

3. **PyTorch version pinning strategy**
   - What we know: Current requirements.txt has `torch==2.5.1+cu121`
   - What's unclear: Should we support multiple CUDA versions?
   - Recommendation: Document the CUDA requirement; single CUDA target is simpler

## Sources

### Primary (HIGH confidence)
- [FastAPI Official Docs - Error Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/) - Exception handler patterns
- [Python tempfile docs](https://docs.python.org/3/library/tempfile.html) - Cross-platform temp directories
- [Python pathlib docs](https://docs.python.org/3/library/pathlib.html) - Cross-platform path handling

### Secondary (MEDIUM confidence)
- [Salvatore Security - Path Traversal Prevention](https://salvatoresecurity.com/preventing-directory-traversal-vulnerabilities-in-python/) - Path security patterns
- [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal) - Security background
- [Real Python - pathlib](https://realpython.com/python-pathlib/) - Cross-platform best practices
- [Better Stack - FastAPI Error Handling](https://betterstack.com/community/guides/scaling-python/error-handling-fastapi/) - Production patterns

### Tertiary (LOW confidence - for awareness)
- [PEP 760 Discussion](https://peps.python.org/pep-0760/) - Bare except deprecation proposal (withdrawn)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) - CUDA compatibility reference

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib or already in project
- Architecture patterns: HIGH - Based on official FastAPI/Python documentation
- Path security: HIGH - Well-documented security best practice
- Exception handling: HIGH - Based on PEP discussions and linter rules
- Pitfalls: MEDIUM - Based on community articles, not official docs

**Research date:** 2026-01-21
**Valid until:** 90 days (stable patterns, not fast-moving)

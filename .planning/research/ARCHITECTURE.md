# Architecture Patterns for Stable ML Desktop Applications

**Project:** Filharmonia AI
**Focus:** Stability and maintainability patterns for existing React + FastAPI + PyTorch stack
**Researched:** 2026-01-20
**Overall Confidence:** HIGH (patterns verified against FastAPI and React official docs + industry practices)

## Executive Summary

This research addresses architecture patterns for stabilizing an existing ML desktop application with specific problems: bare except blocks, file-based job tracking, 1000+ line components, and hardcoded Windows paths. The recommendations prioritize **incremental brownfield refactoring** over rewrites, using established patterns that work with the existing FastAPI + React stack.

Key insight: Most stability issues stem from **missing abstraction layers** rather than wrong technology choices. The core architecture (React SPA + FastAPI + subprocess workers) is sound for this use case. The fix is adding proper error handling, configuration, and component boundaries.

---

## 1. Error Handling Patterns for ML Applications

### Current Problem

The codebase has 11+ bare `except:` blocks that silently swallow errors:

```python
# Current anti-pattern (backend/app/api/v1/analyze.py:35-38)
try:
    return json.loads(job_file.read_text())
except:
    pass
```

This makes debugging impossible and hides real failures.

### Pattern 1A: Specific Exception Handling with Logging

**Recommendation:** Replace bare except with specific exceptions and structured logging.

```python
import logging
from json import JSONDecodeError

logger = logging.getLogger(__name__)

def read_job_status(job_id: str) -> dict | None:
    job_file = get_job_file(job_id)
    if not job_file.exists():
        return None

    try:
        return json.loads(job_file.read_text())
    except JSONDecodeError as e:
        logger.warning(f"Corrupted job file {job_id}: {e}")
        return None
    except PermissionError as e:
        logger.error(f"Cannot read job file {job_id}: {e}")
        raise  # Re-raise - this is a real problem
```

**When to catch all:** Only at the top-level entry point (FastAPI exception handlers) to prevent server crashes. Never in business logic.

**Source:** [Miguel Grinberg - Ultimate Guide to Error Handling in Python](https://blog.miguelgrinberg.com/post/the-ultimate-guide-to-error-handling-in-python)

### Pattern 1B: FastAPI Global Exception Handler

**Recommendation:** Add a global exception handler for unexpected errors.

```python
# backend/app/main.py
from fastapi import Request
from fastapi.responses import JSONResponse
import logging
import traceback

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full traceback for debugging
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")

    # Return generic error to client (don't leak internals)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_id": str(uuid.uuid4())}
    )
```

This catches anything that slips through, logs it for debugging, but returns safe responses to clients.

### Pattern 1C: ML-Specific Error Categories

**Recommendation:** Define error categories specific to ML operations.

```python
# backend/app/errors.py
class FilharmoniaError(Exception):
    """Base class for application errors"""
    pass

class ModelNotFoundError(FilharmoniaError):
    """Raised when the ML model file doesn't exist"""
    pass

class ModelLoadError(FilharmoniaError):
    """Raised when model exists but cannot be loaded"""
    pass

class AudioProcessingError(FilharmoniaError):
    """Raised when audio file cannot be processed"""
    pass

class AnalysisInterrupted(FilharmoniaError):
    """Raised when analysis is cancelled by user"""
    pass
```

Then use these in services:

```python
def load_model(self, model_path: Path = None):
    if not model_path.exists():
        raise ModelNotFoundError(f"Model not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise ModelLoadError(f"Failed to load model weights: {e}") from e
```

**Confidence:** HIGH - standard Python practice, verified against [KDnuggets Advanced Error Handling](https://www.kdnuggets.com/advanced-error-handling-in-python-beyond-try-except)

---

## 2. Job/Task Management Patterns

### Current Problem

Jobs are tracked via JSON files in `/tmp/filharmonia_jobs/`:
- No cleanup mechanism (files accumulate indefinitely)
- Lost on reboot (in-memory cache `_jobs` dict is primary)
- Race conditions possible when reading/writing files
- No way to query job history

### Pattern 2A: SQLite Job Registry

**Recommendation:** Replace file-based tracking with SQLite. This is the right complexity level for a desktop app - no Redis/Celery needed.

```python
# backend/app/services/job_registry.py
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

Base = declarative_base()

class Job(Base):
    __tablename__ = 'jobs'

    id = Column(String, primary_key=True)
    type = Column(String)  # 'single' or 'batch'
    status = Column(String)  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    progress = Column(Float, default=0.0)
    current_file = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)

class JobRegistry:
    def __init__(self, db_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_job(self, job_id: str, job_type: str) -> Job:
        session = self.Session()
        job = Job(id=job_id, type=job_type, status='pending')
        session.add(job)
        session.commit()
        return job

    def update_progress(self, job_id: str, progress: float, current_file: str = None):
        session = self.Session()
        job = session.query(Job).filter_by(id=job_id).first()
        if job:
            job.progress = progress
            job.current_file = current_file
            job.status = 'running'
            session.commit()

    def cleanup_old_jobs(self, days: int = 7):
        """Remove jobs older than N days"""
        session = self.Session()
        cutoff = datetime.utcnow() - timedelta(days=days)
        session.query(Job).filter(Job.created_at < cutoff).delete()
        session.commit()
```

**Why SQLite over Redis/Celery:**
- Desktop app, not distributed system
- Persists across reboots
- No external dependencies
- Still queryable (job history, cleanup)
- Thread-safe with `check_same_thread=False`

**Source:** [FastAPI SQL Databases Tutorial](https://fastapi.tiangolo.com/tutorial/sql-databases/)

### Pattern 2B: Worker Process Communication

**Current:** Subprocess writes to JSON file, main process polls file.

**Recommendation:** Keep subprocess architecture (good isolation for heavy ML work) but improve IPC:

```python
# Option A: Keep file-based but add locking
import fcntl  # Unix only, use msvcrt on Windows

def write_job_status(job_id: str, status: dict):
    job_file = get_job_file(job_id)
    with open(job_file, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(status, f)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# Option B: Use multiprocessing.Queue (better)
from multiprocessing import Process, Queue

def run_analysis_worker(job_id: str, mp3_path: str, progress_queue: Queue):
    # Worker sends progress updates to queue
    progress_queue.put({'job_id': job_id, 'progress': 50.0})

# Main process consumes queue and updates SQLite
```

**Confidence:** HIGH - subprocess isolation is correct pattern for ML work, verified in [Leapcell FastAPI Background Tasks guide](https://leapcell.io/blog/managing-background-tasks-and-long-running-operations-in-fastapi)

### Pattern 2C: Startup Cleanup

**Recommendation:** Clean stale jobs on server startup.

```python
# backend/app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    job_registry = get_job_registry()

    # Mark any "running" jobs as interrupted (server crashed)
    job_registry.mark_interrupted_jobs()

    # Clean jobs older than 7 days
    job_registry.cleanup_old_jobs(days=7)

    yield

    # Shutdown
    job_registry.mark_running_as_interrupted()
```

---

## 3. Component Structure Patterns for Complex UIs

### Current Problem

`CsvViewer.tsx` is 1268 lines with 20+ `useState` hooks. It handles:
- CSV file listing and selection
- Track editing and merging
- Audio playback
- Autosave
- Export functionality
- Polling for analysis status
- Error/success toasts

This violates single responsibility and makes testing impossible.

### Pattern 3A: Extract Custom Hooks

**Recommendation:** Extract state logic into domain-specific hooks.

```typescript
// frontend/src/hooks/useTrackEditor.ts
export function useTrackEditor(initialTracks: Track[]) {
  const [tracks, setTracks] = useState<Track[]>(initialTracks);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const updateTrack = useCallback((id: string, updates: Partial<Track>) => {
    setTracks(prev => prev.map(t => t.id === id ? {...t, ...updates} : t));
    setHasUnsavedChanges(true);
  }, []);

  const mergeTracks = useCallback((trackIds: string[]) => {
    // Merge logic
    setHasUnsavedChanges(true);
  }, []);

  const splitTrack = useCallback((trackId: string, splitTime: string) => {
    // Split logic
    setHasUnsavedChanges(true);
  }, []);

  return { tracks, hasUnsavedChanges, updateTrack, mergeTracks, splitTrack };
}

// frontend/src/hooks/useAudioPlayer.ts
export function useAudioPlayer(mp3Path: string) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const play = useCallback(() => audioRef.current?.play(), []);
  const pause = useCallback(() => audioRef.current?.pause(), []);
  const seekTo = useCallback((time: number) => {
    if (audioRef.current) audioRef.current.currentTime = time;
  }, []);

  return { audioRef, isPlaying, currentTime, play, pause, seekTo };
}

// frontend/src/hooks/useAutosave.ts
export function useAutosave(csvPath: string, tracks: Track[], hasChanges: boolean) {
  const [lastSave, setLastSave] = useState<Date | null>(null);

  useEffect(() => {
    if (!hasChanges) return;

    const timer = setTimeout(async () => {
      await axios.post('/api/v1/csv/autosave', { path: csvPath, tracks });
      setLastSave(new Date());
    }, 2000); // Debounce 2 seconds

    return () => clearTimeout(timer);
  }, [csvPath, tracks, hasChanges]);

  return { lastSave };
}
```

**Source:** [React.dev - Reusing Logic with Custom Hooks](https://react.dev/learn/reusing-logic-with-custom-hooks), [CodeScene - Refactoring Components with Custom Hooks](https://codescene.com/blog/refactoring-components-in-react-with-custom-hooks)

### Pattern 3B: Component Composition

**Recommendation:** Split into smaller presentational components.

```
CsvViewer/
  index.tsx           # Container: orchestrates hooks, passes props down
  CsvFileList.tsx     # Presentational: renders file list
  TrackTable.tsx      # Presentational: renders track grid
  TrackRow.tsx        # Presentational: single track row
  AudioControls.tsx   # Presentational: play/pause/seek
  ExportModal.tsx     # Presentational: export dialog
```

Container component becomes thin:

```typescript
// frontend/src/pages/CsvViewer/index.tsx
export default function CsvViewer({ onBack, initialCsv }: Props) {
  const { csvFiles, selectedCsv, loadCsv } = useCsvFiles(initialCsv);
  const { tracks, hasUnsavedChanges, updateTrack, mergeTracks } = useTrackEditor([]);
  const { audioRef, isPlaying, play, pause, seekTo } = useAudioPlayer(mp3Path);
  const { lastSave } = useAutosave(selectedCsv, tracks, hasUnsavedChanges);

  return (
    <div className="csv-viewer">
      <CsvFileList files={csvFiles} selected={selectedCsv} onSelect={loadCsv} />
      <TrackTable tracks={tracks} onUpdate={updateTrack} onMerge={mergeTracks} />
      <AudioControls isPlaying={isPlaying} onPlay={play} onPause={pause} />
      {lastSave && <span>Last saved: {lastSave.toLocaleTimeString()}</span>}
    </div>
  );
}
```

**Confidence:** HIGH - standard React patterns, verified in [Martin Fowler - Modularizing React Apps](https://martinfowler.com/articles/modularizing-react-apps.html)

### Pattern 3C: Code Smell Threshold

Rule of thumb: **More than 5 lines before `return` is a code smell.**

When you see:
- 10+ useState hooks in one component
- 5+ useEffect hooks
- Component > 300 lines

It's time to extract hooks or split components.

---

## 4. Configuration Patterns for Cross-Platform

### Current Problem

- Hardcoded `/tmp/filharmonia_jobs/` (doesn't exist on Windows by default)
- Windows backslash paths in frontend string manipulation
- GPU detection returns `False` on bare except

### Pattern 4A: Centralized Path Configuration

**Recommendation:** Extend existing `Settings` class with cross-platform paths.

```python
# backend/app/config.py
from pathlib import Path
import tempfile
import platform

class Settings:
    # ... existing settings ...

    @property
    def JOBS_DIR(self) -> Path:
        """Cross-platform temporary directory for job files"""
        base = Path(tempfile.gettempdir()) / "filharmonia"
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def DATA_DIR(self) -> Path:
        """User data directory (config, cache, etc.)"""
        if platform.system() == "Windows":
            base = Path.home() / "AppData" / "Local" / "Filharmonia"
        elif platform.system() == "Darwin":  # macOS
            base = Path.home() / "Library" / "Application Support" / "Filharmonia"
        else:  # Linux
            base = Path.home() / ".local" / "share" / "filharmonia"
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def DB_PATH(self) -> Path:
        """SQLite database path"""
        return self.DATA_DIR / "filharmonia.db"
```

**Key insight:** Use `tempfile.gettempdir()` not hardcoded `/tmp/`. It returns correct path on all platforms:
- Linux: `/tmp`
- macOS: `/var/folders/.../T/`
- Windows: `C:\Users\...\AppData\Local\Temp`

**Source:** [DNMTechs - Cross-Platform Temp Directory in Python](https://dnmtechs.com/exploring-cross-platform-methods-for-retrieving-the-temp-directory-in-python-3/)

### Pattern 4B: Frontend Path Handling

**Recommendation:** Never construct file paths in frontend. Always get from API.

```typescript
// BAD - current pattern (CsvViewer.tsx:206-207)
const mp3Stem = csvName.match(/predictions_([^_]+)_/)?.[1];
const mp3Path = `Y:\\!_FILHARMONIA\\SORTED\\${year}\\${month}\\${day}\\${mp3Stem}.MP3`;

// GOOD - ask backend to resolve
const response = await axios.get(`/api/v1/files/mp3-for-csv?csv_path=${csvPath}`);
const mp3Path = response.data.mp3_path;
```

Backend API:

```python
@router.get("/mp3-for-csv")
async def get_mp3_for_csv(csv_path: str):
    """Resolve MP3 path from CSV path - handles all path logic server-side"""
    csv_file = Path(csv_path)
    # ... resolution logic ...
    return {"mp3_path": str(mp3_path)}
```

**Confidence:** HIGH - pathlib is the modern standard, verified in [Agent Factory - Cross-Platform Path Handling with pathlib](https://agentfactory.panaversity.org/docs/Python-Fundamentals/io-file-handling/pathlib)

### Pattern 4C: GPU Detection with Fallback

**Recommendation:** Robust GPU detection with specific error handling.

```python
def detect_compute_device() -> dict:
    """Detect available compute devices with detailed info"""
    result = {
        "device": "cpu",
        "cuda_available": False,
        "mps_available": False,  # Apple Silicon
        "cuda_device_name": None,
        "error": None
    }

    try:
        import torch

        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["device"] = "cuda"
            result["cuda_device_name"] = torch.cuda.get_device_name(0)

        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result["mps_available"] = True
            result["device"] = "mps"

    except ImportError as e:
        result["error"] = f"PyTorch not installed: {e}"
    except RuntimeError as e:
        # CUDA driver issues, etc.
        result["error"] = f"GPU detection failed: {e}"

    return result
```

**Note:** ROCm (AMD) support is emerging but not yet stable. For 2025-2026, recommend CUDA > MPS > CPU fallback.

**Source:** [Hugging Face - PyTorch Install Guide](https://huggingface.co/blog/daya-shankar/pytorch-install-guide)

---

## 5. Incremental Refactoring Strategy

### Pattern 5A: Strangler Fig for Brownfield

**Recommendation:** Don't rewrite - incrementally wrap and replace.

```
Phase 1: Add Abstraction Layer
  - Create JobRegistry interface
  - Existing code continues to use file-based implementation

Phase 2: Implement New Version
  - Create SQLite implementation of JobRegistry
  - Both implementations coexist

Phase 3: Migrate Reads
  - New code reads from SQLite
  - Old code still writes to files
  - Sync process copies file -> SQLite

Phase 4: Migrate Writes
  - New code writes to SQLite
  - Old code deprecated

Phase 5: Remove Old Code
  - Delete file-based implementation
```

**Key principle:** Never stop delivery. Each phase should be deployable.

**Source:** [Shopify Engineering - Refactoring Legacy Code with Strangler Fig Pattern](https://shopify.engineering/refactoring-legacy-code-strangler-fig-pattern)

### Pattern 5B: Feature Flags for Gradual Rollout

```python
# backend/app/config.py
class Settings:
    # Feature flags for gradual migration
    USE_SQLITE_JOBS: bool = os.getenv("USE_SQLITE_JOBS", "false").lower() == "true"
    USE_NEW_ERROR_HANDLING: bool = os.getenv("USE_NEW_ERROR_HANDLING", "true").lower() == "true"

# Usage
if settings.USE_SQLITE_JOBS:
    registry = SQLiteJobRegistry(settings.DB_PATH)
else:
    registry = FileBasedJobRegistry(settings.JOBS_DIR)
```

### Pattern 5C: Refactoring Order

**Recommended sequence based on risk/impact:**

1. **Error Handling** (Low risk, high value)
   - Replace bare except blocks
   - Add logging
   - Add global exception handler
   - *No behavior change, just better observability*

2. **Configuration** (Medium risk, high value)
   - Centralize path handling
   - Remove hardcoded paths
   - Use tempfile.gettempdir()
   - *Cross-platform fixes*

3. **Job Registry** (Medium risk, medium value)
   - Add SQLite alongside files
   - Migrate gradually
   - Add cleanup
   - *Stability improvement*

4. **Frontend Components** (Low risk, medium value)
   - Extract hooks from CsvViewer
   - Split into smaller components
   - *Maintainability improvement*

---

## Anti-Patterns to Avoid

### Don't Over-Engineer

**Bad:** "Let's add Celery, Redis, Kafka, and Kubernetes for job management!"

**Good:** SQLite is sufficient for a single-user desktop app. Add complexity only when you hit actual scaling limits.

### Don't Catch and Ignore

**Bad:**
```python
try:
    do_something()
except:
    pass  # silently ignore all errors
```

**Good:**
```python
try:
    do_something()
except SpecificError as e:
    logger.warning(f"Expected error handled: {e}")
    return fallback_value
```

### Don't Rewrite from Scratch

**Bad:** "This component is messy, let me rewrite it completely."

**Good:** Extract one hook at a time. Each extraction is testable and deployable.

### Don't Mix Path Manipulation Styles

**Bad:**
```python
path = base_dir + "/" + subdir + "\\" + filename  # Mixed separators
```

**Good:**
```python
path = Path(base_dir) / subdir / filename  # pathlib handles separators
```

---

## Summary: What to Implement

| Area | Pattern | Priority | Effort |
|------|---------|----------|--------|
| Error Handling | Specific exceptions + logging | HIGH | Low |
| Error Handling | Global exception handler | HIGH | Low |
| Error Handling | Custom exception classes | MEDIUM | Low |
| Job Tracking | SQLite registry | MEDIUM | Medium |
| Job Tracking | Startup cleanup | HIGH | Low |
| Configuration | Centralize paths | HIGH | Low |
| Configuration | tempfile.gettempdir() | HIGH | Low |
| Configuration | Backend path resolution | HIGH | Low |
| Frontend | Extract useTrackEditor hook | MEDIUM | Medium |
| Frontend | Extract useAudioPlayer hook | MEDIUM | Low |
| Frontend | Split CsvViewer components | LOW | Medium |

**Recommended first milestone:** Error handling + configuration (cross-platform paths). These have highest value with lowest risk.

---

## Sources

### Error Handling
- [Miguel Grinberg - The Ultimate Guide to Error Handling in Python](https://blog.miguelgrinberg.com/post/the-ultimate-guide-to-error-handling-in-python) - HIGH confidence
- [KDnuggets - Advanced Error Handling in Python](https://www.kdnuggets.com/advanced-error-handling-in-python-beyond-try-except) - HIGH confidence
- [Qodo - 6 Best Practices for Python Exception Handling](https://www.qodo.ai/blog/6-best-practices-for-python-exception-handling/) - MEDIUM confidence

### Job Management
- [FastAPI - Background Tasks Tutorial](https://fastapi.tiangolo.com/tutorial/background-tasks/) - HIGH confidence (official docs)
- [FastAPI - SQL Databases Tutorial](https://fastapi.tiangolo.com/tutorial/sql-databases/) - HIGH confidence (official docs)
- [Leapcell - Managing Background Tasks in FastAPI](https://leapcell.io/blog/managing-background-tasks-and-long-running-operations-in-fastapi) - MEDIUM confidence
- [Medium - Practical Background Processing with FastAPI](https://blog.greeden.me/en/2025/12/02/practical-background-processing-with-fastapi-a-job-queue-design-guide-with-backgroundtasks-and-celery/) - MEDIUM confidence

### React Patterns
- [React.dev - Reusing Logic with Custom Hooks](https://react.dev/learn/reusing-logic-with-custom-hooks) - HIGH confidence (official docs)
- [Martin Fowler - Modularizing React Applications](https://martinfowler.com/articles/modularizing-react-apps.html) - HIGH confidence
- [CodeScene - Refactoring Components in React with Custom Hooks](https://codescene.com/blog/refactoring-components-in-react-with-custom-hooks) - MEDIUM confidence
- [Alex Kondov - Common Sense Refactoring of a Messy React Component](https://alexkondov.com/refactoring-a-messy-react-component/) - MEDIUM confidence

### Cross-Platform
- [Agent Factory - Cross-Platform Path Handling with pathlib](https://agentfactory.panaversity.org/docs/Python-Fundamentals/io-file-handling/pathlib) - HIGH confidence
- [DNMTechs - Cross-Platform Temp Directory in Python](https://dnmtechs.com/exploring-cross-platform-methods-for-retrieving-the-temp-directory-in-python-3/) - MEDIUM confidence
- [Hugging Face - PyTorch Install Guide](https://huggingface.co/blog/daya-shankar/pytorch-install-guide) - HIGH confidence

### Refactoring Strategy
- [Shopify Engineering - Refactoring Legacy Code with Strangler Fig Pattern](https://shopify.engineering/refactoring-legacy-code-strangler-fig-pattern) - HIGH confidence
- [AWS - Strangler Fig Pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/strangler-fig.html) - HIGH confidence
- [GoCodeo - How the Strangler Fig Pattern Enables Safe Refactoring](https://www.gocodeo.com/post/how-the-strangler-fig-pattern-enables-safe-and-gradual-refactoring) - MEDIUM confidence

---

*Research completed: 2026-01-20*

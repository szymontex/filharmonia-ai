# Detailed Code Audit: Filharmonia AI

**Date:** 2026-01-20
**Auditor:** Deep code analysis
**Files reviewed:** 15+ source files

---

## CRITICAL: Security & Crashes

### 1. Path Traversal Vulnerability

**Location:** `backend/app/api/v1/csv_parser.py:181`

```python
# CURRENT (VULNERABLE):
csv_path = Path(path)
if not csv_path.exists():
    return ParseResponse(tracks=[], total_segments=0)
df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)
```

**Attack:** User sends `path=../../../etc/passwd` → reads arbitrary system files

**FIX:**
```python
from app.config import settings

csv_path = Path(path).resolve()
allowed_dirs = [settings.SORTED_FOLDER, settings.TRAINING_DATA_FOLDER]

if not any(csv_path.is_relative_to(d) for d in allowed_dirs):
    raise HTTPException(status_code=403, detail="Access denied: path outside allowed directories")

if not csv_path.exists():
    return ParseResponse(tracks=[], total_segments=0)
```

**Same issue in:**
- `files.py:104` - `/api/v1/files/browse`
- `waveform.py` - `/api/v1/waveform`
- `audio.py` - `/api/v1/audio/stream`
- `export.py` - `/api/v1/export/segment`

---

### 2. Bare Except Swallowing All Errors

**Location:** `backend/app/api/v1/batch.py:37-40`

```python
# CURRENT (BAD):
def read_job_status(job_id: str) -> dict:
    job_file = get_job_file(job_id)
    if job_file.exists():
        try:
            return json.loads(job_file.read_text())
        except:  # ← CATCHES EVERYTHING INCLUDING KeyboardInterrupt!
            pass
    return None
```

**Problem:** If JSON is corrupted, you never know. Silent failure.

**FIX:**
```python
import logging
logger = logging.getLogger(__name__)

def read_job_status(job_id: str) -> dict | None:
    job_file = get_job_file(job_id)
    if not job_file.exists():
        return None
    try:
        return json.loads(job_file.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted job file {job_file}: {e}")
        # Optionally: backup corrupted file and delete
        job_file.rename(job_file.with_suffix('.json.corrupted'))
        return None
    except IOError as e:
        logger.error(f"Cannot read job file {job_file}: {e}")
        return None
```

**Same pattern in:**
- `analyze.py:34-38` - same function
- `main.py:84` - `torch.cuda.is_available()` check
- `ast_inference.py:189-190` - model load
- `batch.py:350` - `derive_mp3_path_from_csv`
- `csv_parser.py:67` - duration calculation

---

### 3. Memory Leak: In-Memory Job Dict Never Cleaned

**Location:** `backend/app/api/v1/analyze.py:48` and `batch.py:48-49`

```python
# CURRENT (LEAKING):
_processes = {}
_single_jobs = {}  # ← GROWS FOREVER

@router.post("/", response_model=AnalyzeResponse)
async def analyze_file(request: AnalyzeRequest):
    job_id = str(uuid.uuid4())
    _single_jobs[job_id] = initial_status  # ← NEVER REMOVED
```

**Problem:** After 10,000 analyses, you have 10,000 entries in memory forever.

**FIX (Option A - TTL cleanup):**
```python
from datetime import datetime, timedelta
from collections import OrderedDict

MAX_JOBS = 1000
JOB_TTL = timedelta(hours=24)

_jobs: OrderedDict[str, dict] = OrderedDict()

def add_job(job_id: str, status: dict):
    # Add timestamp
    status['_created'] = datetime.now().isoformat()
    _jobs[job_id] = status

    # Cleanup old jobs
    cutoff = datetime.now() - JOB_TTL
    while _jobs:
        oldest_id, oldest = next(iter(_jobs.items()))
        if datetime.fromisoformat(oldest['_created']) < cutoff:
            _jobs.pop(oldest_id)
        else:
            break

    # Also enforce max size
    while len(_jobs) > MAX_JOBS:
        _jobs.popitem(last=False)
```

**FIX (Option B - SQLite, better):**
See INFRA-01 requirement.

---

### 4. Race Condition in Job Status

**Location:** `backend/app/api/v1/batch.py:138,164`

```python
# CURRENT (RACE CONDITION):
def is_cancelled():
    current_status = read_job_status(job_id)  # ← READ
    return current_status and current_status.get("cancelled", False)

# Meanwhile in progress_callback:
write_job_status(job_id, {...})  # ← WRITE

# If read happens DURING write → corrupted JSON!
```

**FIX (atomic writes):**
```python
import tempfile

def write_job_status(job_id: str, status: dict):
    job_file = get_job_file(job_id)

    # Write to temp file first
    fd, temp_path = tempfile.mkstemp(dir=job_file.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(status, f)
        # Atomic rename
        os.replace(temp_path, job_file)
    except:
        os.unlink(temp_path)
        raise
```

---

### 5. Zombie Processes

**Location:** `backend/app/api/v1/analyze.py:85-91`

```python
# CURRENT (ZOMBIE):
process = subprocess.Popen(
    [python_exe, str(WORKER_SCRIPT), job_id, str(mp3_path)],
    stdout=log,
    stderr=subprocess.STDOUT,
    cwd=str(Path(__file__).parent.parent.parent),
    start_new_session=True  # ← DETACHED, NEVER CLEANED UP
)
_processes[job_id] = process  # ← STORED BUT NEVER .wait()'ed
```

**Problem:** If worker crashes, process stays zombie. After 1000 crashes, OS runs out of PIDs.

**FIX:**
```python
import atexit
import signal

_processes: dict[str, subprocess.Popen] = {}

def cleanup_processes():
    for job_id, proc in list(_processes.items()):
        if proc.poll() is None:  # Still running
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        del _processes[job_id]

atexit.register(cleanup_processes)

# Also add periodic cleanup in background task
async def cleanup_finished_processes():
    while True:
        await asyncio.sleep(60)
        for job_id, proc in list(_processes.items()):
            if proc.poll() is not None:  # Finished
                del _processes[job_id]
```

---

### 6. Blocking I/O in Async Route

**Location:** `backend/app/api/v1/csv_parser.py:273-276`

```python
# CURRENT (BLOCKING):
@router.get("/check-autosave", response_model=AutosaveCheckResponse)
async def check_autosave(path: str = Query(...)):
    # ...
    if original_path.exists():
        with open(original_path, 'r', encoding='utf-8') as f1:
            original_content = f1.read()  # ← BLOCKS EVENT LOOP!
        with open(autosave_path, 'r', encoding='utf-8') as f2:
            autosave_content = f2.read()  # ← BLOCKS EVENT LOOP!
```

**Problem:** If files are large (100MB CSV), entire FastAPI server hangs for other users.

**FIX:**
```python
import asyncio
from functools import partial

async def check_autosave(path: str = Query(...)):
    # ...
    if original_path.exists():
        # Run blocking I/O in thread pool
        loop = asyncio.get_event_loop()
        original_content = await loop.run_in_executor(
            None, partial(original_path.read_text, encoding='utf-8')
        )
        autosave_content = await loop.run_in_executor(
            None, partial(autosave_path.read_text, encoding='utf-8')
        )
```

---

## HIGH: Performance Issues

### 7. CSV Double-Read (2x slower)

**Location:** `backend/app/api/v1/batch.py:335-341`

```python
# CURRENT (INEFFICIENT):
df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=1)  # ← READ #1

if 'model_version' not in df.columns:
    csv_model_version = "unknown"
else:
    full_df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # ← READ #2 (FULL!)
    csv_model_version = full_df['model_version'].iloc[0]
```

**Problem:** Reads header, then reads ENTIRE file again just to get first row.

**FIX:**
```python
# Read small sample (100 rows is enough to get model_version)
df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=100)

if 'model_version' not in df.columns:
    csv_model_version = "unknown"
else:
    csv_model_version = df['model_version'].iloc[0]
```

**Speedup:** 2x for large CSVs

---

### 8. pandas → Polars (5-30x faster)

**Location:** `backend/app/api/v1/csv_parser.py:187`

```python
# CURRENT (SLOW):
import pandas as pd
df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)
```

**FIX:**
```python
import polars as pl

df = pl.read_csv(
    csv_path,
    encoding='utf-8',
    quote_char='"',
    try_parse_dates=False
)

# Conversion patterns:
# pd.isna(value) → value is None
# df.iloc[i][col] → df.row(i)[col_index] or df[i, col]
# df[col].iloc[0] → df[col][0]
# for col in df.columns → same (df.columns works)
```

**Speedup:** 5-30x for CSV operations

---

### 9. Hardcoded Windows Path

**Location:** `frontend/src/pages/CsvViewer.tsx:206`

```typescript
// CURRENT (WINDOWS ONLY):
const match = cleanPath.match(/predictions_(.+?)_(\d{4})-(\d{2})-(\d{2})(?:_\d{2}-\d{2})?\.csv/)
if (match) {
  const [, songName, year, month, day] = match
  const mp3 = `Y:\\!_FILHARMONIA\\SORTED\\${year}\\${month}\\${day}\\${songName}.MP3`  // ← HARDCODED!
  setMp3Path(mp3)
}
```

**Problem:** Only works on this one Windows machine with Y: drive mounted.

**FIX (Backend API):**

Add endpoint:
```python
# backend/app/api/v1/files.py
@router.get("/mp3-for-csv")
async def get_mp3_for_csv(csv_path: str = Query(...)):
    """Derive MP3 path from CSV path"""
    csv_path = Path(csv_path)

    # Parse: predictions_{song}_{YYYY-MM-DD}.csv
    match = re.match(r'predictions_(.+?)_(\d{4})-(\d{2})-(\d{2})', csv_path.stem)
    if not match:
        raise HTTPException(404, "Cannot derive MP3 path from CSV name")

    song, year, month, day = match.groups()
    mp3_path = settings.SORTED_FOLDER / year / month / day / f"{song}.MP3"

    if not mp3_path.exists():
        raise HTTPException(404, f"MP3 not found: {mp3_path}")

    return {"mp3_path": str(mp3_path)}
```

Frontend:
```typescript
// FIXED:
const res = await axios.get(`/api/v1/files/mp3-for-csv?csv_path=${encodeURIComponent(csvPath)}`)
setMp3Path(res.data.mp3_path)
```

---

### 10. Polling Every 2s = 43,200 API calls/day

**Location:** `frontend/src/pages/CsvViewer.tsx:61-80`

```typescript
// CURRENT (WASTEFUL):
const interval = setInterval(async () => {
  const response = await axios.get('/api/v1/analyze/batch')
  // ...
}, 2000)  // ← EVERY 2 SECONDS, FOREVER
```

**Problem:**
- 30 calls/minute × 60 min × 24 hours = 43,200 calls/day
- Even when no jobs running!

**FIX (exponential backoff):**
```typescript
useEffect(() => {
  let timeout: NodeJS.Timeout
  let interval = 2000  // Start fast
  const maxInterval = 30000  // Slow down to 30s when idle

  const poll = async () => {
    try {
      const response = await axios.get('/api/v1/analyze/batch')
      const runningJobs = response.data.filter((job: any) => job.status === 'running')

      if (runningJobs.length > 0) {
        interval = 2000  // Fast polling when jobs running
      } else {
        interval = Math.min(interval * 1.5, maxInterval)  // Slow down when idle
      }

      // ... update state
    } catch (error) {
      interval = Math.min(interval * 2, maxInterval)  // Back off on errors
    }

    timeout = setTimeout(poll, interval)
  }

  poll()
  return () => clearTimeout(timeout)
}, [])
```

**Reduction:** 90%+ fewer API calls when idle

---

### 11. Time Parsing Crash

**Location:** `frontend/src/pages/CsvViewer.tsx:340`

```typescript
// CURRENT (CRASHES):
const timeSeconds = parseInt(timeStr.split(':')[0]) * 3600
  + parseInt(timeStr.split(':')[1]) * 60
  + parseInt(timeStr.split(':')[2])  // ← parseInt("45.5") = 45, but what about parseInt("abc")?
```

**Location:** `backend/app/api/v1/csv_parser.py:203-206`

```python
# CURRENT (ALSO CRASHES):
def time_to_seconds(time_str: str) -> int:
    parts = list(map(int, time_str.split(':')))  # ← int("45.5") crashes!
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
```

**FIX (backend):**
```python
def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS or HH:MM:SS.ms to seconds"""
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
```

**FIX (frontend):**
```typescript
function timeToSeconds(timeStr: string): number {
  const parts = timeStr.split(':')
  if (parts.length !== 3) {
    console.error(`Invalid time format: ${timeStr}`)
    return 0
  }
  return parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseFloat(parts[2])
}
```

---

### 12. Duplicated Time Calculation Logic

**Location:** `frontend/src/pages/CsvViewer.tsx:230-284`

```typescript
// updateStart - lines 230-256
const updateStart = (id: string, start: string) => {
  setTracks(prevTracks => {
    // ... 26 lines of time calculation logic
  })
}

// updateStop - lines 258-284
const updateStop = (id: string, stop: string) => {
  setTracks(prevTracks => {
    // ... SAME 26 lines of logic, copy-pasted!
  })
}
```

**Problem:** If you fix a bug in one, you might forget the other.

**FIX:**
```typescript
// Extract utility
function updateTrackBoundary(
  tracks: Track[],
  trackId: string,
  field: 'start' | 'stop',
  newValue: string
): Track[] {
  const trackIndex = tracks.findIndex(t => t.id === trackId)
  if (trackIndex === -1) return tracks

  const updatedTracks = [...tracks]
  const currentTrack = { ...updatedTracks[trackIndex], [field]: newValue }

  // Recalculate duration
  if (currentTrack.start && currentTrack.stop) {
    currentTrack.duration = calculateDuration(currentTrack.start, currentTrack.stop)
  }
  updatedTracks[trackIndex] = currentTrack

  // Update adjacent track
  if (field === 'start' && trackIndex > 0) {
    const prevTrack = { ...updatedTracks[trackIndex - 1], stop: newValue }
    prevTrack.duration = calculateDuration(prevTrack.start, prevTrack.stop)
    updatedTracks[trackIndex - 1] = prevTrack
  } else if (field === 'stop' && trackIndex < updatedTracks.length - 1) {
    const nextTrack = { ...updatedTracks[trackIndex + 1], start: newValue }
    nextTrack.duration = calculateDuration(nextTrack.start, nextTrack.stop)
    updatedTracks[trackIndex + 1] = nextTrack
  }

  return updatedTracks
}

// Usage:
const updateStart = (id: string, start: string) => {
  setTracks(prev => updateTrackBoundary(prev, id, 'start', start))
  setHasUnsavedChanges(true)
}

const updateStop = (id: string, stop: string) => {
  setTracks(prev => updateTrackBoundary(prev, id, 'stop', stop))
  setHasUnsavedChanges(true)
}
```

---

## HIGH: Component Architecture

### 13. CsvViewer.tsx is 1268 Lines

**Problem:**
- 26 useState hooks (lines 29-54)
- 30+ functions
- Mixed concerns (CSV loading, audio playback, track editing, export, autosave)
- Impossible to test
- Any change risks breaking something else

**Proposed Split:**

```
frontend/src/pages/CsvViewer/
├── index.tsx           (200 lines) - orchestration only
├── CsvSelector.tsx     (150 lines) - file list + selection
├── TrackTable.tsx      (300 lines) - table rendering
├── TrackRow.tsx        (100 lines) - single row
├── ExportModal.tsx     (100 lines) - export dialog
├── hooks/
│   ├── useTrackEditor.ts   (150 lines) - track CRUD operations
│   ├── useAudioPlayer.ts   (100 lines) - audio state
│   ├── useAutosave.ts      (80 lines)  - autosave logic
│   └── useCsvLoader.ts     (100 lines) - CSV loading
└── utils/
    └── timeUtils.ts        (50 lines)  - time calculations
```

**Hook extraction example:**

```typescript
// hooks/useTrackEditor.ts
export function useTrackEditor(initialTracks: Track[]) {
  const [tracks, setTracks] = useState<Track[]>(initialTracks)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

  const updateTrack = useCallback((id: string, updates: Partial<Track>) => {
    setTracks(prev => prev.map(t => t.id === id ? { ...t, ...updates } : t))
    setHasUnsavedChanges(true)
  }, [])

  const deleteTrack = useCallback((id: string) => {
    // ... merge with adjacent logic
    setHasUnsavedChanges(true)
  }, [])

  const updateBoundary = useCallback((id: string, field: 'start' | 'stop', value: string) => {
    setTracks(prev => updateTrackBoundary(prev, id, field, value))
    setHasUnsavedChanges(true)
  }, [])

  return {
    tracks,
    setTracks,
    hasUnsavedChanges,
    updateTrack,
    deleteTrack,
    updateBoundary,
    resetChanges: () => setHasUnsavedChanges(false)
  }
}
```

---

## MEDIUM: Type Safety

### 14. Missing Return Types

**Location:** `backend/app/api/v1/csv_parser.py:52-68`

```python
# CURRENT (NO TYPE HINTS):
def get_duration(start: str, stop: str):  # ← What does this return?
    try:
        # ...
        return f"{minutes}'{seconds}\""
    except:
        return "0'0\""  # ← Returns str, but None is possible?
```

**FIX:**
```python
def get_duration(start: str, stop: str) -> str:
    """
    Calculate duration as M'S" format.

    Args:
        start: Start time in HH:MM:SS format
        stop: Stop time in HH:MM:SS format

    Returns:
        Duration string like "12'30\""

    Raises:
        ValueError: If time format is invalid
    """
    try:
        start_parts = list(map(int, start.split(':')))
        stop_parts = list(map(int, stop.split(':')))

        if len(start_parts) != 3 or len(stop_parts) != 3:
            raise ValueError(f"Invalid time format: {start} or {stop}")

        start_sec = start_parts[0] * 3600 + start_parts[1] * 60 + start_parts[2]
        stop_sec = stop_parts[0] * 3600 + stop_parts[1] * 60 + stop_parts[2]

        diff_sec = stop_sec - start_sec
        if diff_sec < 0:
            raise ValueError(f"Stop time {stop} is before start time {start}")

        minutes = diff_sec // 60
        seconds = diff_sec % 60
        return f"{minutes}'{seconds}\""
    except (ValueError, IndexError) as e:
        raise ValueError(f"Cannot calculate duration: {e}")
```

---

## SUMMARY: Priority Order

**Do first (1-2 days):**
1. Path traversal fix (security)
2. Bare except → specific exceptions (debugging)
3. pandas → polars (instant 5-30x speedup)
4. Memory leak fix (stability)

**Do next (3-5 days):**
5. Hardcoded paths elimination
6. CSV double-read fix
7. Polling optimization
8. Time parsing fixes

**Do later (1-2 weeks):**
9. CsvViewer split
10. Hook extraction
11. Type hints
12. SQLite job registry

---

## MORE FINDINGS (Additional Files)

### 15. Another Hardcoded Windows Path

**Location:** `frontend/src/pages/CalendarBrowser.tsx:268`

```typescript
// CURRENT (WINDOWS ONLY):
const getCsvPath = (recording: Recording) => {
  const stem = recording.name.replace('.MP3', '').replace('.mp3', '')
  const date = recording.date
  return `Y:\\!_FILHARMONIA\\SORTED\\ANALYSIS_RESULTS\\predictions_${stem}_${date}.csv`  // ← HARDCODED!
}
```

**Same fix as CsvViewer** - use backend API for path resolution.

---

### 16. Dead Code: handlePlayRecording Does Nothing

**Location:** `frontend/src/pages/CalendarBrowser.tsx:107-110`

```typescript
// CURRENT (DEAD CODE):
const handlePlayRecording = (recording: Recording) => {
  // TODO: Open player/editor
  console.log('Play recording:', recording)  // ← Button exists but does nothing!
}
```

**Fix:** Either implement (open WaveformEditor) or remove the Play button from UI.

---

### 17. N+1 Query in Uncertainty Stats

**Location:** `backend/app/api/v1/uncertainty.py:281-293`

```python
# CURRENT (N+1 QUERY):
for csv_file in results_folder.glob("predictions_*.csv"):
    # Read CSV header (nrows=1)
    df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=1)  # ← READ #1

    if 'confidence' not in df.columns:
        continue

    # Read FULL CSV again
    full_df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # ← READ #2 (WHOLE FILE!)

    if 'model_version' not in full_df.columns:
        model_version = "unknown"
    else:
        model_version = full_df['model_version'].iloc[0]
```

**Problem:** For 500 CSVs, this reads 1000 files (500 headers + 500 full files).

**FIX:**
```python
for csv_file in results_folder.glob("predictions_*.csv"):
    try:
        # Read once with small sample (100 rows enough for headers + first values)
        df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=100)

        if 'confidence' not in df.columns:
            continue

        model_version = df.get('model_version', pd.Series(["unknown"])).iloc[0]

        # For uncertain count, need full file but only once
        if need_full_stats:
            full_df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)
            # ...
```

**Better FIX (Polars lazy scan):**
```python
import polars as pl

df = pl.scan_csv(csv_file).select(['confidence', 'model_version']).collect()
```

---

### 18. More Bare Except Blocks

**Location:** `backend/app/services/analyze.py:66`

```python
# CURRENT:
try:
    audiofile = eyed3.load(str(mp3_path))
    if audiofile and audiofile.tag and audiofile.tag.title:
        record_date = datetime.strptime(audiofile.tag.title, 'Untitled %m/%d/%Y %H:%M:%S')
        time_str = f"_{record_date.hour:02d}-{record_date.minute:02d}"
except:
    pass  # ← SWALLOWS EVERYTHING
```

**Location:** `backend/app/api/v1/files.py:49`

```python
# SAME PATTERN:
try:
    audiofile = eyed3.load(str(mp3_file))
    # ...
except:
    pass
```

**FIX:**
```python
except (eyed3.Error, ValueError, AttributeError) as e:
    logger.debug(f"Could not extract time from {mp3_path}: {e}")
    time_str = ""
```

---

### 19. Delete Endpoint Has No Path Validation

**Location:** `backend/app/api/v1/files.py:103-121`

```python
# CURRENT (VULNERABLE):
@router.delete("/delete-csv")
async def delete_csv(path: str = Query(...)):
    csv_path = Path(path)

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")

    csv_path.unlink()  # ← DELETES ANY FILE USER SPECIFIES!
```

**Attack:** `DELETE /api/v1/files/delete-csv?path=/etc/passwd` → deletes system files

**FIX:**
```python
@router.delete("/delete-csv")
async def delete_csv(path: str = Query(...)):
    csv_path = Path(path).resolve()

    # SECURITY: Only allow deletion in ANALYSIS_RESULTS folder
    allowed_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"
    if not csv_path.is_relative_to(allowed_folder):
        raise HTTPException(403, "Can only delete files in ANALYSIS_RESULTS folder")

    if not csv_path.exists():
        raise HTTPException(404, "CSV file not found")

    csv_path.unlink()
```

---

### 20. StickyPlayer.tsx - 500+ Lines of Custom Waveform Code

**Location:** `frontend/src/components/StickyPlayer.tsx`

```typescript
// Lines 174-266: Custom canvas drawing
// Lines 269-400+: Custom playhead, regions, markers, dragging
```

**Problem:**
- Reimplements features that exist in libraries
- Custom zoom handling (lines 86-141)
- Custom region drawing
- Custom marker dragging
- Hard to maintain, easy to introduce bugs

**FIX:** Migrate to wavesurfer.js v7:
```typescript
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/plugins/regions'

const wavesurfer = WaveSurfer.create({
  container: containerRef.current,
  waveColor: '#4b5563',
  progressColor: '#3b82f6',
  peaks: waveformData.data.map(p => [p.min, p.max]),
  duration: waveformData.duration,
  plugins: [
    RegionsPlugin.create({
      regions: tracks.map(t => ({
        start: timeToSeconds(t.start),
        end: timeToSeconds(t.stop),
        color: CLASS_COLORS[t.predicted_class].rgba,
        drag: true,
        resize: true,
      }))
    })
  ]
})

// Built-in: zoom, regions, markers, drag handles, keyboard shortcuts
```

**Benefits:**
- -400 lines of custom code
- Built-in accessibility
- Tested by thousands of users
- Timeline, minimap, hover plugins

---

### 21. print() Statements Instead of Logging

**Locations:** Multiple files

```python
# uncertainty.py:240
print(f"Error processing {csv_file}: {e}")

# batch.py:360
print(f"Error processing {csv_file}: {e}")

# ast_inference.py:59-62
print(f"[OK] AST model loaded: {model_path.name}")
print(f"  Device: {self.device}")
```

**Problem:** print() doesn't go to log files, no timestamps, no log levels

**FIX:**
```python
import logging
logger = logging.getLogger(__name__)

# Replace print with appropriate log level
logger.info(f"AST model loaded: {model_path.name}")
logger.debug(f"Device: {self.device}")
logger.error(f"Error processing {csv_file}: {e}")
```

---

### 22. Magic Numbers

**Location:** Multiple files

```python
# ast_inference.py:29-33
self.mel_transform = T.MelSpectrogram(
    sample_rate=settings.SAMPLE_RATE,
    n_fft=2048,      # ← Magic
    hop_length=512,   # ← Magic
    n_mels=128        # ← Magic
)

# ast_inference.py:98-104
if logmel.shape[0] < 1024:  # ← Magic
    pad_width = 1024 - logmel.shape[0]  # ← Magic

# analyze.py:85
BATCH_SIZE = 32  # ← Should be in config
```

**FIX:**
```python
# backend/app/constants.py
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128
AST_TIME_FRAMES = 1024
DEFAULT_BATCH_SIZE = 32

# OR in config.py Settings class
class Settings(BaseSettings):
    # Audio processing
    MEL_N_FFT: int = 2048
    MEL_HOP_LENGTH: int = 512
    MEL_N_MELS: int = 128
    AST_TIME_FRAMES: int = 1024
    INFERENCE_BATCH_SIZE: int = 32
```

---

## FULL ISSUE COUNT

| Severity | Count | Examples |
|----------|-------|----------|
| CRITICAL | 6 | Path traversal (×4), memory leak, race condition |
| HIGH | 10 | CSV double-read, N+1 query, hardcoded paths (×2), blocking I/O, zombie processes |
| MEDIUM | 6 | Dead code, bare except (×4), print→logging |
| LOW | 3 | Magic numbers, wavesurfer migration, type hints |

---

## EVEN MORE FINDINGS (Deeper Analysis)

### 23. Training Service: Another Memory Leak

**Location:** `backend/app/services/ast_training.py:72`

```python
class ASTTrainingService:
    def __init__(self):
        self.jobs: Dict[str, TrainingStatus] = {}  # ← GROWS FOREVER
        self.active_threads: Dict[str, threading.Thread] = {}  # ← NEVER CLEANED
        self.cancel_flags: Dict[str, bool] = {}  # ← NEVER CLEANED
```

**Same problem as analyze.py** - no cleanup of old jobs.

---

### 24. Training Runs in Daemon Thread - Dies Silently

**Location:** `backend/app/services/ast_training.py:103-108`

```python
thread = threading.Thread(
    target=self._train_model_background,
    args=(job_id,),
    daemon=True  # ← If main process exits, training DIES without saving!
)
```

**Problem:** `daemon=True` means thread is killed when main process exits. Training could be 90% complete and lose everything.

**FIX:**
```python
thread = threading.Thread(
    target=self._train_model_background,
    args=(job_id,),
    daemon=False  # Keep alive until complete
)

# Add graceful shutdown handler
import atexit
atexit.register(self._wait_for_training_completion)
```

---

### 25. More Bare Except in Training

**Location:** `backend/app/services/ast_training.py:169-171`

```python
try:
    info = torchaudio.info(str(wav_file))
    duration = info.num_frames / info.sample_rate
    chunk_count += int(duration / settings.FRAME_DURATION_SEC)
except Exception as e:  # At least catches specific!
    print(f"[Training {job_id}]   ERROR reading {wav_file.name}: {e}", flush=True)
    chunk_count += 1
```

**Location:** `backend/app/services/ast_training.py:213-216`

```python
try:
    os.link(str(wav_file), str(link_path))  # Hardlink
except:
    shutil.copy(str(wav_file), str(link_path))  # ← BARE EXCEPT
```

---

### 26. Training Uses print() Extensively

**Location:** `backend/app/services/ast_training.py` - 30+ print statements

```python
print(f"[Training {job_id}] [{datetime.now().strftime('%H:%M:%S')}] Starting dataset preparation...", flush=True)
print(f"[Training {job_id}] Created temp folder: {dataset_folder}", flush=True)
print(f"[Training {job_id}] Scanning {class_name}...", flush=True)
# ... 30 more
```

**Fix:** Use logging module with training-specific logger.

---

### 27. Model Registry Uses print() for Errors

**Location:** `backend/app/services/model_registry.py:38`, `145`

```python
except Exception as e:
    print(f"Error loading metadata: {e}")  # ← LOST IN VOID
    return {"active_model": None, "models": []}

except Exception as e:
    print(f"Error reading edited CSVs: {e}")  # ← LOST IN VOID
    return False
```

**Fix:**
```python
import logging
logger = logging.getLogger(__name__)

except Exception as e:
    logger.error(f"Error loading metadata: {e}")
```

---

### 28. Worker Has Hardcoded /tmp Path

**Location:** `backend/app/workers/analyze_worker.py:22`

```python
JOBS_DIR = Path("/tmp/filharmonia_jobs")  # ← HARDCODED LINUX PATH!
```

**Problem:** Doesn't work on Windows (no `/tmp/`).

**FIX:**
```python
import tempfile
JOBS_DIR = Path(tempfile.gettempdir()) / "filharmonia_jobs"
```

---

### 29. Waveform API: No Path Validation

**Location:** `backend/app/api/v1/waveform.py:23-26`

```python
@router.get("/data")
async def get_waveform_data(path: str = Query(...)):
    mp3_path = Path(path)  # ← NO VALIDATION!

    if not mp3_path.exists():
        raise HTTPException(404, f"File not found: {path}")
```

**Attack:** `GET /api/v1/waveform/data?path=/etc/passwd` - might reveal file existence

**FIX:** Same path validation as other endpoints.

---

### 30. Waveform Generated On Every Request (No Cache)

**Location:** `backend/app/api/v1/waveform.py:29-46`

```python
# Load audio (mono, lower sample rate for speed)
y, sr = librosa.load(str(mp3_path), sr=8000, mono=True)  # ← EVERY REQUEST!

# Calculate how many data points we need
num_pixels = len(y) // samples_per_pixel

# Generate min/max for each pixel
data = []
for i in range(num_pixels):  # ← SLOW LOOP EVERY TIME!
```

**Problem:** For 40-minute file, this takes 2-5 seconds EVERY time UI opens it.

**FIX:**
```python
import hashlib
from functools import lru_cache

WAVEFORM_CACHE_DIR = Path(tempfile.gettempdir()) / "filharmonia_waveforms"

def get_cache_path(mp3_path: Path, samples_per_pixel: int) -> Path:
    """Generate cache path based on file content hash"""
    stat = mp3_path.stat()
    key = f"{mp3_path}:{stat.st_size}:{stat.st_mtime}:{samples_per_pixel}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return WAVEFORM_CACHE_DIR / f"{hash_key}.json"

@router.get("/data")
async def get_waveform_data(path: str, samples_per_pixel: int = 512):
    mp3_path = Path(path)
    cache_path = get_cache_path(mp3_path, samples_per_pixel)

    # Return cached if exists
    if cache_path.exists():
        return JSONResponse(json.loads(cache_path.read_text()))

    # Generate and cache
    data = generate_waveform(mp3_path, samples_per_pixel)
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return JSONResponse(data)
```

---

### 31. Export API: No Path Validation

**Location:** `backend/app/api/v1/export.py:124-126`

```python
@router.post("/training-data")
async def export_training_data(request: ExportRequest):
    mp3_path = Path(request.mp3_path)  # ← NO VALIDATION
    if not mp3_path.exists():
        raise HTTPException(404, f"MP3 file not found: {mp3_path}")
```

**Same issue** - arbitrary file access possible.

---

### 32. Config: Settings Created at Import Time

**Location:** `backend/app/config.py:27-28`

```python
# Ensure base directory exists
FILHARMONIA_BASE.mkdir(parents=True, exist_ok=True)  # ← RUNS ON IMPORT!
```

**Problem:** Importing config module creates directories. Side effects on import are bad practice.

**FIX:**
```python
class Settings(BaseSettings):
    # Use Pydantic BaseSettings for proper lazy loading

    @property
    def FILHARMONIA_BASE(self) -> Path:
        path = Path(os.getenv("FILHARMONIA_BASE_DIR", ...))
        path.mkdir(parents=True, exist_ok=True)
        return path
```

---

### 33. Librosa Load Entire File to RAM

**Location:** `backend/app/services/analyze.py:38`

```python
y, sr = librosa.load(str(mp3_path), sr=settings.SAMPLE_RATE)
# For 40-minute file at 48kHz: 40*60*48000 = 115M samples × 4 bytes = 460 MB RAM!
```

**Location:** `backend/app/api/v1/export.py:133`

```python
y, sr = librosa.load(str(mp3_path), sr=44100, mono=False)
# Same file, stereo: 460 MB × 2 channels = 920 MB RAM!
```

**Problem:** Multiple concurrent analyses could exhaust memory.

**FIX (streaming):**
```python
import soundfile as sf

# Stream audio in chunks
with sf.SoundFile(str(mp3_path)) as audio:
    sr = audio.samplerate
    block_size = int(settings.FRAME_DURATION_SEC * sr)

    for block in audio.blocks(blocksize=block_size):
        # Process chunk, don't accumulate
        prediction = model.predict(block)
        writer.writerow([...])
```

---

### 34. Settings Class Not Using Pydantic Properly

**Location:** `backend/app/config.py:14-52`

```python
class Settings:  # ← Plain class, not BaseSettings!
    FILHARMONIA_BASE: Path = Path(...)

    # No validation, no .env support beyond manual load_dotenv
    # No type coercion
```

**FIX:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FILHARMONIA_BASE_DIR: Path = Path.cwd() / "FILHARMONIA_DATA"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @property
    def SORTED_FOLDER(self) -> Path:
        return self.FILHARMONIA_BASE_DIR / "SORTED"
```

---

## FULL ISSUE COUNT (UPDATED)

| Severity | Count | Examples |
|----------|-------|----------|
| CRITICAL | 7 | Path traversal (×5), memory leaks (×2), race condition |
| HIGH | 14 | CSV double-read, N+1 query, hardcoded paths (×3), blocking I/O, daemon threads, waveform no cache |
| MEDIUM | 10 | Bare except (×8), print→logging (×2) |
| LOW | 4 | Magic numbers, wavesurfer, type hints, Settings class |

---

## FINAL FINDINGS

### 35. Shutdown Uses print() Not Logging

**Location:** `backend/app/main.py:12-28`

```python
print("🔄 Graceful shutdown: marking active analysis jobs as interrupted...")
# ...
print(f"  ⚠️  Batch job {job_id[:8]} interrupted")
print("✓ Shutdown complete")
```

**FIX:** Use logging.info()

---

### 36. Debug console.log Left in Production Code

**Location:** `frontend/src/pages/UncertaintyReview.tsx:67-68`

```typescript
useEffect(() => {
  console.log('[UncertaintyReview] Zoom changed to:', zoom)  // ← DEBUG LEFT IN!
}, [zoom])
```

**Location:** `frontend/src/pages/UncertaintyReview.tsx:172`

```typescript
console.log('[UncertaintyReview] loadWaveform triggered for segment:', ...)  // ← DEBUG!
```

**FIX:** Remove or use conditional debug flag.

---

### 37. Duplicated Wheel Zoom Logic

**Location:** `frontend/src/pages/UncertaintyReview.tsx:115-166` and `frontend/src/components/StickyPlayer.tsx:86-141`

Same 50+ lines of wheel zoom handler copy-pasted between components.

**FIX:**
```typescript
// hooks/useWheelZoom.ts
export function useWheelZoom(canvasRef: RefObject<HTMLCanvasElement>, scrollContainerRef: RefObject<HTMLDivElement>) {
  const [zoom, setZoom] = useState(1)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const handleWheel = (e: WheelEvent) => {
      // ... unified logic
    }

    canvas.addEventListener('wheel', handleWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', handleWheel)
  }, [zoom])

  return { zoom, setZoom }
}
```

---

### 38. Hardcoded Frame Duration in Frontend

**Location:** `frontend/src/pages/UncertaintyReview.tsx:185`

```typescript
const segmentEndSec = segmentStartSec + 2.97  // FRAME_DURATION_SEC ← HARDCODED!
```

**Problem:** If backend changes FRAME_DURATION_SEC, frontend breaks.

**FIX:** Get from API or environment variable.

---

### 39. No Global Error Handler

**Location:** `backend/app/main.py` - missing

**Problem:** Unhandled exceptions return ugly 500 errors without error IDs for debugging.

**FIX:**
```python
from fastapi import Request
from fastapi.responses import JSONResponse
import uuid
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"[{error_id}] Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "error_id": error_id,
            "detail": str(exc) if settings.DEBUG else None
        }
    )
```

---

### 40. No Request Logging/Tracing

**Location:** `backend/app/main.py` - missing

**Problem:** No way to track which requests are slow or failing.

**FIX:**
```python
import time
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )
    return response
```

---

## COMPLETE ISSUE SUMMARY

| Severity | Count | Key Issues |
|----------|-------|------------|
| **CRITICAL** | 7 | Path traversal (×5), memory leaks (×2), race condition |
| **HIGH** | 16 | CSV double-read, N+1 query, hardcoded paths (×3), blocking I/O, zombie processes, daemon threads, waveform no cache, no error handler |
| **MEDIUM** | 12 | Bare except (×8), print→logging (×4) |
| **LOW** | 5 | Magic numbers, type hints, debug console.log, duplicated code, Settings class |

**Total: 40 specific issues with file:line references**

---

## TOP 10 HIGHEST IMPACT FIXES

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | **Path traversal fixes** | Security | 30min |
| 2 | **pandas → Polars** | 5-30x CSV speedup | 2-4h |
| 3 | **Waveform caching** | 2-5s → instant UI | 1-2h |
| 4 | **Memory leak fixes** | Stability | 1h |
| 5 | **Hardcoded path elimination** | Cross-platform | 1h |
| 6 | **CsvViewer split** | Maintainability | 4-6h |
| 7 | **Global error handler** | Debugging | 30min |
| 8 | **Logging infrastructure** | Debugging | 1h |
| 9 | **Bare except cleanup** | Debugging | 2h |
| 10 | **Polling → exponential backoff** | 90% fewer API calls | 1h |

---

## DEPENDENCY & ARCHITECTURE ANALYSIS

### 41. Unused Frontend Dependencies

**Location:** `frontend/package.json`

```json
{
  "dependencies": {
    "howler": "^2.2.4",           // ← NOT USED (HTML5 audio instead)
    "react-hook-form": "^7.51.0", // ← NOT USED (no forms!)
    "react-router-dom": "^6.22.0", // ← NOT USED (state-based navigation)
    "waveform-data": "^4.5.2",     // ← BARELY USED (custom canvas code)
  }
}
```

**FIX:**
```bash
npm uninstall howler react-hook-form react-router-dom
# Saves ~50KB bundle size
```

---

### 42. Legacy TensorFlow/Keras - 500MB+ Bloat

**Location:** `backend/requirements.txt`

```txt
tensorflow==2.13.1        # ← ONLY USED IN LEGACY training.py
tensorflow-estimator==2.13.0
tensorflow-intel==2.13.1
tensorflow-io-gcs-filesystem==0.31.0
keras==2.13.1             # ← LEGACY, ast_training.py uses PyTorch

# Plus all TF dependencies (~30 packages)
```

**Problem:**
- TensorFlow + deps = ~500MB
- Only used in `backend/app/services/training.py` (legacy CNN training)
- PyTorch AST is the active model, Keras CNN is deprecated

**FIX:** Remove TensorFlow if not needed:
```bash
pip uninstall tensorflow tensorflow-estimator keras
# Or move training.py to separate optional module
```

---

### 43. CUDA Version Mismatch

**Location:** `backend/requirements.txt`

```txt
nvidia-cublas-cu11==11.11.3.6    # ← CUDA 11
nvidia-cudnn-cu11==9.10.2.21     # ← CUDA 11
torch==2.5.1+cu121               # ← CUDA 12.1 !!!
```

**Problem:** PyTorch compiled for CUDA 12.1 but old CUDA 11 packages installed.

**FIX:**
```txt
# Remove CUDA 11 packages
# nvidia-cublas-cu11  ← DELETE
# nvidia-cudnn-cu11   ← DELETE

# torch==2.5.1+cu121 already includes CUDA 12.1 deps
```

---

### 44. Frontend Components Are Too Large

**Component Line Counts:**

| Component | Lines | Status |
|-----------|-------|--------|
| CsvViewer.tsx | 1268 | **CRITICAL** - split into 5+ |
| UncertaintyReview.tsx | 970 | **HIGH** - split into 4+ |
| StickyPlayer.tsx | 718 | **MEDIUM** - extract hooks |
| TrainingManager.tsx | 600 | Acceptable |
| CalendarBrowser.tsx | 555 | Acceptable |
| SortManager.tsx | 490 | Acceptable |
| WaveformEditor.tsx | 387 | Good |
| AnalysisMonitor.tsx | 350 | Good |
| Toast.tsx | 118 | Good |
| **Total** | **5456** | avg 606 lines/file |

**Target:** Max 300 lines per component.

**Priority splits:**
1. CsvViewer.tsx → 5 components + 4 hooks
2. UncertaintyReview.tsx → 4 components + 3 hooks
3. StickyPlayer.tsx → extract to wavesurfer.js or 2 hooks

---

### 45. No Code Splitting / Lazy Loading

**Location:** `frontend/src/App.tsx`

```typescript
// ALL pages imported at once
import CsvViewer from './pages/CsvViewer'
import CalendarBrowser from './pages/CalendarBrowser'
import UncertaintyReview from './pages/UncertaintyReview'
// ...all 8 pages
```

**Problem:** Initial bundle loads ALL pages even if user only uses CsvViewer.

**FIX:**
```typescript
import { lazy, Suspense } from 'react'

const CsvViewer = lazy(() => import('./pages/CsvViewer'))
const CalendarBrowser = lazy(() => import('./pages/CalendarBrowser'))
const UncertaintyReview = lazy(() => import('./pages/UncertaintyReview'))

// In render:
<Suspense fallback={<div>Loading...</div>}>
  {currentPage === 'csv' && <CsvViewer />}
</Suspense>
```

**Benefit:** Initial load 30-50% faster.

---

### 46. No Tests

**Location:** Entire codebase

**Backend:**
- pytest installed but no tests exist
- 0 test files in backend/

**Frontend:**
- No test runner configured
- No test files
- No component tests

**Minimum test coverage needed:**
1. Path validation functions (security critical!)
2. Time parsing functions (crash sources)
3. CSV parsing functions
4. Job status management
5. Waveform caching

---

### 47. No ESLint / Prettier in Frontend

**Location:** `frontend/package.json`

```json
{
  "devDependencies": {
    // No eslint
    // No prettier
    // No husky/lint-staged
  }
}
```

**Problem:** No code style enforcement, inconsistent formatting, no automatic bug detection.

**FIX:**
```bash
npm install -D eslint @typescript-eslint/eslint-plugin @typescript-eslint/parser prettier eslint-config-prettier
```

---

### 48. Backend Has No Linting

**Location:** Backend has no pyproject.toml, no ruff, no mypy config

**FIX:**
```toml
# pyproject.toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "C4", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
```

---

### 49. Multiple Implementations of Time Parsing

**Locations:**
- `backend/app/api/v1/csv_parser.py:203` - `time_to_seconds()`
- `backend/app/api/v1/uncertainty.py:19` - `time_to_seconds()`
- `frontend/src/pages/CsvViewer.tsx:340` - inline parsing
- `frontend/src/pages/UncertaintyReview.tsx:22` - `timeToSeconds()`
- `frontend/src/components/StickyPlayer.tsx:28` - `timeToSeconds()`

**5 different implementations of the same function!**

**FIX:**

Backend:
```python
# backend/app/utils/time.py
def time_to_seconds(time_str: str) -> float:
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
```

Frontend:
```typescript
// frontend/src/utils/time.ts
export function timeToSeconds(time: string): number {
  const [h, m, s] = time.split(':').map(Number)
  return h * 3600 + m * 60 + s
}
```

---

### 50. No API Client / Types Sharing

**Location:** Frontend makes raw axios calls everywhere

```typescript
// CsvViewer.tsx
const res = await axios.get('/api/v1/files/analysis-results')

// CalendarBrowser.tsx
const [recordingsRes, csvRes] = await Promise.all([
  axios.get('/api/v1/files/sorted'),
  axios.get('/api/v1/files/analysis-results')
])
```

**Problem:**
- No type safety on API responses
- Duplicate endpoint strings
- No error handling standardization

**FIX:**
```typescript
// api/client.ts
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

// api/files.ts
export interface FileInfo {
  path: string
  name: string
  size: number
  date: string
  type: 'mp3' | 'csv'
}

export async function getAnalysisResults(): Promise<FileInfo[]> {
  const { data } = await api.get<FileInfo[]>('/files/analysis-results')
  return data
}
```

---

## UPDATED SUMMARY

| Category | Count | Examples |
|----------|-------|----------|
| **CRITICAL** | 7 | Path traversal ×5, memory leaks ×2 |
| **HIGH** | 18 | Waveform cache, hardcoded paths, component splits, TF bloat |
| **MEDIUM** | 15 | Bare except, logging, unused deps, code duplication |
| **LOW** | 10 | Linting, types, lazy loading, tests |

---

## FINAL BATCH: Security, Accessibility, Performance

### 51. ANOTHER Delete Endpoint Without Path Validation

**Location:** `backend/app/api/v1/sort.py:79-106`

```python
@router.post("/delete-duplicates")
async def delete_duplicates(request: DeleteDuplicatesRequest):
    for file_path in request.file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)  # ← DELETES ANY FILE!
```

**Attack:** POST with `file_paths: ["/etc/passwd", "/boot/vmlinuz"]` → deletes system files

**Total path traversal vulnerabilities: 6**

---

### 52. Audio Stream Has No Path Validation

**Location:** `backend/app/api/v1/audio.py:21-24`

```python
@router.get("/stream")
async def stream_audio(path: str = Query(...)):
    mp3_path = Path(path)  # ← NO VALIDATION
    if not mp3_path.exists():
        raise HTTPException(404, f"File not found: {path}")
```

**Total endpoints without path validation: 7**

---

### 53. Zero Accessibility (a11y)

**Location:** Entire frontend

```
aria-* attributes: 0
role= attributes: 0
tabIndex attributes: 0
onKeyDown handlers: 0 (except wheel zoom)
```

**Problems:**
- Screen readers cannot navigate
- Keyboard-only users blocked
- Not WCAG compliant

**Minimum fixes needed:**
```tsx
// Buttons need labels
<button aria-label="Play segment" onClick={...}>▶</button>

// Interactive elements need keyboard support
<div
  role="button"
  tabIndex={0}
  onKeyDown={(e) => e.key === 'Enter' && handleClick()}
>

// Tables need headers
<table role="grid" aria-label="Tracks">
  <thead><tr><th scope="col">Time</th></tr></thead>
```

---

### 54. Bare Except in Sort Service

**Location:** `backend/app/services/sort.py:88-95`

```python
except Exception as e:
    new_files.append({
        'path': path,
        'status': 'error',
        'error': str(e)  # At least captures error, but too broad
    })
```

**Total bare/broad except blocks: 12**

---

### 55. os.walk on Network Drive Can Be Slow

**Location:** `backend/app/services/sort.py:25-28`

```python
for root, dirs, files in os.walk(self.source_folder):
    # This scans ENTIRE folder tree synchronously
    # On network drive = minutes of blocking
```

**FIX:**
```python
# Use async with aiofiles or run in executor
import asyncio
loop = asyncio.get_event_loop()
files = await loop.run_in_executor(None, self._scan_sync)
```

---

### 56. No Rate Limiting

**Location:** `backend/app/main.py` - missing

**Problem:** Any endpoint can be called unlimited times. DoS attack trivial.

**FIX:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/v1/files/sorted")
@limiter.limit("60/minute")
async def list_sorted_files(request: Request):
    ...
```

---

### 57. No Health Check for Dependencies

**Location:** `backend/app/main.py:75-77`

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}  # ← Always returns healthy!
```

**Problem:** Doesn't check if model loaded, GPU available, disk space, etc.

**FIX:**
```python
@app.get("/health")
async def health():
    checks = {
        "api": "healthy",
        "gpu": "available" if torch.cuda.is_available() else "unavailable",
        "model": "loaded" if _service and _service.model else "not_loaded",
        "disk_space": "ok" if shutil.disk_usage("/").free > 1e9 else "low",
    }
    status = "healthy" if all(v in ("healthy", "available", "loaded", "ok") for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
```

---

### 58. No Graceful Degradation for GPU

**Location:** `backend/app/services/ast_inference.py:26`

```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set once at init, never rechecked
```

**Problem:** If GPU becomes unavailable (driver crash, OOM), inference fails completely.

**FIX:**
```python
def get_device(self) -> torch.device:
    """Get best available device, with fallback"""
    try:
        if torch.cuda.is_available():
            # Test that CUDA actually works
            torch.zeros(1).cuda()
            return torch.device('cuda')
    except RuntimeError as e:
        logger.warning(f"CUDA unavailable: {e}, falling back to CPU")
    return torch.device('cpu')
```

---

### 59. Frontend Has No Error Boundaries

**Location:** `frontend/src/App.tsx` - missing

**Problem:** Any component crash kills entire app with white screen.

**FIX:**
```tsx
import { ErrorBoundary } from 'react-error-boundary'

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <div role="alert">
      <p>Something went wrong:</p>
      <pre>{error.message}</pre>
      <button onClick={resetErrorBoundary}>Try again</button>
    </div>
  )
}

<ErrorBoundary FallbackComponent={ErrorFallback}>
  <App />
</ErrorBoundary>
```

---

### 60. No Loading States for Async Operations

**Location:** Multiple components

```tsx
// CsvViewer.tsx - loading state exists but not always shown
const [loading, setLoading] = useState(false)

// But many operations don't set loading:
const handleExport = async () => {
  // setLoading(true) ← MISSING
  await axios.post('/api/v1/export/training-data', ...)
  // setLoading(false) ← MISSING
}
```

**User sees: Frozen UI, no feedback that something is happening.**

---

## COMPLETE AUDIT STATISTICS

### By Category

| Category | Count |
|----------|-------|
| **Security** | 8 (path traversal ×7, no rate limiting) |
| **Memory/Stability** | 5 (leaks, zombies, race conditions) |
| **Performance** | 12 (caching, queries, blocking I/O) |
| **Architecture** | 10 (component size, dependencies, patterns) |
| **Code Quality** | 15 (bare except, logging, duplication) |
| **DX/Tooling** | 5 (linting, tests, types) |
| **Accessibility** | 3 (keyboard, screen readers, WCAG) |
| **UX** | 2 (error boundaries, loading states) |

### By Severity

| Severity | Count | Action |
|----------|-------|--------|
| **CRITICAL** | 8 | Fix before any deployment |
| **HIGH** | 20 | Fix in v1 |
| **MEDIUM** | 18 | Fix in v1 if time |
| **LOW** | 14 | Nice to have |

**Total: 60 specific issues with file:line references**

---

## RECOMMENDED FIX ORDER

### Week 1: Security & Stability
1. Path validation on ALL 7 endpoints (CRITICAL)
2. Memory leak fixes (CRITICAL)
3. Global error handler + logging (HIGH)
4. Waveform caching (HIGH)

### Week 2: Performance
5. pandas → Polars migration (HIGH)
6. CSV double-read fixes (HIGH)
7. Polling → exponential backoff (MEDIUM)
8. Remove TensorFlow bloat (MEDIUM)

### Week 3: Architecture
9. CsvViewer.tsx split (HIGH)
10. Extract shared hooks (MEDIUM)
11. Add API client layer (MEDIUM)
12. Code splitting / lazy loading (LOW)

### Week 4: Quality & Polish
13. Bare except cleanup (MEDIUM)
14. Add logging throughout (MEDIUM)
15. Add ESLint + Prettier (LOW)
16. Basic test coverage (LOW)

---

*Detailed audit completed: 2026-01-20*
*60 issues identified across 8 categories*

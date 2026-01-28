# Phase 2: Core UX Polish - Research

**Researched:** 2026-01-28
**Domain:** React keyboard shortcuts, undo/redo state management, toast notifications, backend error standardization
**Confidence:** HIGH

## Summary

This phase adds keyboard-driven editing workflow, undo/redo, progress feedback, and error handling polish to an existing React+TypeScript frontend (React 18, zustand, TailwindCSS) with a FastAPI backend. The codebase already has well-structured hooks (`useTrackEditor`, `useAudioPlayer`, `useAutosave`), an existing `Toast` component, and exception handlers in `main.py`.

The primary work is: (1) a `useUndoRedo` hook wrapping `useTrackEditor` to capture track snapshots, (2) a `useKeyboardShortcuts` hook for global key bindings, (3) a zustand-based toast manager for persistent error notifications, (4) standardizing the backend error response `code` field, and (5) atomic file writes for save.

**Primary recommendation:** Build undo/redo as a snapshot-based history stack wrapping the existing `useTrackEditor` hook. Keep all new logic in hooks, not components.

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 18.3.1 | UI framework | Already in use |
| zustand | 4.5.2 | State management | Already in use, ideal for toast store |
| TailwindCSS | 3.4.1 | Styling | Already in use |
| axios | 1.6.7 | HTTP client | Already in use |
| lucide-react | 0.344.0 | Icons | Already in use |

### No New Dependencies Needed

All phase requirements can be implemented with existing dependencies. No new libraries required.

## Architecture Patterns

### Recommended New Files
```
frontend/src/
├── hooks/
│   ├── useUndoRedo.ts          # Snapshot-based undo/redo wrapping track state
│   ├── useKeyboardShortcuts.ts # Global keyboard event handler
│   └── useToastManager.ts      # (or store) zustand store for toast queue
├── components/
│   ├── Toast.tsx               # EXISTS - enhance with close button, retry action
│   ├── ToastContainer.tsx      # Renders toast queue from zustand store
│   ├── ProgressIndicator.tsx   # Inline progress with stage + percentage
│   └── KeyboardHelp.tsx        # Shortcut reference panel (? key toggle)
├── stores/
│   └── toastStore.ts           # zustand store for toast notifications
├── utils/
│   └── errorHandler.ts         # Axios interceptor + error formatting
backend/app/
├── api/v1/
│   └── csv_parser.py           # MODIFY - atomic write in save endpoint
├── main.py                     # MODIFY - add `code` field to error responses
```

### Pattern 1: Snapshot-Based Undo/Redo
**What:** Store `Track[]` snapshots in a bounded circular buffer (max 20). Each mutation in `useTrackEditor` pushes current state before applying change.
**When to use:** All track edit operations (updateClass, deleteTrack, mergeWithNext, etc.)
**Example:**
```typescript
interface UndoState {
  past: Track[][]    // max 20
  present: Track[]
  future: Track[][]  // cleared on new edit, populated on undo
}

function useUndoRedo(trackEditor: UseTrackEditorReturn) {
  const [history, setHistory] = useState<UndoState>({
    past: [], present: [], future: []
  })

  const pushState = useCallback((newTracks: Track[]) => {
    setHistory(prev => ({
      past: [...prev.past.slice(-19), prev.present],
      present: newTracks,
      future: [] // clear redo on new action
    }))
  }, [])

  const undo = useCallback(() => {
    setHistory(prev => {
      if (prev.past.length === 0) return prev
      const previous = prev.past[prev.past.length - 1]
      return {
        past: prev.past.slice(0, -1),
        present: previous,
        future: [prev.present, ...prev.future]
      }
    })
  }, [])

  const redo = useCallback(() => {
    setHistory(prev => {
      if (prev.future.length === 0) return prev
      const next = prev.future[0]
      return {
        past: [...prev.past, prev.present],
        present: next,
        future: prev.future.slice(1)
      }
    })
  }, [])

  return { undo, redo, canUndo: history.past.length > 0, canRedo: history.future.length > 0 }
}
```

### Pattern 2: Global Keyboard Shortcuts Hook
**What:** Single `useEffect` with `keydown` listener on `document`, conditional on focus context.
**When to use:** CsvViewer mounts this hook.
**Example:**
```typescript
function useKeyboardShortcuts(handlers: Record<string, () => void>) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if user is typing in an input/textarea
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) return

      // Build key string: "ctrl+s", "shift+z", "space", "1", etc.
      const parts: string[] = []
      if (e.ctrlKey || e.metaKey) parts.push('ctrl')
      if (e.shiftKey) parts.push('shift')
      parts.push(e.key.toLowerCase())
      const combo = parts.join('+')

      if (handlers[combo]) {
        e.preventDefault()
        handlers[combo]()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handlers])
}
```

### Pattern 3: Zustand Toast Store
**What:** Global toast queue with max 5 visible, manual dismiss required (per user decision), optional retry action.
**Example:**
```typescript
interface ToastItem {
  id: string
  title: string
  message?: string
  color: 'green' | 'red' | 'blue' | 'yellow'
  icon?: string
  retry?: () => void
}

interface ToastStore {
  toasts: ToastItem[]
  addToast: (toast: Omit<ToastItem, 'id'>) => void
  removeToast: (id: string) => void
  clearAll: () => void
}
```

### Pattern 4: Axios Error Interceptor
**What:** Single axios response interceptor that parses standardized error responses and dispatches to toast store.
**Example:**
```typescript
axios.interceptors.response.use(
  response => response,
  error => {
    const data = error.response?.data
    if (data?.status === 'error') {
      toastStore.getState().addToast({
        title: data.message || 'Error',
        message: data.error_id ? `Error ID: ${data.error_id}` : undefined,
        color: 'red',
        retry: error.config?.method !== 'get' ? undefined : () => axios(error.config)
      })
    }
    return Promise.reject(error)
  }
)
```

### Anti-Patterns to Avoid
- **Undo via inverse operations:** Don't try to compute reverse of each operation (mergeWithNext inverse is complex). Snapshot approach is simpler and correct.
- **Keyboard listeners per component:** Don't add `onKeyDown` to individual components. One global listener with context awareness.
- **Auto-dismissing error toasts:** User explicitly requires manual dismissal (had bad experience missing auto-dismissed errors).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Toast queue management | Manual useState arrays | Zustand store | Global access, no prop drilling, easy from axios interceptor |
| Keyboard combo parsing | Complex switch statements | Simple key string builder | `${ctrl?'ctrl+':''}${key}` pattern covers all cases |
| Undo/redo | Command pattern with inverse ops | Snapshot stack | Track arrays are small (<200 items), snapshots are cheap, inverse operations are error-prone |

## Common Pitfalls

### Pitfall 1: Keyboard shortcuts fire while typing
**What goes wrong:** User types "5" in a text input, triggers class change.
**Why it happens:** Global listener without focus checks.
**How to avoid:** Check `e.target.tagName` for INPUT/TEXTAREA/contentEditable before handling.
**Warning signs:** Shortcuts trigger unexpectedly during text editing.

### Pitfall 2: Undo history becomes stale after save
**What goes wrong:** User saves (Ctrl+S), then undoes expecting to go back to pre-save state, but undo was cleared.
**Why it happens:** Save resetting undo history.
**How to avoid:** Per user decision, undo history persists across saves. Only clear on file switch.

### Pitfall 3: Spacebar play/pause scrolls page
**What goes wrong:** Pressing space scrolls the page down (browser default).
**Why it happens:** Space key default behavior in browsers.
**How to avoid:** `e.preventDefault()` when handling spacebar.

### Pitfall 4: Race condition on rapid Ctrl+S
**What goes wrong:** Multiple save requests in flight simultaneously.
**Why it happens:** User presses Ctrl+S rapidly.
**How to avoid:** Debounce or use `isSaving` guard from existing `useAutosave` hook.

### Pitfall 5: Non-atomic file writes corrupt CSV on crash
**What goes wrong:** `write_text()` writes partial file if process crashes mid-write.
**Why it happens:** Direct file write without temp+rename pattern.
**How to avoid:** Write to `.tmp` file then `os.replace()` (atomic on both Unix and Windows).
**Current state:** Both `save_csv` and `autosave_csv` endpoints use direct `write_text()` -- both need atomic write.

### Pitfall 6: Number keys 1-5 with no selected track
**What goes wrong:** User presses 1-5 but no track is selected, nothing happens, no feedback.
**How to avoid:** Show brief feedback or ignore gracefully. Consider using `selectedTrackId` from `useAudioPlayer` or table selection.

## Code Examples

### Atomic Write (Backend - Python)
```python
import os
import tempfile
from pathlib import Path

def atomic_write(path: Path, content: str, encoding: str = 'utf-8') -> None:
    """Write content atomically using temp file + os.replace."""
    parent = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except:
        os.unlink(tmp_path)
        raise
```

### Standardized Error Response Format
Current backend responses already include `status`, `message`, `type`. Missing: `code` field.
```python
# Proposed standard format (all error responses):
{
    "status": "error",
    "message": "Human-readable message",
    "code": "VALIDATION_ERROR",   # NEW - machine-readable error code
    "type": "validation_error",   # existing
    "error_id": "a1b2c3d4",      # existing (500 errors only)
    "details": [...]               # existing (validation errors only)
}
```

### Number Key Classification Cycling
```typescript
// 5 classes map to keys 1-5
const CLASS_ORDER = ['MUSIC', 'APPLAUSE', 'SPEECH', 'PUBLIC', 'TUNING'] as const

// Key press assigns directly (not cycles): 1=MUSIC, 2=APPLAUSE, etc.
// This is more intuitive than cycling through all classes
function handleNumberKey(key: number, selectedTrackId: string | null) {
  if (!selectedTrackId || key < 1 || key > 5) return
  const className = CLASS_ORDER[key - 1]
  updateClass(selectedTrackId, className)
}
```

### Progress Indicator Pattern
```typescript
// Inline progress that replaces the triggering button
interface ProgressState {
  stage: string      // "Loading...", "Analyzing...", "Saving..."
  percent?: number   // 0-100, optional
}

// Use in button: show progress state instead of normal label when active
{progress
  ? <span>{progress.stage} {progress.percent != null ? `${progress.percent}%` : ''}</span>
  : <span>Analyze</span>
}
```

### AbortController for Cancellation (UX-05)
```typescript
const abortRef = useRef<AbortController | null>(null)

const startOperation = async () => {
  // Cancel previous in-flight request
  abortRef.current?.abort()
  abortRef.current = new AbortController()

  try {
    const response = await axios.post('/api/v1/analyze/', data, {
      signal: abortRef.current.signal
    })
  } catch (err) {
    if (axios.isCancel(err)) return // silently ignore cancelled
    throw err
  }
}

const cancelOperation = () => {
  abortRef.current?.abort()
  abortRef.current = null
}
```

## Existing Code Integration Points

| Existing Code | Phase 2 Integration |
|--------------|---------------------|
| `useTrackEditor` hook | Wrap with `useUndoRedo` - intercept all mutations to push snapshots |
| `useAudioPlayer` hook | `playFromSegment`/`togglePlayer` wired to spacebar shortcut |
| `useAutosave` hook | `manualSave()` wired to Ctrl+S shortcut |
| `Toast` component | Enhance with close button (X), retry action; wrap in `ToastContainer` |
| `StickyPlayer.handlePlayPause` | Expose `isPlaying` state for spacebar toggle; add ref forwarding or callback |
| `CsvViewer` | Mount `useKeyboardShortcuts`, pass handlers from other hooks |
| `CalendarBrowser.handlePlayRecording` (line 107) | Currently TODO stub - implement or remove per CLEAN-04 |
| `csv_parser.py save_csv` (line 347) | Replace `write_text` with atomic temp+replace pattern |
| `csv_parser.py autosave_csv` (line 311) | Same atomic write pattern |
| `main.py` exception handlers | Add `code` field to all error response dicts |

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Per-component keyboard handlers | Single global hook with focus awareness | Cleaner, no conflicts |
| Command pattern undo | Snapshot-based undo for small datasets | Simpler, no inverse op bugs |
| Ad-hoc error display | Zustand toast store with axios interceptor | Consistent, global |
| Direct file write | Temp file + os.replace | Crash-safe |

## Open Questions

1. **StickyPlayer spacebar integration**
   - StickyPlayer manages its own `isPlaying` state and `audioRef`
   - Global spacebar needs to call `handlePlayPause` on StickyPlayer
   - Recommendation: Either lift play/pause to `useAudioPlayer` hook or use a callback ref pattern
   - This is the trickiest integration point

2. **handlePlayRecording (CLEAN-04)**
   - Currently a TODO stub logging to console
   - Options: (a) navigate to CsvViewer with recording path, (b) remove the button entirely
   - Recommendation: Implement navigation since the button exists in the UI and users expect it to work

3. **Number keys: direct assignment vs cycling**
   - Direct assignment (1=MUSIC, 2=APPLAUSE...) is simpler and more predictable
   - Cycling (press same key to go to next class) is fewer keystrokes for sequential edits
   - Recommendation: Direct assignment - more intuitive, discoverable via tooltip showing "1: MUSIC, 2: APPLAUSE..."

## Sources

### Primary (HIGH confidence)
- Codebase analysis: All hooks, components, and backend endpoints read directly
- React 18 keyboard event patterns: standard DOM API, well-known
- Zustand store patterns: already used in codebase

### Secondary (MEDIUM confidence)
- Atomic write with `os.replace`: Python docs confirm atomic on POSIX, near-atomic on Windows (same filesystem)
- AbortController + axios: standard pattern, axios supports `signal` option since v0.22

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, everything already in codebase
- Architecture: HIGH - patterns are straightforward React hooks, well-understood
- Pitfalls: HIGH - identified from direct codebase analysis
- Integration points: MEDIUM - StickyPlayer spacebar integration needs exploration during implementation

**Research date:** 2026-01-28
**Valid until:** 2026-02-28 (stable domain, no fast-moving dependencies)

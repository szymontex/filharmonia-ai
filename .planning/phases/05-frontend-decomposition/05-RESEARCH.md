# Phase 5: Frontend Decomposition - Research

**Researched:** 2026-01-28
**Domain:** React/TypeScript Component Architecture
**Confidence:** HIGH

## Summary

This research covers best practices for decomposing large React components into maintainable, single-responsibility components with extracted custom hooks. The current CsvViewer.tsx is 1279 lines with multiple responsibilities: file selection, track editing, audio playback, autosaving, export management, and UI state. Modern React architecture (2026) emphasizes component decomposition, custom hooks for logic reuse, and utility extraction for better testability, maintainability, and code clarity.

The standard approach is to apply the Single Responsibility Principle: extract presentational components (TrackTable, CsvSelector, PlayerControls), extract stateful logic into custom hooks (useTrackEditor, useAudioPlayer, useAutosave), and move pure functions into utilities. This transforms the monolithic component into an orchestration layer that composes smaller, focused pieces.

**Primary recommendation:** Use Container/Presentational pattern with custom hooks. CsvViewer becomes a thin orchestration layer that delegates to specialized components and hooks for each domain concern.

## Standard Stack

The established patterns/tools for React component decomposition:

### Core Patterns
| Pattern | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Custom Hooks | React 18+ | Extract stateful logic | De-facto pattern for logic reuse without HOCs/render props |
| Container/Presentational | - | Separate logic from UI | Industry standard for component organization |
| Single Responsibility | - | Component focus | SOLID principle applied to React components |
| Compound Components | - | Related component composition | Allows flexible composition while sharing state |

### Supporting Tools
| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| TypeScript | 5.x | Type safety | Essential for large components with complex state |
| React DevTools | Latest | Component tree debugging | Visualize component hierarchy after refactor |
| ESLint react-hooks | Latest | Hook rules enforcement | Prevent dependencies array mistakes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom Hooks | Context API | Context adds provider overhead, hooks more lightweight |
| Component Split | HOCs | Hooks are simpler and avoid wrapper hell |
| Manual Extract | Automated Refactor Tools | Manual gives more control, automated faster but less precise |

**Installation:**
```bash
# Already in project (from package.json analysis)
# React 18.3.1, TypeScript 5.4.2 already installed
# No additional dependencies needed for refactoring
```

## Architecture Patterns

### Recommended Project Structure
```
frontend/src/
├── pages/
│   └── CsvViewer.tsx       # Orchestration only (~200 lines)
├── components/
│   ├── CsvSelector.tsx     # File selection UI (~150 lines)
│   ├── TrackTable.tsx      # Table rendering (~300 lines)
│   └── PlayerControls.tsx  # Player integration (~200 lines)
├── hooks/
│   ├── useTrackEditor.ts   # Track CRUD operations
│   ├── useAudioPlayer.ts   # Audio playback state
│   └── useAutosave.ts      # Autosave logic
└── utils/
    └── timeCalculations.ts # Time format conversions
```

### Pattern 1: Container/Presentational Split
**What:** Separate data/logic (container) from rendering (presentational)
**When to use:** Component has multiple responsibilities (data fetching + state management + rendering)
**Example:**
```typescript
// Container: CsvViewer.tsx (orchestration)
// Source: Modern React patterns 2026
export default function CsvViewer({ onBack, initialCsv }: CsvViewerProps) {
  // Custom hooks handle state/logic
  const { tracks, updateTrack, deleteTrack } = useTrackEditor(selectedCsv)
  const { isPlaying, play, pause } = useAudioPlayer(mp3Path)
  const { hasChanges, save, lastSave } = useAutosave(tracks, selectedCsv)

  // Presentational components handle rendering
  return (
    <div>
      <CsvSelector files={csvFiles} onSelect={loadCsv} />
      <TrackTable tracks={tracks} onUpdate={updateTrack} />
      <PlayerControls isPlaying={isPlaying} onToggle={play} />
    </div>
  )
}

// Presentational: TrackTable.tsx
interface TrackTableProps {
  tracks: Track[]
  onUpdate: (id: string, updates: Partial<Track>) => void
  onDelete: (id: string) => void
  exportedSegments: Set<number>
}

export function TrackTable({ tracks, onUpdate, onDelete }: TrackTableProps) {
  // Pure rendering, no business logic
  return (
    <table>
      {tracks.map(track => (
        <TrackRow key={track.id} track={track} onUpdate={onUpdate} />
      ))}
    </table>
  )
}
```

### Pattern 2: Custom Hook for Stateful Logic
**What:** Extract useState/useEffect logic into reusable hooks
**When to use:** Multiple related state variables + effects that belong together
**Example:**
```typescript
// Source: React official docs - Reusing Logic with Custom Hooks
// hooks/useTrackEditor.ts
export function useTrackEditor(csvPath: string | null) {
  const [tracks, setTracks] = useState<Track[]>([])
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

  // Load tracks when CSV changes
  useEffect(() => {
    if (csvPath) {
      loadTracks(csvPath).then(setTracks)
    }
  }, [csvPath])

  // Track update operations
  const updateTrack = useCallback((id: string, updates: Partial<Track>) => {
    setTracks(prev => prev.map(t => t.id === id ? { ...t, ...updates } : t))
    setHasUnsavedChanges(true)
  }, [])

  const deleteTrack = useCallback((id: string) => {
    setTracks(prev => {
      // Merge logic (extracted from CsvViewer.tsx:304-348)
      const idx = prev.findIndex(t => t.id === id)
      if (idx > 0) {
        const merged = mergeWithPrevious(prev, idx)
        return [...prev.slice(0, idx - 1), merged, ...prev.slice(idx + 1)]
      }
      return prev.filter(t => t.id !== id)
    })
    setHasUnsavedChanges(true)
  }, [])

  return { tracks, hasUnsavedChanges, updateTrack, deleteTrack, setTracks }
}
```

### Pattern 3: Autosave Hook with Debouncing
**What:** Automatically save changes after user stops editing
**When to use:** Forms or editors with unsaved changes
**Example:**
```typescript
// Source: React autosave patterns (Synthace, Codemzy)
// hooks/useAutosave.ts
export function useAutosave<T>(
  data: T,
  saveFn: (data: T) => Promise<void>,
  delay: number = 1000
) {
  const [lastSave, setLastSave] = useState<Date | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const dataRef = useRef(data)

  useEffect(() => {
    dataRef.current = data
  }, [data])

  useEffect(() => {
    // Don't save on mount or if data unchanged
    if (!lastSave && !isSaving) return

    const timer = setTimeout(async () => {
      setIsSaving(true)
      try {
        await saveFn(dataRef.current)
        setLastSave(new Date())
      } catch (error) {
        console.error('Autosave failed:', error)
      } finally {
        setIsSaving(false)
      }
    }, delay)

    return () => clearTimeout(timer)
  }, [data, delay, saveFn])

  return { lastSave, isSaving }
}
```

### Pattern 4: Utility Functions for Pure Logic
**What:** Extract calculations and transformations into pure functions
**When to use:** Logic that doesn't need React state/effects (time conversions, formatting)
**Example:**
```typescript
// Source: Best practices for utility functions in TypeScript
// utils/timeCalculations.ts

/**
 * Calculate duration between two HH:MM:SS timestamps
 * Returns duration in M'S" format
 */
export function calculateDuration(start: string, stop: string): string {
  const [sh, sm, ss] = start.split(':').map(Number)
  const [eh, em, es] = stop.split(':').map(Number)

  const startSec = sh * 3600 + sm * 60 + ss
  const endSec = eh * 3600 + em * 60 + es
  const diffSec = endSec - startSec

  const minutes = Math.floor(diffSec / 60)
  const seconds = diffSec % 60

  return `${minutes}'${seconds}"`
}

/**
 * Convert HH:MM:SS timestamp to seconds
 */
export function timeToSeconds(timeStr: string): number {
  const parts = timeStr.split(':').map(Number)
  return parts[0] * 3600 + parts[1] * 60 + parts[2]
}

/**
 * Convert seconds to HH:MM:SS timestamp
 */
export function secondsToTimeFormat(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}
```

### Anti-Patterns to Avoid
- **Half-way extraction:** Don't stop at "component that both composes AND implements" - go full orchestration or full implementation
- **God component:** Avoid one component that knows about too many child components' internals
- **Prop drilling:** If passing props through 3+ levels, use composition or context instead
- **Over-extraction:** Don't extract until component has clear multiple responsibilities (premature abstraction)
- **Inline handlers:** Don't define handlers inline in JSX if they contain complex logic (extract to functions)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Autosave debouncing | Custom setTimeout logic | useAutosave hook pattern | Edge cases: unmount during save, race conditions, stale closures |
| Time parsing | Manual string.split() everywhere | Utility functions | DRY principle, centralized validation, easier testing |
| State synchronization | Manual useEffect chains | Custom hook that encapsulates related state | Avoids stale closures, reduces useEffect dependencies bugs |
| Component size limits | Arbitrary line counts | Single Responsibility check | 300-line focused component > 150-line multi-purpose component |

**Key insight:** React hooks solve the "logic reuse without HOCs" problem. Before hooks, developers used render props or HOCs (higher-order components) which created "wrapper hell". Custom hooks are the 2026 standard for extracting stateful logic.

## Common Pitfalls

### Pitfall 1: Asynchronous State Updates
**What goes wrong:** Accessing state immediately after setState doesn't reflect the update
**Why it happens:** React batches state updates for performance, state setters are asynchronous
**How to avoid:** Use functional updates when new state depends on old state
**Warning signs:** Stale state values, off-by-one errors in counters
**Example:**
```typescript
// WRONG: Stale state
setCount(count + 1)
setCount(count + 1) // Still uses old count, not count + 1

// RIGHT: Functional update
setCount(prev => prev + 1)
setCount(prev => prev + 1) // Uses updated value
```

### Pitfall 2: Missing useCallback Dependencies
**What goes wrong:** Callback functions close over stale props/state values
**Why it happens:** Missing dependencies in useCallback/useMemo dependency arrays
**How to avoid:** Use ESLint react-hooks/exhaustive-deps rule, include all referenced values
**Warning signs:** Callbacks using old prop values, infinite re-render loops
**Example:**
```typescript
// WRONG: Missing 'tracks' dependency
const updateTrack = useCallback((id: string) => {
  setTracks(tracks.map(t => t.id === id ? { ...t, edited: true } : t))
}, []) // Closes over initial tracks value

// RIGHT: Include all dependencies or use functional update
const updateTrack = useCallback((id: string) => {
  setTracks(prev => prev.map(t => t.id === id ? { ...t, edited: true } : t))
}, []) // No external dependencies
```

### Pitfall 3: Direct State Mutation
**What goes wrong:** Mutating state directly doesn't trigger re-renders
**Why it happens:** React compares state by reference, mutation doesn't change reference
**How to avoid:** Always create new objects/arrays with spread or array methods
**Warning signs:** UI not updating after state change, stale data in components
**Example:**
```typescript
// WRONG: Direct mutation
tracks[0].name = "New Name"
setTracks(tracks) // Same reference, no re-render

// RIGHT: Create new array
setTracks(prev => prev.map((t, i) =>
  i === 0 ? { ...t, name: "New Name" } : t
))
```

### Pitfall 4: Over-Engineering Component Split
**What goes wrong:** Too many tiny components make code harder to follow
**Why it happens:** Misunderstanding "small components" - it's about responsibility not size
**How to avoid:** Split when component has multiple concerns, not when it reaches line count
**Warning signs:** Components with single JSX element, excessive prop drilling
**Example:**
```typescript
// WRONG: Over-split
<TableCell><TableCellContent>{value}</TableCellContent></TableCell>

// RIGHT: Split by responsibility
<TrackRow track={track} onUpdate={handleUpdate} />
<ExportControls segments={selected} onExport={handleExport} />
```

### Pitfall 5: Forgetting Cleanup in useEffect
**What goes wrong:** Memory leaks, stale callbacks, race conditions
**Why it happens:** Async operations in useEffect continue after component unmounts
**How to avoid:** Always return cleanup function, check mounted state for async operations
**Warning signs:** "Can't perform state update on unmounted component" warnings
**Example:**
```typescript
// WRONG: No cleanup
useEffect(() => {
  fetchData().then(data => setData(data))
}, [])

// RIGHT: Cleanup with abort controller or flag
useEffect(() => {
  let cancelled = false

  fetchData().then(data => {
    if (!cancelled) setData(data)
  })

  return () => { cancelled = true }
}, [])
```

### Pitfall 6: Prop Drilling Through Many Levels
**What goes wrong:** Intermediate components receive props they don't use, just to pass down
**Why it happens:** Decomposing without considering prop flow
**How to avoid:** Use composition (children prop), Context API, or restructure component tree
**Warning signs:** Props passed through 3+ component levels unchanged
**Example:**
```typescript
// WRONG: Prop drilling
<CsvViewer tracks={tracks}>
  <Header tracks={tracks}>
    <TrackCount tracks={tracks} /> {/* Only Header passes it */}
  </Header>
</CsvViewer>

// RIGHT: Composition or context
<CsvViewer tracks={tracks}>
  <Header>
    <TrackCount count={tracks.length} /> {/* Only pass what's needed */}
  </Header>
</CsvViewer>
```

## Code Examples

Verified patterns from official sources:

### Example 1: Extract Track Editor Hook
```typescript
// Source: React official docs + analysis of CsvViewer.tsx
// hooks/useTrackEditor.ts

interface UseTrackEditorOptions {
  threshold?: number
  onError?: (error: Error) => void
}

export function useTrackEditor(
  csvPath: string | null,
  options: UseTrackEditorOptions = {}
) {
  const { threshold = 5, onError } = options
  const [tracks, setTracks] = useState<Track[]>([])
  const [loading, setLoading] = useState(false)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

  // Load CSV with threshold
  useEffect(() => {
    if (!csvPath) return

    setLoading(true)
    axios.get(`/api/v1/csv/parse?path=${encodeURIComponent(csvPath)}&threshold=${threshold}`)
      .then(res => {
        setTracks(res.data.tracks)
        setHasUnsavedChanges(false)
      })
      .catch(err => {
        onError?.(err)
      })
      .finally(() => setLoading(false))
  }, [csvPath, threshold, onError])

  const updateName = useCallback((id: string, name: string) => {
    setTracks(prev => prev.map(t => t.id === id ? { ...t, name } : t))
    setHasUnsavedChanges(true)
  }, [])

  const updateClass = useCallback((id: string, predicted_class: string) => {
    setTracks(prev => prev.map(t => t.id === id ? { ...t, predicted_class } : t))
    setHasUnsavedChanges(true)
  }, [])

  const updateStart = useCallback((id: string, start: string) => {
    setTracks(prev => {
      const trackIndex = prev.findIndex(t => t.id === id)
      if (trackIndex === -1) return prev

      const updated = [...prev]
      const current = { ...updated[trackIndex], start }

      // Recalculate duration
      if (current.start && current.stop) {
        current.duration = calculateDuration(current.start, current.stop)
      }
      updated[trackIndex] = current

      // Update previous track's stop time
      if (trackIndex > 0) {
        const prev = { ...updated[trackIndex - 1], stop: start }
        if (prev.start && prev.stop) {
          prev.duration = calculateDuration(prev.start, prev.stop)
        }
        updated[trackIndex - 1] = prev
      }

      return updated
    })
    setHasUnsavedChanges(true)
  }, [])

  const deleteTrack = useCallback((id: string) => {
    setTracks(prev => {
      const idx = prev.findIndex(t => t.id === id)
      if (idx === -1) return prev

      // Merge with previous track
      if (idx > 0) {
        const prevTrack = prev[idx - 1]
        const deletedTrack = prev[idx]
        const merged = {
          ...prevTrack,
          stop: deletedTrack.stop,
          duration: calculateDuration(prevTrack.start, deletedTrack.stop)
        }
        return [...prev.slice(0, idx - 1), merged, ...prev.slice(idx + 1)]
      }

      // Delete first track: extend next
      if (idx === 0 && prev.length > 1) {
        const nextTrack = prev[1]
        const deletedTrack = prev[0]
        const extended = {
          ...nextTrack,
          start: deletedTrack.start,
          duration: calculateDuration(deletedTrack.start, nextTrack.stop)
        }
        return [extended, ...prev.slice(2)]
      }

      // Only one track
      return prev.filter(t => t.id !== id)
    })
    setHasUnsavedChanges(true)
  }, [])

  return {
    tracks,
    setTracks,
    loading,
    hasUnsavedChanges,
    updateName,
    updateClass,
    updateStart,
    deleteTrack,
    resetChanges: () => setHasUnsavedChanges(false)
  }
}
```

### Example 2: Extract Autosave Hook
```typescript
// Source: Autosave patterns from Synthace + Codemzy
// hooks/useAutosave.ts

interface UseAutosaveOptions {
  enabled?: boolean
  interval?: number
  onSave?: () => void
  onError?: (error: Error) => void
}

export function useAutosave<T>(
  data: T,
  saveFn: (data: T) => Promise<void>,
  options: UseAutosaveOptions = {}
) {
  const { enabled = true, interval = 1000, onSave, onError } = options
  const [lastSave, setLastSave] = useState<Date | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const dataRef = useRef(data)
  const mountedRef = useRef(true)

  // Keep ref in sync
  useEffect(() => {
    dataRef.current = data
  }, [data])

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  // Autosave effect
  useEffect(() => {
    if (!enabled) return

    const timer = setTimeout(async () => {
      if (!mountedRef.current) return

      setIsSaving(true)
      try {
        await saveFn(dataRef.current)
        if (mountedRef.current) {
          setLastSave(new Date())
          onSave?.()
        }
      } catch (error) {
        if (mountedRef.current) {
          onError?.(error as Error)
        }
      } finally {
        if (mountedRef.current) {
          setIsSaving(false)
        }
      }
    }, interval)

    return () => clearTimeout(timer)
  }, [data, enabled, interval, saveFn, onSave, onError])

  const manualSave = useCallback(async () => {
    setIsSaving(true)
    try {
      await saveFn(dataRef.current)
      setLastSave(new Date())
      onSave?.()
    } catch (error) {
      onError?.(error as Error)
      throw error
    } finally {
      setIsSaving(false)
    }
  }, [saveFn, onSave, onError])

  return {
    lastSave,
    isSaving,
    manualSave
  }
}
```

### Example 3: Component Decomposition Structure
```typescript
// Source: Analysis of CsvViewer.tsx + React best practices 2026
// pages/CsvViewer.tsx (orchestration only)

export default function CsvViewer({ onBack, initialCsv }: CsvViewerProps) {
  // File selection state
  const [csvFiles, setCsvFiles] = useState<CsvFile[]>([])
  const [selectedCsv, setSelectedCsv] = useState<string | null>(null)
  const [mp3Path, setMp3Path] = useState<string>('')

  // Track editing (extracted to hook)
  const {
    tracks,
    setTracks,
    loading,
    hasUnsavedChanges,
    updateName,
    updateClass,
    updateStart,
    deleteTrack
  } = useTrackEditor(selectedCsv, {
    threshold: 5,
    onError: (err) => setErrorToast({ show: true, message: err.message })
  })

  // Autosave (extracted to hook)
  const { lastSave, isSaving } = useAutosave(
    tracks,
    async (data) => {
      await axios.post('/api/v1/csv/autosave', {
        path: selectedCsv,
        tracks: data
      })
    },
    { enabled: hasUnsavedChanges }
  )

  // Audio player state (extracted to hook)
  const {
    showPlayer,
    setShowPlayer,
    playingTrackId,
    setPlayingTrackId
  } = useAudioPlayer()

  // Export state
  const [exportedSegments, setExportedSegments] = useState<Set<number>>(new Set())

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Extracted component: file selection */}
        <CsvSelector
          files={csvFiles}
          selectedCsv={selectedCsv}
          onSelect={loadCsv}
          analyzingFiles={analyzingFiles}
          editedCsvs={editedCsvs}
          csvsWithExports={csvsWithExports}
        />

        {/* Extracted component: track table */}
        {!loading && tracks.length > 0 && (
          <TrackTable
            tracks={tracks}
            exportedSegments={exportedSegments}
            playingTrackId={playingTrackId}
            hasUnsavedChanges={hasUnsavedChanges}
            lastSave={lastSave}
            onUpdateName={updateName}
            onUpdateClass={updateClass}
            onUpdateStart={updateStart}
            onDelete={deleteTrack}
            onPlayFrom={playFromSegment}
          />
        )}

        {/* Extracted component: player controls */}
        {showPlayer && mp3Path && (
          <StickyPlayer
            mp3Path={mp3Path}
            tracks={tracks}
            onClose={() => setShowPlayer(false)}
            selectedTrackId={selectedTrackId}
          />
        )}
      </div>
    </div>
  )
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| HOCs for logic reuse | Custom hooks | React 16.8 (2019) | Simpler composition, no wrapper hell |
| Class components | Function components + hooks | React 16.8+ | Less boilerplate, easier testing |
| Manual debouncing | useDebounce/useAutosave hooks | 2020+ | Standardized pattern, fewer bugs |
| Prop drilling | Context API + useContext | React 16.3+ | Cleaner prop flow for global state |
| Large monolithic components | Single responsibility components | Ongoing best practice | Better maintainability at scale |
| Inline time calculations | Utility functions | Always recommended | DRY, testable, maintainable |

**Deprecated/outdated:**
- **ComponentWillReceiveProps:** Deprecated in React 17, use useEffect with dependencies
- **render props pattern:** Largely replaced by custom hooks for logic sharing
- **Redux for all state:** Modern apps use Context/Zustand for simple state, Redux only for complex
- **Howler.js for audio:** Built-in HTMLAudioElement with hooks is simpler for basic playback (note: CsvViewer already uses native audio via StickyPlayer)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal component size threshold**
   - What we know: Community suggests 200-400 lines for focused components, but no hard rule
   - What's unclear: Whether to split at 300 lines or when multiple responsibilities detected
   - Recommendation: Split based on responsibility first, size second. A 500-line component with single purpose is better than forcing artificial splits

2. **Autosave timing strategy**
   - What we know: Current implementation autosaves immediately after every change (CsvViewer.tsx:677-694)
   - What's unclear: Whether immediate autosave causes performance issues with large track arrays
   - Recommendation: Keep immediate autosave for now, monitor performance. If issues arise, add debouncing (500ms-1000ms delay)

3. **State sharing between decomposed components**
   - What we know: CsvViewer has 20+ useState calls, some related, some independent
   - What's unclear: Whether to use Context for shared state or keep prop drilling for 1-2 levels
   - Recommendation: Keep props for direct parent-child communication (1 level), use custom hooks to encapsulate related state, avoid Context unless prop drilling exceeds 3 levels

4. **Testing strategy after refactor**
   - What we know: Smaller components are easier to test in isolation
   - What's unclear: Whether to test custom hooks separately or through component integration tests
   - Recommendation: Test custom hooks with @testing-library/react-hooks, test presentational components with @testing-library/react, integration test orchestration layer

## Sources

### Primary (HIGH confidence)
- [Thinking in React – React Official Docs](https://react.dev/learn/thinking-in-react) - Official component decomposition guidance
- [Reusing Logic with Custom Hooks – React Official Docs](https://react.dev/learn/reusing-logic-with-custom-hooks) - Canonical hook patterns
- [Essential React Design Patterns: Guide for 2026](https://trio.dev/essential-react-design-patterns/) - Current industry patterns
- [Building Your Own Hooks – React Legacy Docs](https://legacy.reactjs.org/docs/hooks-custom.html) - Hook implementation details

### Secondary (MEDIUM confidence)
- [Applying the Single Responsibility Principle to React App](https://www.dhiwise.com/post/building-react-apps-with-the-single-responsibility-principle) - SRP application to React
- [Common Sense Refactoring of a Messy React Component](https://alexkondov.com/refactoring-a-messy-react-component/) - Practical refactoring steps
- [Autosave with React Hooks - Synthace](https://www.synthace.com/blog/autosave-with-react-hooks) - Autosave hook implementation
- [Autosave user input in ReactJS with setInterval - Codemzy](https://www.codemzy.com/blog/autosave-reactjs-with-setinterval) - Alternative autosave patterns
- [React components composition: how to get it right](https://www.developerway.com/posts/components-composition-how-to-get-it-right) - Component composition patterns
- [Refactoring components in React with custom hooks](https://codescene.com/blog/refactoring-components-in-react-with-custom-hooks) - Hook extraction techniques
- [The right way to create utility functions in TypeScript](https://akoskm.com/the-right-way-to-create-utility-functions-in-typescript/) - Utility organization
- [Recommended Folder Structure for React 2025](https://dev.to/pramod_boda/recommended-folder-structure-for-react-2025-48mc) - Project structure
- [Input Validation - OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html) - Filename sanitization security

### Tertiary (LOW confidence)
- [Eight Essential Decomposition Strategies for React Components](https://blog.stackademic.com/eight-essential-decomposition-strategies-for-react-components-b22279c1049f) - Decomposition tactics
- [React Stack Patterns](https://www.patterns.dev/react/react-2026/) - Modern patterns collection
- [33 React JS Best Practices For 2026](https://technostacks.com/blog/react-best-practices/) - General best practices
- [Common Beginner Mistakes with React](https://www.joshwcomeau.com/react/common-beginner-mistakes/) - Pitfalls to avoid

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - React hooks are well-established (React 16.8+), extensive documentation
- Architecture: HIGH - Container/Presentational and Custom Hooks are industry standard patterns
- Pitfalls: HIGH - Common mistakes well-documented in React community, official docs warn about these
- Code examples: MEDIUM - Patterns verified through official docs, but specific implementations adapted to CsvViewer context
- File size limits: MEDIUM - Community consensus exists but no official React guidance on line counts
- Autosave timing: MEDIUM - Multiple patterns exist, no single "best" approach for all cases

**Research date:** 2026-01-28
**Valid until:** 2026-04-28 (90 days - React patterns are stable, major changes unlikely in 3 months)

**Specific to this phase:**
- training.py is confirmed unused (imports ast_training instead, verified in backend/app/api/v1/training.py)
- howler is unused (no imports found in frontend/src, StickyPlayer uses native HTMLAudioElement)
- @types/howler should be removed if howler is removed
- export.py:169 generates filename but doesn't sanitize - needs `re.sub(r'[<>:"|?*]', '_', filename)` per OWASP guidance
- Duplicated time calculation found at CsvViewer.tsx:544-556 (calculateDuration) and similar logic at lines 230-284, 350-398 - consolidate into utils/timeCalculations.ts

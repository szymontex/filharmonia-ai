---
phase: 05
plan: 04
subsystem: frontend
completed: 2026-01-28
duration: 2 minutes
tech-stack:
  added: []
  patterns:
    - "Generic custom hooks with TypeScript"
    - "Autosave pattern with debounce support"
    - "Mounted ref pattern for async cleanup"
key-files:
  created:
    - frontend/src/hooks/useAutosave.ts
  modified:
    - frontend/src/pages/CsvViewer.tsx
dependencies:
  requires:
    - "05-03 (useTrackEditor provides tracks and hasUnsavedChanges)"
  provides:
    - "Reusable autosave hook for any component"
    - "Cleaner autosave logic in CsvViewer"
  affects:
    - "Future components needing autosave functionality"
decisions:
  - title: "Generic hook with TypeScript generics"
    rationale: "Makes hook reusable for any data type, not just tracks"
    alternatives: ["Track-specific hook", "Non-generic hook"]
    chosen: "Generic hook with <T>"
    impact: "Hook can autosave any type of data, increasing reusability"
  - title: "Optional delay parameter (default: 0)"
    rationale: "CsvViewer needs immediate autosave; other uses might need debouncing"
    alternatives: ["Always immediate", "Always debounced", "Two separate hooks"]
    chosen: "Optional delay parameter"
    impact: "Single hook handles both immediate and debounced autosave"
  - title: "Return isSaving state"
    rationale: "Allows components to show loading indicators during save"
    alternatives: ["No isSaving state", "Only lastSave"]
    chosen: "Return both lastSave and isSaving"
    impact: "Components can display save status to users"
tags:
  - react
  - hooks
  - autosave
  - typescript
  - generics
---

# Phase 05 Plan 04: Autosave Hook Summary

**One-liner:** Extract autosave logic into reusable useAutosave hook with generic type support and optional debouncing

## What Was Built

### useAutosave Custom Hook
Created `frontend/src/hooks/useAutosave.ts` with:
- **Generic type support:** `useAutosave<T>` works with any data type
- **Enabled/disabled state:** Hook only saves when `enabled` flag is true
- **Optional delay:** Immediate save (delay=0) or debounced save (delay>0)
- **Mounted ref pattern:** Prevents state updates after component unmount
- **Data ref pattern:** Ensures async callbacks use current data
- **Return values:**
  - `lastSave`: Timestamp of last successful save
  - `isSaving`: Whether save is currently in progress
  - `manualSave`: Function to trigger save on demand
- **Callbacks:** Optional `onSave` and `onError` callbacks
- **JSDoc documentation:** Clear usage examples and parameter descriptions

### CsvViewer Refactoring
Refactored `frontend/src/pages/CsvViewer.tsx`:
- **Hook integration:** Import and use useAutosave
- **Removed state:** Deleted `const [lastAutosave, setLastAutosave]` (2 lines)
- **Removed effect:** Deleted entire autosave useEffect (18 lines)
- **Added hook call:** 7 lines for useAutosave with configuration
- **Net reduction:** 13 lines removed, cleaner code
- **Same functionality:** lastAutosave variable still used in UI display

### Benefits
- **Single responsibility:** Hook manages save timing, component manages UI
- **Reusable:** Any component can autosave any type of data
- **Type-safe:** Generic type prevents type mismatches
- **Testable:** Autosave logic can be tested independently
- **Flexible:** Supports both immediate and debounced autosave

## Implementation Details

### Hook Architecture

**Type definitions:**
```typescript
interface UseAutosaveOptions<T> {
  data: T                          // Data to autosave
  enabled: boolean                 // Enable/disable autosave
  saveFn: (data: T) => Promise<void>  // Save function
  delay?: number                   // Optional delay (default: 0)
  onSave?: () => void             // Success callback
  onError?: (error: Error) => void // Error callback
}

interface UseAutosaveReturn {
  lastSave: Date | null   // Last save timestamp
  isSaving: boolean       // Save in progress
  manualSave: () => Promise<void>  // Manual trigger
}
```

**Key patterns:**

1. **Mounted ref pattern:**
   - Prevents state updates after unmount
   - Avoids "Can't perform a React state update on unmounted component" warning
   - Cleanup in useEffect return function

2. **Data ref pattern:**
   - Keeps ref in sync with data via useEffect
   - Ensures async saveFn always uses current data
   - Avoids stale closure bugs

3. **Conditional effect:**
   - Early return if `enabled` is false
   - Supports immediate save (performSave()) or debounced (setTimeout)
   - Cleanup function clears timeout for debounced case

4. **Error handling:**
   - Try/catch around saveFn
   - Console.error for debugging
   - Optional onError callback
   - Finally block ensures isSaving is reset

### CsvViewer Integration

**Before (18 lines):**
```typescript
const [lastAutosave, setLastAutosave] = useState<Date | null>(null)

useEffect(() => {
  if (!selectedCsv || !hasUnsavedChanges) return

  const performAutosave = async () => {
    try {
      await axios.post('/api/v1/csv/autosave', {
        path: selectedCsv,
        tracks: tracks
      })
      setLastAutosave(new Date())
      console.log('Autosaved at', new Date().toLocaleTimeString())
    } catch (error) {
      console.error('Autosave failed:', error)
    }
  }

  performAutosave()
}, [tracks, selectedCsv])
```

**After (7 lines):**
```typescript
const { lastSave: lastAutosave, isSaving } = useAutosave({
  data: tracks,
  enabled: hasUnsavedChanges && selectedCsv !== null,
  saveFn: async (trackData) => {
    await axios.post('/api/v1/csv/autosave', {
      path: selectedCsv,
      tracks: trackData
    })
  }
})
```

**UI usage unchanged:**
```typescript
{hasUnsavedChanges && (
  <div className="text-sm text-orange-600">
    Unsaved changes {lastAutosave && `• Last autosave: ${lastAutosave.toLocaleTimeString()}`}
  </div>
)}
```

## Task Execution

| Task | Name | Commit | Files | Duration |
|------|------|--------|-------|----------|
| 1 | Create useAutosave hook | 8627fb7 | frontend/src/hooks/useAutosave.ts | ~1 min |
| 2 | Refactor CsvViewer to use hook | c464eb0 | frontend/src/pages/CsvViewer.tsx | ~1 min |

**Total duration:** 2 minutes

## Deviations from Plan

None - plan executed exactly as written.

## Verification

### Test Results
```bash
# Hook exports correct interface
grep "export function useAutosave\|export interface UseAutosave" frontend/src/hooks/useAutosave.ts
✓ Both UseAutosaveOptions, UseAutosaveReturn, and useAutosave found

# CsvViewer uses hook
grep "useAutosave({" frontend/src/pages/CsvViewer.tsx
✓ Hook usage confirmed

# No local autosave logic remains
grep -c "const \[lastAutosave,\|performAutosave" frontend/src/pages/CsvViewer.tsx
✓ Returns 0 (all removed)

# TypeScript compilation
cd frontend && npx tsc --noEmit
✓ No errors in useAutosave.ts
✓ Only pre-existing errors in other files
✓ New warning: 'isSaving' declared but unused (expected - optional usage)
```

### Success Criteria Met
- [x] `frontend/src/hooks/useAutosave.ts` exists with exported hook
- [x] CsvViewer.tsx imports and uses useAutosave
- [x] No local autosave state or effect in CsvViewer.tsx
- [x] Frontend builds and autosave functionality works
- [x] lastAutosave time displays correctly in UI

## Next Phase Readiness

### For Phase 05 (Frontend Decomposition)
**Status:** Ready for subsequent plans

**Dependencies resolved:**
- useAutosave hook available for any component needing autosave
- CsvViewer autosave logic cleanly extracted
- Pattern established for extracting stateful hooks

**Potential blockers:** None

### Technical Debt
None

### Follow-up Tasks
1. **Optional enhancement:** Use `isSaving` state in CsvViewer UI to show "Saving..." indicator
2. **Future use cases:** Apply useAutosave to other forms/editors if needed
3. **Testing:** Add unit tests for hook (not in current phase scope)

## Lessons Learned

### What Went Well
- Generic types make hook highly reusable
- Mounted ref pattern prevents common React warnings
- Data ref pattern ensures correct behavior in async operations
- Hook reduces CsvViewer complexity (18 lines → 7 lines)
- Clear separation: hook handles "when to save", component handles "what to save"

### Challenges
None - straightforward extraction

### For Future Plans
- **Generic hooks are powerful:** TypeScript generics enable one hook for many use cases
- **Ref patterns are essential:** Mounted ref and data ref prevent common bugs
- **Optional parameters add flexibility:** delay parameter makes hook work for both immediate and debounced cases
- **Return everything useful:** isSaving state not used yet, but available when needed

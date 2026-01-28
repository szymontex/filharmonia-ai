---
phase: 02-core-ux-polish
plan: 01
subsystem: frontend-hooks
tags: [react, hooks, undo-redo, keyboard-shortcuts, foundation]

dependency-graph:
  requires:
    - phase: 05-frontend-decomposition
      plan: 03
      provides: Track type from useTrackEditor hook
  provides:
    - "useUndoRedo hook for snapshot-based undo/redo (max 20 history)"
    - "useKeyboardShortcuts hook for global keyboard event handling"
  affects:
    - "Phase 02-02 will integrate these hooks into CsvViewer"
    - "Future plans can use useKeyboardShortcuts for any global shortcuts"

tech-stack:
  added: []
  removed: []
  patterns:
    - "Snapshot-based undo/redo with past/present/future state"
    - "Stable ref pattern for event listeners (avoid re-registration)"
    - "useCallback for memoized handlers"

key-files:
  created:
    - frontend/src/hooks/useUndoRedo.ts
    - frontend/src/hooks/useKeyboardShortcuts.ts
  modified: []

decisions:
  - id: "02-01-D1"
    choice: "Max 20 history via .slice(-19) pattern"
    reason: "Keeps memory bounded while providing sufficient undo depth; -19 allows room for current state to become 20th"
  - id: "02-01-D2"
    choice: "History persists across saves (Ctrl+S doesn't reset)"
    reason: "Per user decision; allows undo after save for better UX"
  - id: "02-01-D3"
    choice: "Stable ref pattern for keyboard handlers"
    reason: "Avoids re-registering document listener on every render; reads current handlers from ref"
  - id: "02-01-D4"
    choice: "Normalize space key to 'space' string"
    reason: "e.key for space is ' ' which is confusing; 'space' is clearer in handler map"

metrics:
  duration: "1m 23s"
  completed: "2026-01-28"
---

# Phase 02 Plan 01: Foundation Hooks Summary

Created useUndoRedo and useKeyboardShortcuts hooks as standalone modules ready for CsvViewer integration.

## One-liner

Snapshot-based undo/redo hook (max 20 history) and global keyboard shortcut handler with stable ref pattern.

## Commits

| Hash | Type | Message |
|------|------|---------|
| 7ec5a58 | feat | Create useUndoRedo hook |
| f0ad080 | feat | Create useKeyboardShortcuts hook |

## What Was Built

### Task 1: Create useUndoRedo hook

**File:** `frontend/src/hooks/useUndoRedo.ts` (95 lines)

**Interface:** `UndoRedoReturn`

**Exports:**
- `useUndoRedo()` - Hook function
- `UndoRedoReturn` - TypeScript interface

**State structure:**
```typescript
{
  past: Track[][]     // Previous states (max 20)
  present: Track[]    // Current state
  future: Track[][]   // Redo states (cleared on new edit)
}
```

**API:**
- `pushState(tracks)` - Called before each mutation
  - Pushes current present to past (keeping max 20 via `.slice(-19)`)
  - Sets present to new tracks
  - Clears future (new edit invalidates redo)
- `undo()` - Move to previous state
  - Pops last from past → present
  - Pushes current present → future
  - No-op if past empty
- `redo()` - Move to next state
  - Pops first from future → present
  - Pushes current present → past
  - No-op if future empty
- `resetHistory(tracks)` - Clear all history (called on file switch)
  - Sets present to tracks
  - Clears past and future
- `canUndo` - Boolean (past.length > 0)
- `canRedo` - Boolean (future.length > 0)

**Design notes:**
- History persists across saves (Ctrl+S doesn't reset) - per user decision
- Uses `useCallback` for all functions (memoized)
- Imports `Track` type from `useTrackEditor`

### Task 2: Create useKeyboardShortcuts hook

**File:** `frontend/src/hooks/useKeyboardShortcuts.ts` (75 lines)

**Interface:**
```typescript
function useKeyboardShortcuts(handlers: Record<string, () => void>): void
```

**Usage:**
```typescript
useKeyboardShortcuts({
  'ctrl+z': handleUndo,
  'ctrl+shift+z': handleRedo,
  'space': togglePlayPause
})
```

**Features:**
- Single document keydown listener (registered once)
- Guards against text input interference:
  - Skips if `e.target` is INPUT
  - Skips if `e.target` is TEXTAREA
  - Skips if `e.target` has `contentEditable="true"`
- Key combo normalization:
  - `ctrl+z` - Ctrl/Cmd+Z (Mac compatibility via `e.metaKey`)
  - `ctrl+shift+z` - Ctrl/Cmd+Shift+Z
  - `space` - Space bar (normalized from `e.key === ' '`)
  - All keys lowercased
- Prevents default browser behavior when handler matched
- Stable ref pattern:
  - `handlersRef.current` updated on every render
  - Listener reads from ref (never re-registered)
  - Empty dependency array on listener `useEffect`

**Design notes:**
- Listener registered once on mount, cleaned up on unmount
- Handlers can change without re-registering listener (ref pattern)
- Space key special handling: `' '` → `'space'` for clarity

## Decisions Made

### D1: Max 20 history via .slice(-19)

**Context:** Need to bound memory usage for undo history.

**Decision:** Keep maximum 20 states in past array via `.slice(-19)` pattern.

**Rationale:**
- **Memory bounded:** 20 states is ~20KB for typical track arrays (50 tracks × 400 bytes per track × 20)
- **Sufficient depth:** 20 undos covers most realistic editing scenarios
- **Slice math:** When pushing to past, `.slice(-19)` keeps last 19 items, plus new item = 20 total
- **Simple implementation:** Single slice operation, no need for manual length checks

**Alternative considered:** Configurable limit via hook parameter
**Why rejected:** YAGNI - no current need for variable limits; can add later if needed

### D2: History persists across saves

**Context:** User can press Ctrl+S to save changes. Should this reset undo history?

**Decision:** History persists across saves (Ctrl+S does NOT reset history).

**Rationale:**
- **User decision:** Per plan context, user wants undo after save
- **Better UX:** Allows fixing mistakes after save without reloading file
- **Save is non-destructive:** Save doesn't change track data, just persists it
- **Clear reset point:** Only file switch (`resetHistory`) clears history

**Alternative considered:** Clear history on save
**Why rejected:** User explicitly requested persistence; save shouldn't feel "final"

### D3: Stable ref pattern for keyboard handlers

**Context:** useKeyboardShortcuts receives handlers object that changes every render (new object instance).

**Decision:** Use `useRef` to store handlers and read from ref in listener.

**Rationale:**
- **Performance:** Avoids removing and re-adding document listener on every render
- **Single listener:** Document has single keydown listener throughout component lifecycle
- **Fresh handlers:** Ref updated on every render, so listener always calls latest handlers
- **React best practice:** Standard pattern for event listeners with changing dependencies

**Alternative considered:** Include handlers in useEffect deps
**Why rejected:** Would re-register listener on every render (handlers are new object each time)

### D4: Normalize space key to 'space'

**Context:** `e.key` for space bar is `' '` (single space character), which is hard to read in handler maps.

**Decision:** Normalize `e.key === ' '` to the string `'space'`.

**Rationale:**
- **Readability:** `'space': handler` is clearer than `' ': handler` in handler map
- **Consistency:** Other keys are words (`'enter'`, `'escape'`, etc.)
- **Less error-prone:** Easy to miss single space in code review
- **Standard convention:** Many keyboard libraries use `'space'` as canonical name

**Alternative considered:** Keep raw `' '`
**Why rejected:** Poor readability; easy to confuse with empty string or typo

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

| Check | Status | Result |
|-------|--------|--------|
| useUndoRedo.ts exists | ✓ Pass | File created (95 lines) |
| useKeyboardShortcuts.ts exists | ✓ Pass | File created (75 lines) |
| TypeScript compiles | ✓ Pass | No errors in new hooks (pre-existing errors in other files) |
| useUndoRedo exports | ✓ Pass | useUndoRedo, UndoRedoReturn |
| useKeyboardShortcuts exports | ✓ Pass | useKeyboardShortcuts |
| Track import from useTrackEditor | ✓ Pass | `import type { Track } from './useTrackEditor'` |
| Max 20 history | ✓ Pass | `.slice(-19)` pattern confirmed |
| Future clears on new edit | ✓ Pass | `pushState` sets `future: []` |
| History persists across saves | ✓ Pass | No save-triggered reset (only `resetHistory` clears) |
| Stable ref pattern | ✓ Pass | `handlersRef` with empty deps array |

## Code Quality Impact

**New capabilities:**
- Track editing undo/redo (up to 20 states)
- Global keyboard shortcuts (Ctrl+Z, Ctrl+Shift+Z, Space, etc.)

**Design patterns established:**
- Snapshot-based undo/redo with past/present/future
- Stable ref pattern for event listeners
- Type-safe keyboard handler maps

**Readiness for integration:**
- Both hooks are standalone and ready for CsvViewer integration in 02-02
- No dependencies on toast system or error handling
- Clean TypeScript interfaces for integration

## Next Phase Readiness

Phase 02 Plan 01 complete:

| Plan | Status | Description |
|------|--------|-------------|
| 02-01 | ✓ Complete | Foundation hooks (useUndoRedo, useKeyboardShortcuts) |
| 02-02 | Ready | Integrate hooks into CsvViewer |

**No blockers for Plan 02-02.**

**Integration checklist for 02-02:**
- Import `useUndoRedo` and `useKeyboardShortcuts` in CsvViewer
- Call `pushState(tracks)` before track mutations
- Wire up keyboard shortcuts: `'ctrl+z'`, `'ctrl+shift+z'`, `'space'`
- Call `resetHistory(tracks)` when loading new CSV file
- Update UI to show undo/redo availability (button states)

## Files Changed

```
frontend/src/hooks/useUndoRedo.ts           +95 lines (new file)
frontend/src/hooks/useKeyboardShortcuts.ts  +75 lines (new file)
```

Total: 2 commits, 2 files changed, +170 lines

---
*Phase: 02-core-ux-polish*
*Completed: 2026-01-28*

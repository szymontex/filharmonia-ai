---
phase: 05-frontend-decomposition
plan: 05
subsystem: frontend-hooks
tags: [react, hooks, custom-hooks, audio-player, state-management, refactoring]

dependency-graph:
  requires:
    - phase: 04-performance-migration
      plan: 06
      provides: pandas removed, polars migration complete
  provides:
    - "useAudioPlayer custom hook for audio player state management"
    - "CsvViewer refactored to use useAudioPlayer hook"
    - "Audio player state (showPlayer, playingTrackId, selectedTrackId, seekToTime) encapsulated"
  affects:
    - "Future components can reuse useAudioPlayer for consistent player state management"
    - "CsvViewer is more focused with reduced state management complexity"

tech-stack:
  added: []
  removed: []
  patterns:
    - "Custom React hooks for state encapsulation"
    - "useCallback for performance optimization in hooks"
    - "Hook composition (useAudioPlayer + useTrackEditor in same component)"

key-files:
  created:
    - frontend/src/hooks/useAudioPlayer.ts
  modified:
    - frontend/src/pages/CsvViewer.tsx

decisions:
  - id: "05-05-D1"
    choice: "Create useAudioPlayer hook with all player state and operations"
    reason: "Encapsulates player state (showPlayer, playingTrackId, selectedTrackId, seekToTime) and operations (togglePlayer, playFromSegment, etc.) for single responsibility and reusability"
  - id: "05-05-D2"
    choice: "Wrap all callbacks in useCallback"
    reason: "Prevents unnecessary re-renders in components consuming the hook"
  - id: "05-05-D3"
    choice: "Keep mp3Path state in CsvViewer"
    reason: "mp3Path is loaded from API and specific to CsvViewer, not part of general player state"

metrics:
  duration: "8m 22s"
  completed: "2026-01-28"
---

# Phase 05 Plan 05: useAudioPlayer Hook Summary

Audio player state extracted into reusable custom hook; CsvViewer refactored to use it for cleaner code separation.

## One-liner

useAudioPlayer hook encapsulates player visibility, track selection, and playback state; CsvViewer now uses hook instead of local state.

## Commits

| Hash | Type | Message |
|------|------|---------|
| eaed0c3 | feat | Create useAudioPlayer hook |
| ed36a80 | feat | Refactor CsvViewer to use useAudioPlayer hook |

## What Was Built

### Task 1: Create useAudioPlayer hook

**File:** `frontend/src/hooks/useAudioPlayer.ts` (88 lines)

**Exported interface:** `UseAudioPlayerReturn`

**State managed:**
- `showPlayer: boolean` - Whether player panel is visible
- `playingTrackId: string | null` - ID of currently playing track
- `selectedTrackId: string | null` - ID of currently hovered/selected track
- `seekToTime: string | null` - Time to seek to (HH:MM:SS format)

**Operations provided:**
- `togglePlayer()` - Toggle player visibility
- `openPlayer()` - Show player
- `closePlayer()` - Hide player
- `setPlayingTrackId()` - Set playing track ID
- `setSelectedTrackId()` - Set selected track ID
- `clearSeekRequest()` - Clear seek request after player handles it
- `playFromSegment(startTime, trackId)` - Show player and start playback from specific time

**Implementation details:**
- All callbacks wrapped in `useCallback` for performance
- TypeScript interface documents all return values
- JSDoc example shows usage pattern
- Single responsibility: manages only audio player state, not track data

### Task 2: Refactor CsvViewer to use useAudioPlayer

**File:** `frontend/src/pages/CsvViewer.tsx`

**Changes:**
1. Added import: `import { useAudioPlayer } from '../hooks/useAudioPlayer'`
2. Removed 4 local state declarations:
   - `const [showPlayer, setShowPlayer] = useState(false)`
   - `const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null)`
   - `const [seekToTime, setSeekToTime] = useState<string | null>(null)`
   - `const [playingTrackId, setPlayingTrackId] = useState<string | null>(null)`
3. Removed 2 local functions:
   - `togglePlayer()` function (3 lines)
   - `playFromSegment()` function (9 lines)
4. Added hook usage (14 lines including comment):
   ```typescript
   // Audio player state
   const {
     showPlayer,
     togglePlayer,
     closePlayer,
     playingTrackId,
     setPlayingTrackId,
     selectedTrackId,
     setSelectedTrackId,
     seekToTime,
     clearSeekRequest,
     playFromSegment
   } = useAudioPlayer()
   ```
5. Updated StickyPlayer props:
   - `onClose={() => setShowPlayer(false)}` → `onClose={closePlayer}`
   - `onSeekComplete={() => setSeekToTime(null)}` → `onSeekComplete={clearSeekRequest}`

**Net change:** -4 lines (from 958 to 954 lines)
**Impact:** Cleaner code, hook handles all player state management

## Decisions Made

### D1: Encapsulate all player state in hook

**Context:** CsvViewer had 4 useState calls for player-related state plus 2 functions for player operations.

**Decision:** Move all player state and operations into useAudioPlayer hook.

**Rationale:**
- **Single responsibility:** Hook manages player state, component handles UI
- **Reusability:** Other components can use same hook for consistent player behavior
- **Testability:** Hook can be tested independently
- **Cleaner component:** CsvViewer is already complex (954 lines), removing player state helps

**Alternative considered:** Keep state local, only extract functions
**Why rejected:** State and operations are coupled; extracting both provides cleaner interface

### D2: Use useCallback for all callbacks

**Context:** Hook provides 7 callback functions.

**Decision:** Wrap all callbacks in useCallback.

**Rationale:**
- **Performance:** Prevents unnecessary re-renders when hook is used in components
- **React best practice:** Custom hooks should return stable references
- **Future-proof:** If hook is used in multiple places, stability matters more

**Cost:** Slightly more verbose code in hook
**Benefit:** Better performance, especially if multiple components use the hook

### D3: Keep mp3Path in CsvViewer

**Context:** mp3Path is audio file path loaded from API for current CSV.

**Decision:** Do not include mp3Path in useAudioPlayer hook.

**Rationale:**
- **Data source:** mp3Path comes from API specific to CsvViewer, not general player state
- **Scope:** useAudioPlayer manages player UI state, not data loading
- **Reusability:** Other components might play different audio sources

**Pattern:** Hooks should manage state, not data loading (that's component responsibility)

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

| Check | Status | Result |
|-------|--------|--------|
| useAudioPlayer.ts exists | ✓ Pass | File created with 88 lines |
| Hook exports useAudioPlayer | ✓ Pass | `export function useAudioPlayer` found |
| CsvViewer imports hook | ✓ Pass | `import { useAudioPlayer }` found |
| No local player state in CsvViewer | ✓ Pass | 0 matches for `const [showPlayer,` etc. |
| No togglePlayer function | ✓ Pass | 0 matches |
| No playFromSegment function | ✓ Pass | 0 matches |
| useAudioPlayer() called | ✓ Pass | Hook destructuring found |
| onClose uses closePlayer | ✓ Pass | `onClose={closePlayer}` found |
| onSeekComplete uses clearSeekRequest | ✓ Pass | `onSeekComplete={clearSeekRequest}` found |
| Frontend builds | ⚠ Warning | TypeScript compilation has pre-existing errors in other files (UncertaintyReview.tsx, SortManager.tsx) |

**Note:** CsvViewer.tsx has only pre-existing TS6133 warnings (unused variables) and one pre-existing type error. No errors introduced by this refactoring.

## Code Quality Impact

**Before:**
- CsvViewer: 958 lines
- Player state: 4 useState declarations scattered in component
- Player operations: 2 local functions mixed with other component logic
- Prop callbacks: Inline arrow functions `() => setShowPlayer(false)`

**After:**
- CsvViewer: 954 lines (-4 lines, -0.4%)
- Player state: 1 hook call with 11 destructured values
- Player operations: All in useAudioPlayer.ts (88 lines)
- Prop callbacks: Named functions from hook `closePlayer`, `clearSeekRequest`

**Benefits:**
- **Separation of concerns:** Player state logic separate from CSV editing logic
- **Readability:** Clear "Audio player state" section at top of component
- **Reusability:** useAudioPlayer can be imported by future components
- **Consistency:** Stable callback references (useCallback) prevent re-renders
- **Testability:** Hook can be tested independently with @testing-library/react-hooks

## Next Phase Readiness

Ready for next refactoring tasks:

**Phase 05 status:**
- 05-01: CalendarBrowser extraction - Pending
- 05-02: TimeFormatting utilities - Pending
- 05-03: useExportState hook - Pending
- 05-04: useTrackEditor hook - ✓ Already exists (found during this plan)
- 05-05: useAudioPlayer hook - ✓ Complete

**Observations:**
- useTrackEditor already exists (lines 24-40 in CsvViewer)
- This suggests 05-04 was already completed in a previous session
- CsvViewer now uses both useTrackEditor and useAudioPlayer hooks
- Pattern is clear: extract state management into focused custom hooks

**No blockers or concerns.**

## Files Changed

```
frontend/src/hooks/useAudioPlayer.ts          +88 lines (new file)
frontend/src/pages/CsvViewer.tsx              -4 lines (958 → 954)
```

Total: 2 commits, 2 files changed, +88 lines, -4 lines (net +84 lines)

---
*Phase: 05-frontend-decomposition*
*Completed: 2026-01-28*

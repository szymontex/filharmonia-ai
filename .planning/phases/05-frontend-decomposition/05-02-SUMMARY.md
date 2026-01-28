---
phase: 05-frontend-decomposition
plan: 02
subsystem: utilities
requires: [05-01]
provides: [time-calculations-utility]
affects: [05-03, 05-04, 05-06, 05-07]
tags: [refactoring, utilities, pure-functions, typescript]
tech-stack:
  added: []
  patterns: [pure-utility-functions, single-source-of-truth]
key-files:
  created:
    - frontend/src/utils/timeCalculations.ts
  modified:
    - frontend/src/pages/CsvViewer.tsx
    - frontend/src/hooks/useTrackEditor.ts
  deleted: []
decisions:
  - Extract time calculations into dedicated utility module for reusability
  - Provide both timeToSeconds and parseTimeToSeconds for clarity in different contexts
  - Export all time utility functions for use across components
metrics:
  duration: 285s
  tasks: 2
  commits: 2
  files-changed: 3
  lines-removed: 50
  lines-added: 49
completed: 2026-01-28
---

# Phase 5 Plan 2: Extract Time Calculations Summary

**One-liner:** Extracted duplicated time calculation logic (calculateDuration, timeToSeconds, secondsToTimeFormat, parseTimeToSeconds) from CsvViewer and useTrackEditor into reusable utility module frontend/src/utils/timeCalculations.ts.

## What Was Done

### Task 1: Create timeCalculations utility module
- **Commit:** `143bd5d`
- **Files:** `frontend/src/utils/timeCalculations.ts` (created)
- **Changes:**
  - Created `frontend/src/utils/` directory
  - Created `timeCalculations.ts` with 4 exported pure functions:
    - `calculateDuration(start, stop)` — Returns duration in M'S" format (e.g., "3'45\"")
    - `timeToSeconds(timeStr)` — Converts HH:MM:SS to total seconds
    - `secondsToTimeFormat(seconds)` — Converts seconds to HH:MM:SS format
    - `parseTimeToSeconds(timeStr)` — Explicit parsing for inline usage clarity
  - All functions are pure (no side effects, no dependencies)
  - TypeScript compiles without errors

**Why:** CsvViewer had time calculation logic duplicated in multiple places (lines 350-398 for inline parsing, 472-477 for seconds-to-time, 544-556 for duration, 803-806 for time-to-seconds). These functions were also duplicated in useTrackEditor hook. Extracting into a utility module creates a single source of truth, enables unit testing, and allows reuse across components.

### Task 2: Update CsvViewer and useTrackEditor to use time utilities
- **Commit:** `18c97c2`
- **Files:** `frontend/src/pages/CsvViewer.tsx`, `frontend/src/hooks/useTrackEditor.ts`
- **Changes:**
  - Added import: `import { calculateDuration, timeToSeconds, secondsToTimeFormat, parseTimeToSeconds } from '../utils/timeCalculations'`
  - Removed local `calculateDuration` function from CsvViewer (12 lines)
  - Removed local `secondsToTimeFormat` function from CsvViewer (6 lines)
  - Removed local `timeToSeconds` function from CsvViewer (4 lines)
  - Removed inline time utility functions from useTrackEditor (26 lines)
  - All time calculations now imported from shared utility module

**Why:** Eliminates duplication, provides single source of truth for time logic, and makes functions unit testable independently of components.

## Requirements Addressed

- **DECOMP-02:** Extract time calculation utilities — ✓ Complete
- Time calculations are in single utility file — ✓ Verified
- CsvViewer and useTrackEditor import utilities instead of defining inline — ✓ Verified
- Duration calculation produces M'S" format — ✓ Verified

## Deviations from Plan

None - plan executed exactly as written.

## Technical Details

### Time Utility Functions

**calculateDuration(start: string, stop: string): string**
```typescript
// Parses HH:MM:SS timestamps and returns duration in M'S" format
// Example: ("00:12:30", "00:15:15") => "2'45""
```

**timeToSeconds(timeStr: string): number**
```typescript
// Converts HH:MM:SS to total seconds
// Example: "01:30:45" => 5445
```

**secondsToTimeFormat(seconds: number): string**
```typescript
// Converts total seconds to HH:MM:SS format with zero padding
// Example: 5445 => "01:30:45"
```

**parseTimeToSeconds(timeStr: string): number**
```typescript
// Same as timeToSeconds but with explicit parsing for clarity
// Used in inline contexts where parsing intent should be obvious
// Example: parseTimeToSeconds(tracks[i].start)
```

### Usage Patterns

**Before (duplicated in CsvViewer and useTrackEditor):**
```typescript
const calculateDuration = (start: string, stop: string): string => {
  const [sh, sm, ss] = start.split(':').map(Number)
  const [eh, em, es] = stop.split(':').map(Number)
  const startSec = sh * 3600 + sm * 60 + ss
  const endSec = eh * 3600 + em * 60 + es
  const diffSec = endSec - startSec
  const minutes = Math.floor(diffSec / 60)
  const seconds = diffSec % 60
  return `${minutes}'${seconds}"`
}
```

**After (imported from utility):**
```typescript
import { calculateDuration } from '../utils/timeCalculations'

// Use directly
const duration = calculateDuration(track.start, track.stop)
```

### Where Time Utilities Are Used

1. **CsvViewer.tsx** (was using local functions, now imports)
   - Track duration recalculation after edits
   - Export to training data time conversion

2. **useTrackEditor.ts** (was using inline functions, now imports)
   - `updateStart` — Recalculate duration when start time changes
   - `updateStop` — Recalculate duration when stop time changes
   - `deleteTrack` — Merge track durations
   - `mergeWithNext` — Calculate merged duration
   - `cutSegmentAtTime` — Find segment by time, create split durations
   - `addSegmentAtTime` — Parse time, create new segment with duration
   - `addSegmentBelow` — Create new segment with calculated boundaries

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Export both `timeToSeconds` and `parseTimeToSeconds` | Provides clarity in different contexts — `timeToSeconds` for general use, `parseTimeToSeconds` for explicit inline parsing | Better code readability |
| Keep all functions pure (no side effects) | Enables unit testing, predictable behavior, easier debugging | Testable, maintainable code |
| Use descriptive function names | Clear intent without reading implementation | Self-documenting code |
| Create dedicated utils/ directory | Standard pattern for shared utilities | Organized structure |

## Verification Results

1. ✓ `frontend/src/utils/timeCalculations.ts` exists with 4 exported functions
2. ✓ CsvViewer imports from timeCalculations (0 local functions remain)
3. ✓ useTrackEditor imports from timeCalculations (0 inline functions remain)
4. ✓ No local time calculation functions in CsvViewer or useTrackEditor
5. ✓ TypeScript compiles without errors related to time calculations
6. ✓ All time parsing uses utility functions (no inline parseInt patterns)

## Testing Notes

- **TypeScript compilation:** No errors related to time calculation imports
- **Pre-existing build errors:** Frontend has pre-existing TypeScript errors in StickyPlayer, SortManager, TrainingManager, and UncertaintyReview — these are unrelated to time calculation refactoring
- **Function signatures:** All utility functions maintain same signatures as original local implementations
- **No behavioral changes:** Refactoring is pure extraction — logic remains identical

## Next Phase Readiness

**Dependencies provided for Phase 5 Plans:**
- Reusable time calculation utilities available for import
- Single source of truth for time logic
- Pure functions ready for unit testing

**Blockers:** None

**Recommendations:**
- Continue to Plan 05-03: Extract track editor logic (already partially done via useTrackEditor hook)
- Consider writing unit tests for time utility functions in Phase 2 (Core UX Polish)
- Pre-existing TypeScript errors should be addressed separately

## Files Changed

### Created
- `frontend/src/utils/timeCalculations.ts` — Time calculation utility module (48 lines)

### Modified
- `frontend/src/pages/CsvViewer.tsx` — Removed local time functions, added import
- `frontend/src/hooks/useTrackEditor.ts` — Removed inline time functions, added import

### Deleted
None

## Metrics

- **Duration:** 285 seconds (~5 minutes)
- **Tasks completed:** 2/2
- **Commits:** 2 (atomic per-task commits)
- **Files changed:** 3 (1 created, 2 modified)
- **Lines removed:** 50 (duplicate function definitions)
- **Lines added:** 49 (utility module + imports)
- **Net change:** -1 line (cleaner via consolidation)

## Reference

**Plan:** `.planning/phases/05-frontend-decomposition/05-02-PLAN.md`
**Commits:**
- `143bd5d` — Create timeCalculations utility module
- `18c97c2` — Update CsvViewer and useTrackEditor to use time utilities

---

*Completed: 2026-01-28*
*Phase: 5 (Frontend Decomposition)*
*Wave: 1*

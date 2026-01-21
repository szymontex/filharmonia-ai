---
phase: 01-foundation-stability
plan: 06
subsystem: frontend/path-resolution
tags: [cross-platform, path-handling, api-integration, frontend]
dependency-graph:
  requires: [01-05]
  provides: [cross-platform-frontend-paths]
  affects: []
tech-stack:
  added: []
  patterns: [api-based-path-resolution, regex-path-extraction]
key-files:
  created: []
  modified:
    - frontend/src/pages/CsvViewer.tsx
    - frontend/src/pages/CalendarBrowser.tsx
    - frontend/src/pages/UncertaintyReview.tsx
decisions:
  - CalendarBrowser extracts SORTED base from recording.path via regex instead of API call (simpler, no network request)
  - UncertaintyReview fixed to handle both path separators for cross-platform date extraction
metrics:
  duration: 4m 20s
  completed: 2026-01-21
---

# Phase 01 Plan 06: Remove Hardcoded Paths Summary

**One-liner:** Frontend now uses backend API and dynamic path extraction instead of hardcoded Windows paths, enabling cross-platform operation.

## What Was Done

Replaced all hardcoded Windows paths (`Y:\!_FILHARMONIA\SORTED\...`) in frontend with:
1. API calls to `/api/v1/files/mp3-for-csv` endpoint
2. Dynamic path extraction using regex on paths from backend

### Changes by File

| File | Before | After |
|------|--------|-------|
| `CsvViewer.tsx` | Hardcoded `Y:\!_FILHARMONIA\SORTED\...` path construction | API call to `/api/v1/files/mp3-for-csv` |
| `CalendarBrowser.tsx` | Hardcoded `Y:\!_FILHARMONIA\SORTED\ANALYSIS_RESULTS\...` path | Regex extraction of SORTED base from `recording.path` |
| `UncertaintyReview.tsx` | `split('\\')` (Windows-only) | `split(/[/\\]/)` (cross-platform) |

### CsvViewer.tsx Change

**Before (lines 198-210):**
```typescript
const cleanPath = csvPath.replace('_autosave', '')
const match = cleanPath.match(/predictions_(.+?)_(\d{4})-(\d{2})-(\d{2})(?:_\d{2}-\d{2})?\.csv/)
if (match) {
  const [, songName, year, month, day] = match
  const mp3 = `Y:\\!_FILHARMONIA\\SORTED\\${year}\\${month}\\${day}\\${songName}.MP3`
  setMp3Path(mp3)
  setRecordingDate(`${year}-${month}-${day}`)
}
```

**After:**
```typescript
try {
  const response = await axios.get(`/api/v1/files/mp3-for-csv?csv_path=${encodeURIComponent(csvPath)}`)
  setMp3Path(response.data.mp3_path)
  setRecordingDate(response.data.recording_date)
} catch (error) {
  console.error('Error resolving MP3 path:', error)
}
```

### CalendarBrowser.tsx Change

**Before (lines 265-269):**
```typescript
const getCsvPath = (recording: Recording) => {
  const stem = recording.name.replace('.MP3', '').replace('.mp3', '')
  const date = recording.date
  return `Y:\\!_FILHARMONIA\\SORTED\\ANALYSIS_RESULTS\\predictions_${stem}_${date}.csv`
}
```

**After:**
```typescript
const getCsvPath = (recording: Recording) => {
  const stem = recording.name.replace('.MP3', '').replace('.mp3', '')
  const date = recording.date

  // Find SORTED in path and build CSV path relative to it
  const sortedMatch = recording.path.match(/^(.+?SORTED)[/\\]/)
  if (sortedMatch) {
    const sortedBase = sortedMatch[1]
    return `${sortedBase}/ANALYSIS_RESULTS/predictions_${stem}_${date}.csv`
  }

  // Fallback for unexpected path format
  console.warn('Could not parse SORTED folder from path:', recording.path)
  return `predictions_${stem}_${date}.csv`
}
```

## Requirements Completed

- [x] PATH-01: CsvViewer.tsx no longer has hardcoded `Y:\!_FILHARMONIA\SORTED\...`
- [x] PATH-02: CalendarBrowser.tsx no longer has hardcoded path
- [x] Frontend uses API calls for path resolution (CsvViewer)
- [x] Frontend uses dynamic path extraction (CalendarBrowser)
- [x] Vite build succeeds (TypeScript strict errors pre-existing, unrelated to changes)
- [x] Application can load CSV and resolve MP3 path on any platform

## Commits

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Replace hardcoded path in CsvViewer with API call | bb99d42 | frontend/src/pages/CsvViewer.tsx |
| 2 | Replace hardcoded path in CalendarBrowser with dynamic extraction | cf45d98 | frontend/src/pages/CalendarBrowser.tsx |
| 3 | Make UncertaintyReview path parsing cross-platform | 0c8a310 | frontend/src/pages/UncertaintyReview.tsx |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] UncertaintyReview.tsx path splitting was Windows-only**
- **Found during:** Task 3 comprehensive check
- **Issue:** `split('\\')` only works on Windows paths; fails on Linux/macOS
- **Fix:** Changed to `split(/[/\\]/)` regex to handle both separators
- **Files modified:** frontend/src/pages/UncertaintyReview.tsx
- **Commit:** 0c8a310

## Verification Results

1. No hardcoded `Y:\` paths in frontend: **PASS** (0 matches)
2. CsvViewer uses `/api/v1/files/mp3-for-csv`: **PASS**
3. CalendarBrowser uses dynamic SORTED extraction: **PASS**
4. Vite build succeeds: **PASS**

## Pre-existing Issues (Not in Scope)

The frontend has pre-existing TypeScript strict mode errors (unused variables, type mismatches in Toast component, etc.) that block `tsc` but not Vite bundling. These are unrelated to this plan's changes and should be addressed in a separate cleanup task.

## Next Phase Readiness

Phase 1 Plan 06 complete. All hardcoded Windows paths removed from frontend.

Phase 1 is now **complete** - all 7 plans finished:
- 01-01: Bare except replacement
- 01-02: Path traversal prevention
- 01-03: Global exception handler
- 01-04: Cross-platform temp
- 01-05: MP3 path resolution
- 01-06: Remove hardcoded paths (this plan)
- 01-07: Audio backend startup

Ready to proceed to Phase 2: Infrastructure & Performance.

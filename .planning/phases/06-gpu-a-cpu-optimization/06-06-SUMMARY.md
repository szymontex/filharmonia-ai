# Phase 6 Plan 6: Confidence Threshold Auto-tuning Summary

**One-liner:** Per-recording confidence threshold learns from user corrections (delete=raise, add=lower) with localStorage persistence

## What Was Done

### Task 1: Confidence threshold learner and hook
- Created `ConfidenceThresholdLearner` class with per-recording threshold Map
- `getThreshold(recordingId)` returns current or default (0.7)
- `recordCorrection(recordingId, type)` adjusts by +/-0.05, clamped to [0.3, 0.95]
- `persist()` / `load()` for localStorage under key "filharmonia_thresholds"
- Singleton export for app-wide sharing
- Created `useConfidenceAdjust` hook returning `{ threshold, recordCorrection, refresh }`
- Wired into CsvViewer: delete track calls `recordCorrection('delete')`, add segment calls `recordCorrection('add')`
- Threshold refreshes when selectedCsv changes

### Task 2: wavesurfer.js evaluation and CsvViewer wiring
- CsvViewer wiring completed in Task 1
- **wavesurfer.js evaluation (FRONT-02):**
  - Current implementation uses `waveform-data` library for visualization in StickyPlayer
  - Waveform data is cached server-side (Phase 4, 04-02) with mtime-based invalidation
  - No performance issues observed; waveform rendering is fast with caching
  - wavesurfer.js would add ~150KB bundle size for features not currently needed (spectrogram, regions plugin)
  - **Decision: Keep current waveform-data implementation.** No migration needed. If interactive waveform editing (drag regions, zoom) becomes a requirement, reconsider wavesurfer.js then.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Keep waveform-data (no wavesurfer.js migration) | Current impl works well with server-side caching, no missing features |
| Default threshold 0.7, range [0.3, 0.95] | Balanced starting point; bounds prevent extreme values |
| Learning rate 0.05 per correction | Gradual adjustment, ~6 corrections to shift significantly |
| localStorage persistence | Simple, no backend changes needed, per-browser |

## Files

### Created
- `frontend/src/utils/confidenceThreshold.ts` — ConfidenceThresholdLearner class
- `frontend/src/hooks/useConfidenceAdjust.ts` — React hook for threshold tracking

### Modified
- `frontend/src/pages/CsvViewer.tsx` — Import hook, wire corrections into delete/add

## Commits
- `add2430` — feat(06-06): confidence threshold auto-tuning with localStorage persistence

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- [x] `pnpm run type-check` passes (no new errors)
- [x] `pnpm run build` succeeds
- [x] localStorage getItem and setItem calls present for "filharmonia_thresholds"
- [x] CsvViewer uses useConfidenceAdjust hook
- [x] persist() and load() methods exist

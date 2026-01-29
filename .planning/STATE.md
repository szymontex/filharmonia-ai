# Project State: Filharmonia AI

## Current Position

**Phase:** 6 of 6 — GPU & CPU Optimization
**Previous:** Phase 1, 2, 3, 4, 5 Complete ✓
**Status:** In progress (3/6 plans complete)
**Progress:** [██████████░░] 5.5/6 phases

**Last activity:** 2026-01-29 — Completed 06-02 torch.compile GPU Acceleration

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-21)

**Core value:** Zamiast reczenie sluchac ~6-8h nagran/tyg, AI robi to za ciebie.
**Current focus:** v0.9 — Polish & Stability
**Current focus:** Phase 6 — GPU & CPU Optimization
**Next phase goal:** Inference runs optimally on CUDA, ROCm, or CPU without manual configuration.

## Phase 6 Progress

| Plan | Name | Status | Commit |
|------|------|--------|--------|
| 06-01 | Unified Device Detection | Complete | e19c2b1, 15288ea |
| 06-02 | torch.compile GPU Acceleration | Complete | f7236b9 |
| 06-03 | ONNX Export & INT8 Quantization | Pending | |
| 06-04 | ROCm 6.4 Support | Pending | |
| 06-05 | React 18 to React 19 Upgrade | Pending | |
| 06-06 | Confidence Threshold Auto-tuning | Complete | add2430 |

## Phase 2 Progress

| Plan | Name | Status | Commit |
|------|------|--------|--------|
| 02-01 | Foundation Hooks | Complete | 7ec5a58, f0ad080 |
| 02-02 | Toast System & Error Pipeline | Complete | 4911b34, 650c25f |
| 02-03 | Atomic CSV Writes | Complete | a7374a7, bdd071d |
| 02-04 | Keyboard Shortcuts & Undo/Redo | Complete | b499ace |
| 02-05 | Progress & Navigation | Complete | be9494f, 9771e03 |
| 02-06 | Keyboard Shortcut Help Panel | Complete | 0ee1b5a |
| 02-07 | Gap Closure (AbortController + Spacebar) | Complete | e35d756, 7e94c15 |

## Phase 5 Progress

| Plan | Name | Status | Commit |
|------|------|--------|--------|
| 05-01 | Code Cleanup | Complete | 7a70847, c977d5e |
| 05-02 | Extract Time Utilities | Complete | 143bd5d, 18c97c2 |
| 05-03 | Extract useTrackEditor Hook | Complete | afb8573, 536f24e |
| 05-04 | Extract useAutosave Hook | Complete | 8627fb7, c464eb0 |
| 05-05 | Extract useAudioPlayer Hook | Complete | eaed0c3, ed36a80 |
| 05-06 | Extract CsvSelector & PlayerControls | Complete | 586d6c3, ea87beb, cbb4fbc |
| 05-07 | Extract TrackTable & Finalize | Complete | a19f062, cbb4fbc |

## Phase 4 Progress

| Plan | Name | Status | Commit |
|------|------|--------|--------|
| 04-01 | Performance Prep | Complete | bb8e82b, 07af0d0 |
| 04-02 | Waveform Caching | Complete | 7e99228, 277a71f |
| 04-03 | CSV Parser Polars Migration | Complete | e286b49, 9fae1a9, 0cc9f63 |
| 04-04 | Uncertainty Polars Migration | Complete | 4986d6e, b787373, c9e9bdc |
| 04-05 | Batch Polars Migration | Complete | af2084d, 74f9ac4 |
| 04-06 | Remove Pandas | Complete | f8164d7 |

## Phase 3 Progress

| Plan | Name | Status | Commit |
|------|------|--------|--------|
| 03-01 | SQLite Job Registry | Complete | 6435beb, 692d45d, 06c5b4e |
| 03-02 | Memory Leak & Race Condition Fixes | Complete | c03d39f, 185dc73 |
| 03-03 | Resource Cleanup | Complete | 6cdc584, 13d4fb1, 7ba548d, c217e90 |
| 03-04 | Frontend Polling Optimization | Complete | dd7b371, 3771f2d |

## Phase 1 Success Criteria

1. [x] When an error occurs, user sees specific message with error ID (not "Internal Server Error")
2. [x] Application starts successfully on Windows, Linux, and macOS without path modifications
3. [x] File operations reject paths outside allowed directories (path traversal blocked)
4. [x] Application logs show which audio backend initialized at startup
5. [x] All function signatures in csv_parser.py have return type hints

## Phase 1 Progress

| Plan | Name | Status | Commit |
|------|------|--------|--------|
| 01-01 | Bare Except Replacement | Complete | d6bc00c |
| 01-02 | Path Traversal Prevention | Complete | 371cd4d |
| 01-03 | Global Exception Handler | Complete | 49b1694 |
| 01-04 | Cross-Platform Temp | Complete | 94094fb |
| 01-05 | MP3 Path Resolution | Complete | a17e914 |
| 01-06 | Remove Hardcoded Paths | Complete | 0c8a310 |
| 01-07 | Audio Backend Startup | Complete | 5138e8c |

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Phase completion | 6 | 5 |
| Phase 1 plans | 7 | 7 |
| Phase 3 plans | 4 | 4 |
| Phase 4 plans | 6 | 6 |
| Phase 2 plans | 7 | 7 |
| Phase 5 plans | 7 | 7 |
| Requirements done | 62 | 59 (22 from Phase 1 + 9 from Phase 2 + 9 from Phase 3 + 10 from Phase 4 + 9 from Phase 5) |
| Critical issues fixed | 15 | 15 |

## Accumulated Context

### Key Decisions
- Brownfield improvement approach (refactor > rewrite)
- Keep FastAPI (not Litestar) — bottleneck is I/O not API layer
- Keep AST model — works well, no need to migrate
- Migrate pandas to Polars for CSV (5-30x faster)
- Keep SQLite for job registry (adequate for single-user)
- Use tempfile.gettempdir() for cross-platform temp directories
- Use Path.resolve() for path traversal prevention (handles symlinks)
- Centralized security utility at backend/app/core/security.py
- Three-handler exception chain: StarletteHTTPException, RequestValidationError, Exception
- 8-character UUID prefix for error_id correlation
- CalendarBrowser extracts SORTED base from recording.path via regex (no API call needed)
- UncertaintyReview uses regex split for cross-platform path parsing
- TTLCache for job dicts: 1h/100 for single jobs, 4h/50 for batch jobs
- Atomic write pattern: temp file + os.replace (works on Unix and Windows)
- POLL-001: 1.5x multiplier for exponential backoff polling (1s->1.5s->2.25s->10s max)
- aiosqlite for job registry (not SQLAlchemy async) — simpler, no ORM overhead
- 5s timeout then force kill for process termination on shutdown
- TimeoutMiddleware: 60s for all endpoints except /analyze (long-running by design)
- Job lookup order: temp file -> TTLCache -> SQLite (most current to restart recovery)
- Use polars 1.23.0 for pandas-to-polars migration (stable, backwards compatible)
- Module-level regex compilation: DATE_PATTERN and PREDICTIONS_PATTERN for filename parsing
- Waveform cache uses mtime in cache key for automatic invalidation (no manual cleanup)
- Cache location: SORTED_FOLDER/.waveform_cache (hidden folder for metadata)
- N+1 double CSV read eliminated in uncertainty.py (single pl.read_csv() per file)
- polars patterns: pl.filter() with pl.col(), iter_rows(named=True), with_row_index() for index tracking
- Double-read eliminated in batch.py: single pl.read_csv() instead of nrows=1 + full_df pattern
- csv_parser.py fully migrated to polars: df[row, col] indexing, df.height for row count, None checks for nulls
- Polars auto-strips column whitespace and handles quotes (no manual preprocessing needed)
- pandas completely removed from requirements.txt and codebase (polars migration complete)
- useTrackEditor hook: encapsulates 10 track operations + tracks/hasUnsavedChanges state (CsvViewer reduced 1279→958 lines)
- Hook composition pattern: useTrackEditor + useAudioPlayer + useAutosave for clean component structure
- Inline utilities temporarily when dependencies unavailable (refactor after parallel plans merge)
- useAutosave hook: generic hook with TypeScript generics, supports immediate/debounced autosave, mounted ref pattern
- OWASP filename sanitization in export.py: re.sub(r'[<>:"|?*]', '_', song_name) prevents filesystem issues
- Legacy training.py (535 lines Keras CNN) deleted - ast_training service is used instead
- howler audio library removed from frontend (unused, native HTMLAudioElement sufficient)
- Time calculations extracted to frontend/src/utils/timeCalculations.ts (calculateDuration, timeToSeconds, secondsToTimeFormat, parseTimeToSeconds)
- Single source of truth for time logic - imported by CsvViewer and useTrackEditor hook
- useAudioPlayer hook: encapsulates player state (showPlayer, playingTrackId, selectedTrackId, seekToTime) and operations
- CsvSelector component: presentational component for file selection UI with status badges (edited/exported/analyzing)
- PlayerControls component: presentational component for player toggle, threshold slider, save/discard, and export buttons
- TrackTable component: presentational component for editable track rows (214 lines) with all operations via callback props
- CsvViewer reduced from 946 to 708 lines (25% reduction) via component extraction (CsvSelector, PlayerControls, TrackTable)
- Component composition pattern: CsvViewer orchestrates extracted components with single responsibility
- Unicode checkmarks (✓/✗) instead of emoji for better cross-platform consistency
- Ref-based callback pattern for playback control: StickyPlayer owns audio state, exposes handlePlayPause via ref
- Silent cancellation pattern: axios.isCancel() checks prevent error toasts on intentional request aborts
- Remove legacy unused functions during cleanup (exportSelected was never called)
- CsvViewer refactored: reduced from ~842 to 708 lines by extracting CsvSelector and PlayerControls (~134 line reduction)
- useUndoRedo hook: snapshot-based undo/redo with max 20 history via .slice(-19), history persists across saves
- useKeyboardShortcuts hook: stable ref pattern for global keyboard shortcuts with INPUT/TEXTAREA guards
- Space key normalized to 'space' string for clarity in keyboard handler maps
- Atomic write pattern: tempfile.mkstemp in same directory + os.replace (atomic on POSIX, near-atomic on Windows)
- CSV save operations now crash-safe via atomic_write utility (no partial write corruption)
- zustand toast store: global toast queue with 5-toast limit (oldest removed when exceeded)
- Axios error interceptor: auto-dispatches error toasts for backend errors via useToastStore.getState()
- Backend error codes: HTTP_{status_code}, VALIDATION_ERROR, INTERNAL_ERROR in all exception handlers
- Toast behavior: error toasts manual dismiss only, success toasts auto-dismiss 5s, retry button for GET errors
- Toast X close button: explicit dismissal UI in top-right corner with stopPropagation
- Progress indicators: inline in PlayerControls with pulsing animation, shows "Loading..."/"Saving..." during operations
- CalendarBrowser play button: navigates to CsvViewer with recording's CSV path via onOpenCsv callback
- AbortController pattern: infrastructure in place for request cancellation (cleanup on unmount)
- Keyboard shortcuts in CsvViewer: Space (toggle player), Ctrl+S (save), Ctrl+Z (undo), Ctrl+Y/Shift+Z (redo), 1-5 (classify)
- Wrapper functions for track mutations: pushState before each edit enables 20-step undo/redo history
- useUndoRedo exposes present state for sync after undo/redo via useEffect in CsvViewer
- Undo/Redo buttons in PlayerControls: disabled states (opacity-50) when no history, tooltips show shortcuts
- KeyboardHelp modal: ? key (Shift+?) toggles help panel listing all 12 shortcuts with visual key combinations
- Help discoverability: ? icon button in PlayerControls + modal with click-outside-to-close and Escape dismissal
- Ref-based callback pattern: StickyPlayer exposes handlePlayPause via ref for keyboard control without lifting audio state
- Spacebar play/pause: togglePlaybackRef pattern enables CsvViewer to control playback while StickyPlayer owns audio element
- AbortController wired to axios: signal passed to CSV parse, autosave check, MP3 resolution, and export requests
- Silent cancellation: axios.isCancel() checks prevent error toasts on intentional request cancellations
- Code cleanup: removed duplicate timeToSeconds, unused exportSelected function, and other unused variables
- DeviceManager singleton: device_type strings "cuda_nvidia"/"cuda_amd"/"cpu" for downstream optimization branching
- Device detection runs once at startup via get_device_manager(), not per-inference call
- Keep waveform-data library (no wavesurfer.js migration) — current impl works well with server-side caching
- Confidence threshold auto-tuning: per-recording, localStorage persistence, default 0.7, range [0.3, 0.95], learning rate 0.05

### Research Completed (2026-01-20)
- .planning/research/STACK.md — PyTorch/torchaudio recommendations
- .planning/research/FEATURES.md — Table stakes vs differentiators
- .planning/research/ARCHITECTURE.md — Component patterns
- .planning/research/PITFALLS.md — ROCm Windows, CPU performance
- .planning/research/SUMMARY.md — Synthesis with phase recommendations

### Detailed Audit Completed (2026-01-20)
- 60 specific issues identified with file:line references
- .planning/DETAILED_AUDIT.md — Full findings
- .planning/TECHNOLOGY_AUDIT.md — Migration recommendations

### Plans Completed (2026-01-21 to 2026-01-28)
- .planning/phases/01-foundation-stability/01-01-SUMMARY.md — Bare except replacement
- .planning/phases/01-foundation-stability/01-02-SUMMARY.md — Path traversal prevention
- .planning/phases/01-foundation-stability/01-03-SUMMARY.md — Exception handlers & type hints
- .planning/phases/01-foundation-stability/01-04-SUMMARY.md — Cross-platform temp directories
- .planning/phases/01-foundation-stability/01-05-SUMMARY.md — MP3 path resolution endpoint
- .planning/phases/01-foundation-stability/01-06-SUMMARY.md — Remove hardcoded paths
- .planning/phases/01-foundation-stability/01-07-SUMMARY.md — Audio backend startup validation
- .planning/phases/03-backend-stability/03-01-SUMMARY.md — SQLite job registry
- .planning/phases/03-backend-stability/03-02-SUMMARY.md — Memory leak & race condition fixes
- .planning/phases/03-backend-stability/03-03-SUMMARY.md — Resource cleanup
- .planning/phases/03-backend-stability/03-04-SUMMARY.md — Frontend exponential backoff
- .planning/phases/04-performance-migration/04-01-SUMMARY.md — Performance prep (polars + regex)
- .planning/phases/04-performance-migration/04-02-SUMMARY.md — Waveform caching
- .planning/phases/04-performance-migration/04-03-SUMMARY.md — CSV parser polars migration
- .planning/phases/04-performance-migration/04-04-SUMMARY.md — Uncertainty polars migration
- .planning/phases/04-performance-migration/04-05-SUMMARY.md — Batch polars migration
- .planning/phases/04-performance-migration/04-06-SUMMARY.md — Remove pandas dependency
- .planning/phases/05-frontend-decomposition/05-01-SUMMARY.md — Code cleanup (training.py, howler, filename sanitization)
- .planning/phases/05-frontend-decomposition/05-02-SUMMARY.md — Extract time calculation utilities
- .planning/phases/05-frontend-decomposition/05-03-SUMMARY.md — Extract useTrackEditor hook
- .planning/phases/05-frontend-decomposition/05-04-SUMMARY.md — Extract useAutosave hook
- .planning/phases/05-frontend-decomposition/05-05-SUMMARY.md — Extract useAudioPlayer hook
- .planning/phases/05-frontend-decomposition/05-06-SUMMARY.md — Extract CsvSelector & PlayerControls components
- .planning/phases/05-frontend-decomposition/05-07-SUMMARY.md — Extract TrackTable & finalize refactor
- .planning/phases/02-core-ux-polish/02-01-SUMMARY.md — Foundation hooks (useUndoRedo, useKeyboardShortcuts)
- .planning/phases/02-core-ux-polish/02-02-SUMMARY.md — Toast System & Error Pipeline
- .planning/phases/02-core-ux-polish/02-03-SUMMARY.md — Atomic CSV writes
- .planning/phases/02-core-ux-polish/02-04-SUMMARY.md — Keyboard shortcuts & undo/redo integration
- .planning/phases/02-core-ux-polish/02-05-SUMMARY.md — Progress indicators & CalendarBrowser navigation
- .planning/phases/02-core-ux-polish/02-06-SUMMARY.md — Keyboard shortcut help panel (? key toggle)

### Blockers
(None)

### TODOs
- [x] Execute Phase 1 plans - COMPLETE (7/7)
- [x] Verify Phase 1 goal achievement - PASSED
- [x] Execute Phase 3 plans - COMPLETE (4/4)
- [x] Verify Phase 3 goal achievement - PASSED
- [x] Execute Phase 4 Plan 01 - COMPLETE (2/2 tasks)
- [x] Execute Phase 4 Plan 02 - COMPLETE (2/2 tasks)
- [x] Execute Phase 4 Plan 03 - COMPLETE (3/3 tasks)
- [x] Execute Phase 4 Plan 04 - COMPLETE (3/3 tasks)
- [x] Execute Phase 4 Plan 05 - COMPLETE (2/2 tasks)
- [x] Execute Phase 4 Plan 06 - COMPLETE (3/3 tasks)
- [x] Verify Phase 4 goal achievement - READY FOR VERIFICATION
- [x] Execute Phase 5 Plan 01 - COMPLETE (2/2 tasks)
- [x] Execute Phase 5 Plan 02 - COMPLETE (2/2 tasks)
- [x] Execute Phase 5 Plan 03 - COMPLETE (2/2 tasks)
- [x] Execute Phase 5 Plan 04 - COMPLETE (2/2 tasks)
- [x] Execute Phase 5 Plan 05 - COMPLETE (2/2 tasks)
- [x] Execute Phase 5 Plan 06 - COMPLETE (3/3 tasks)
- [x] Execute Phase 5 Plan 07 - COMPLETE (2/2 tasks)
- [x] Verify Phase 5 goal achievement - READY FOR VERIFICATION
- [x] Execute Phase 2 Plan 01 - COMPLETE (2/2 tasks)
- [x] Execute Phase 2 Plan 02 - COMPLETE (2/2 tasks)
- [x] Execute Phase 2 Plan 03 - COMPLETE (2/2 tasks)
- [x] Execute Phase 2 Plan 04 - COMPLETE (2/2 tasks, Task 2 already done in 02-02)
- [x] Execute Phase 2 Plan 05 - COMPLETE (2/2 tasks)
- [x] Execute Phase 2 Plan 06 - COMPLETE (1/1 tasks)
- [x] Execute Phase 2 Plan 07 - COMPLETE (2/2 tasks, gap closure)
- [x] Verify Phase 2 goal achievement - READY FOR VERIFICATION
- [x] Begin Phase 6: GPU & CPU Optimization
- [x] Execute Phase 6 Plan 01 - COMPLETE (2/2 tasks)
- [x] Execute Phase 6 Plan 02 - COMPLETE (1/1 tasks)
- [ ] Execute Phase 6 Plan 03
- [ ] Execute Phase 6 Plan 04
- [ ] Execute Phase 6 Plan 05
- [x] Execute Phase 6 Plan 06 - COMPLETE (2/2 tasks)

**Stopped at:** Completed 06-02 torch.compile GPU Acceleration

## Session Continuity

**Last session:** 2026-01-29
**Stopped at:** Completed 06-02-PLAN.md (Phase 6, Plan 2)
**Resume file:** None

**If context is lost, read these files in order:**
1. .planning/PROJECT.md — Core value and constraints
2. .planning/ROADMAP.md — Phase structure and requirements
3. .planning/STATE.md — Current position (this file)
4. .planning/phases/05-frontend-decomposition/05-03-SUMMARY.md — Track editor hook
5. .planning/phases/05-frontend-decomposition/05-07-SUMMARY.md — TrackTable & finalize refactor
6. .planning/phases/02-core-ux-polish/02-01-SUMMARY.md — Foundation hooks
7. .planning/phases/02-core-ux-polish/02-02-SUMMARY.md — Toast System & Error Pipeline
8. .planning/phases/02-core-ux-polish/02-04-SUMMARY.md — Keyboard shortcuts & undo/redo
9. .planning/phases/02-core-ux-polish/02-05-SUMMARY.md — Progress indicators & navigation
10. .planning/phases/02-core-ux-polish/02-06-SUMMARY.md — Keyboard shortcut help panel
11. .planning/phases/02-core-ux-polish/02-07-SUMMARY.md — Gap closure (spacebar + AbortController)

---
*State updated: 2026-01-29 — 06-02 complete (torch.compile GPU acceleration with warmup benchmarking, graceful CPU fallback)*

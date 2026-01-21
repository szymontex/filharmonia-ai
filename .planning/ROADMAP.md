# Roadmap: Filharmonia AI v0.9 — Polish & Stability

**Created:** 2026-01-21
**Depth:** Comprehensive
**Phases:** 6
**Requirements:** 62 total

## Overview

This milestone transforms Filharmonia AI from a working but fragile prototype into a production-ready tool. The roadmap prioritizes foundation fixes (security, error handling) before UX improvements, then performance, then optional GPU optimizations. Every phase delivers observable improvements — no phase depends on unwritten future phases to be useful.

---

## Phase 1: Foundation Stability

**Goal:** Users see meaningful error messages instead of silent failures; paths work cross-platform.

**Dependencies:** None (first phase)

**Plans:** 7 plans

Plans:
- [x] 01-01-PLAN.md — Replace bare except clauses with specific exceptions
- [x] 01-02-PLAN.md — Add path traversal prevention
- [x] 01-03-PLAN.md — Global exception handler and type hints
- [x] 01-04-PLAN.md — Cross-platform temp directories
- [x] 01-05-PLAN.md — MP3 path resolution API endpoint
- [x] 01-06-PLAN.md — Remove hardcoded frontend paths
- [x] 01-07-PLAN.md — Startup validation and PyTorch pinning

**Requirements:**
- CRIT-01: Replace bare `except:` at `main.py:84`
- CRIT-02: Replace bare `except:` at `analyze.py:36,49`
- CRIT-03: Replace bare `except:` at `batch.py:39`
- CRIT-04: Replace bare `except:` at `files.py:49,66`
- CRIT-05: Replace bare `except:` at `ast_inference.py:189-190`
- CRIT-06: Replace bare `except:` at `ast_training.py:214`
- CRIT-07: Path traversal prevention in `files.py:104`
- CRIT-08: Path traversal prevention in `csv_parser.py:181`
- CRIT-09: Path traversal prevention in `waveform.py`
- CRIT-10: Global FastAPI exception handler
- PATH-01: Remove hardcoded path from `CsvViewer.tsx:206`
- PATH-02: Remove hardcoded path from `CalendarBrowser.tsx:268-269`
- PATH-03: Backend API endpoint for MP3 path resolution
- PATH-04: Use `tempfile.gettempdir()` in `analyze.py`
- PATH-05: Use `tempfile.gettempdir()` in `batch.py`
- PATH-06: Use `tempfile.gettempdir()` in `analyze_worker.py`
- TYPE-01: Return type hints `csv_parser.py:52`
- TYPE-02: Return type hints `csv_parser.py:70`
- TYPE-03: Return type hints `csv_parser.py:198,203,208,215,222`
- TYPE-04: Fix time parsing in `uncertainty.py:21-22`
- INFRA-04: Startup validation for audio backends
- INFRA-05: Pin exact PyTorch versions

**Success Criteria:**
1. When an error occurs, user sees specific message with error ID (not "Internal Server Error")
2. Application starts successfully on Windows, Linux, and macOS without path modifications
3. File operations reject paths outside allowed directories (path traversal blocked)
4. Application logs show which audio backend initialized at startup
5. All function signatures in csv_parser.py have return type hints

---

## Phase 2: Core UX Polish

**Goal:** Users can efficiently edit audio classifications with keyboard shortcuts and undo mistakes.

**Dependencies:** Phase 1 (error handling enables meaningful feedback)

**Plans:** 0 plans

Plans:
- [ ] TBD — to be planned

**Requirements:**
- UX-01: Keyboard shortcuts — spacebar play/pause
- UX-02: Ctrl+S explicit save
- UX-03: Ctrl+Z undo (single step first)
- UX-04: Number keys 1-5 for class cycling
- UX-05: Debounce abort — cancel in-flight requests
- UX-06: Standardize error response format
- UX-07: Progress indicators
- CLEAN-04: Implement or remove `handlePlayRecording` in `CalendarBrowser.tsx:108`
- CLEAN-05: Autosave atomicity — write to .tmp then rename

**Success Criteria:**
1. User can play/pause audio with spacebar, save with Ctrl+S, and undo last change with Ctrl+Z
2. User can press 1-5 to cycle through classification labels on selected segment
3. Progress indicator shows current operation stage ("Loading...", "Analyzing...", "Saving...")
4. After making a change, user can press Ctrl+Z to restore previous state
5. All API errors return consistent JSON format with status, message, and code fields

---

## Phase 3: Backend Stability

**Goal:** Jobs persist across server restarts; memory does not leak over time.

**Dependencies:** Phase 1 (exception handling must be in place)

**Plans:** 4 plans

Plans:
- [ ] 03-01-PLAN.md — SQLite job registry with TTL cache (INFRA-01)
- [ ] 03-02-PLAN.md — TTLCache for memory leaks + atomic writes (CRIT-11, CRIT-12, CRIT-13)
- [ ] 03-03-PLAN.md — Process cleanup, blocking I/O fix, timeout middleware (CRIT-14, CRIT-15, INFRA-02)
- [ ] 03-04-PLAN.md — Frontend exponential backoff polling (PERF-05)

**Requirements:**
- CRIT-11: Fix memory leak in `analyze.py:48` — TTL cleanup for `_single_jobs`
- CRIT-12: Fix memory leak in `batch.py:48-49` — TTL cleanup for `_jobs`
- CRIT-13: Fix race condition in `batch.py:138,164` — atomic writes
- CRIT-14: Fix zombie processes in `analyze.py:90` — cleanup on shutdown
- CRIT-15: Fix blocking I/O in `csv_parser.py:273-276` — use `asyncio.to_thread()`
- INFRA-01: SQLite job registry
- INFRA-02: Request timeout for long operations
- INFRA-03: SQLAlchemy connection pooling (note: using aiosqlite instead, no explicit pooling needed)
- PERF-05: Replace polling with exponential backoff

**Success Criteria:**
1. Job status survives server restart (user can see old jobs after reboot)
2. Server memory usage stays stable after processing 100+ files (no unbounded growth)
3. Long-running batch analysis does not block concurrent single-file analysis
4. Server shutdown terminates all worker processes cleanly (no orphans in process list)
5. Frontend uses exponential backoff for status polling (network tab shows increasing intervals)

---

## Phase 4: Performance & Migration

**Goal:** CSV operations complete 5-30x faster; waveforms load instantly on repeat views.

**Dependencies:** Phase 3 (job registry enables caching metadata)

**Requirements:**
- TECH-01: Install polars
- TECH-02: Migrate `csv_parser.py` to polars
- TECH-03: Migrate `uncertainty.py` to polars
- TECH-04: Migrate `batch.py` to polars
- TECH-05: Remove pandas from requirements
- PERF-01: Fix CSV double-read in `batch.py:335-341`
- PERF-02: Fix N+1 query in `uncertainty.py:273-293`
- PERF-03: Fix regex recompilation in `files.py:86`
- PERF-04: Waveform caching

**Success Criteria:**
1. Opening a 1000-row CSV completes in under 100ms (measured in browser devtools)
2. Waveform displays instantly (<500ms) when opening a previously-viewed file
3. Batch analysis of 10 files shows no redundant CSV reads in server logs
4. `pip show pandas` returns "package not found" after migration complete
5. Uncertainty review page loads in under 2 seconds for 50+ files

---

## Phase 5: Frontend Decomposition

**Goal:** CsvViewer is maintainable — each component has single responsibility.

**Dependencies:** Phase 2 (UX features implemented first, then refactored)

**Requirements:**
- COMP-01: Split CsvViewer into TrackTable, CsvSelector, PlayerControls
- COMP-02: Extract `useTrackEditor` hook
- COMP-03: Extract `useAudioPlayer` hook
- COMP-04: Extract `useAutosave` hook
- COMP-05: Extract time calculation utility
- CLEAN-01: Delete unused `training.py`
- CLEAN-02: Remove unused `howler` from package.json
- CLEAN-03: Remove `@types/howler`
- CLEAN-06: Sanitize filenames in `export.py:169`

**Success Criteria:**
1. CsvViewer.tsx is under 300 lines (orchestration only)
2. Each extracted hook has single responsibility and can be unit tested in isolation
3. `npm ls howler` returns empty (unused dependency removed)
4. Exported filenames contain no invalid characters (tested with `<>:"|?*` in input)
5. Deleting training.py causes no test failures or import errors

---

## Phase 6: GPU & CPU Optimization

**Goal:** Inference runs optimally on CUDA, ROCm, or CPU without manual configuration.

**Dependencies:** Phase 4 (performance baseline established)

**Requirements:**
- GPU-01: Unified device detection (distinguish NVIDIA from AMD)
- GPU-02: torch.compile for GPU acceleration
- GPU-03: ONNX export for CPU optimization
- GPU-04: INT8 quantization for CPU inference
- GPU-05: ROCm 6.4 support on Linux
- GPU-06: ROCm Windows preview support
- FRONT-01: Upgrade React 18 to React 19
- FRONT-02: Consider wavesurfer.js migration
- FRONT-03: Confidence threshold auto-tuning

**Success Criteria:**
1. Startup logs show detected device type (NVIDIA CUDA, AMD ROCm, or CPU) without user configuration
2. GPU inference with torch.compile is measurably faster than eager mode (benchmark logged)
3. CPU inference with ONNX INT8 is at least 3x faster than eager PyTorch (benchmark logged)
4. Application runs on Linux with AMD GPU using ROCm 6.4 (tested on RX 7000/9000 series)
5. React 19 upgrade causes no regressions in existing functionality

---

## Progress

| Phase | Name | Requirements | Plans | Status |
|-------|------|--------------|-------|--------|
| 1 | Foundation Stability | 22 | 7 | Complete |
| 2 | Core UX Polish | 9 | 0 | Pending |
| 3 | Backend Stability | 9 | 4 | Planned |
| 4 | Performance & Migration | 10 | 0 | Pending |
| 5 | Frontend Decomposition | 9 | 0 | Pending |
| 6 | GPU & CPU Optimization | 9 | 0 | Pending |
| **Total** | | **68** | **11** | |

Note: Some requirements split across phases where logical (e.g., type hints with foundation). Total in phases exceeds 62 due to research-recommended additions being implicit.

---

## Requirement Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| CRIT-01 | 1 | Complete |
| CRIT-02 | 1 | Complete |
| CRIT-03 | 1 | Complete |
| CRIT-04 | 1 | Complete |
| CRIT-05 | 1 | Complete |
| CRIT-06 | 1 | Complete |
| CRIT-07 | 1 | Complete |
| CRIT-08 | 1 | Complete |
| CRIT-09 | 1 | Complete |
| CRIT-10 | 1 | Complete |
| CRIT-11 | 3 | Pending |
| CRIT-12 | 3 | Pending |
| CRIT-13 | 3 | Pending |
| CRIT-14 | 3 | Pending |
| CRIT-15 | 3 | Pending |
| PERF-01 | 4 | Pending |
| PERF-02 | 4 | Pending |
| PERF-03 | 4 | Pending |
| PERF-04 | 4 | Pending |
| PERF-05 | 3 | Pending |
| TECH-01 | 4 | Pending |
| TECH-02 | 4 | Pending |
| TECH-03 | 4 | Pending |
| TECH-04 | 4 | Pending |
| TECH-05 | 4 | Pending |
| PATH-01 | 1 | Complete |
| PATH-02 | 1 | Complete |
| PATH-03 | 1 | Complete |
| PATH-04 | 1 | Complete |
| PATH-05 | 1 | Complete |
| PATH-06 | 1 | Complete |
| COMP-01 | 5 | Pending |
| COMP-02 | 5 | Pending |
| COMP-03 | 5 | Pending |
| COMP-04 | 5 | Pending |
| COMP-05 | 5 | Pending |
| TYPE-01 | 1 | Complete |
| TYPE-02 | 1 | Complete |
| TYPE-03 | 1 | Complete |
| TYPE-04 | 1 | Complete |
| CLEAN-01 | 5 | Pending |
| CLEAN-02 | 5 | Pending |
| CLEAN-03 | 5 | Pending |
| CLEAN-04 | 2 | Pending |
| CLEAN-05 | 2 | Pending |
| CLEAN-06 | 5 | Pending |
| UX-01 | 2 | Pending |
| UX-02 | 2 | Pending |
| UX-03 | 2 | Pending |
| UX-04 | 2 | Pending |
| UX-05 | 2 | Pending |
| UX-06 | 2 | Pending |
| UX-07 | 2 | Pending |
| INFRA-01 | 3 | Pending |
| INFRA-02 | 3 | Pending |
| INFRA-03 | 3 | Pending |
| INFRA-04 | 1 | Complete |
| INFRA-05 | 1 | Complete |
| GPU-01 | 6 | Pending |
| GPU-02 | 6 | Pending |
| GPU-03 | 6 | Pending |
| GPU-04 | 6 | Pending |
| GPU-05 | 6 | Pending |
| GPU-06 | 6 | Pending |
| FRONT-01 | 6 | Pending |
| FRONT-02 | 6 | Pending |
| FRONT-03 | 6 | Pending |

**Coverage:** 62/62 requirements mapped (100%)

---

*Roadmap created: 2026-01-21*
*Phase 1 complete: 2026-01-21*
*Phase 3 planned: 2026-01-21*

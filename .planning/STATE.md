# Project State: Filharmonia AI

## Current Position

**Phase:** 3 of 6 — Backend Stability
**Previous:** Phase 1 Foundation Stability ✓ Complete
**Status:** In progress
**Progress:** [██░░░░░░░░] 2/6 phases complete (03: 3/4 plans)

**Last activity:** 2026-01-21 — Completed 03-01-PLAN.md (SQLite Job Registry)

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-21)

**Core value:** Zamiast reczenie sluchac ~6-8h nagran/tyg, AI robi to za ciebie.
**Current focus:** v0.9 — Polish & Stability
**Next phase goal:** Backend memory and I/O stability - no leaks, atomic writes, proper caching.

## Phase 3 Progress

| Plan | Name | Status | Commit |
|------|------|--------|--------|
| 03-01 | SQLite Job Registry | Complete | 6435beb, 692d45d, 06c5b4e |
| 03-02 | Memory Leak & Race Condition Fixes | Complete | c03d39f, 185dc73 |
| 03-03 | Resource Cleanup | Pending | - |
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
| Phase completion | 6 | 1 |
| Phase 1 plans | 7 | 7 |
| Requirements done | 62 | 22 (CRIT-01 to CRIT-10, TYPE-01 to TYPE-04, PATH-01 to PATH-06, INFRA-04, INFRA-05) |
| Critical issues fixed | 15 | 12 |

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
- aiosqlite for job registry (not SQLAlchemy async) — simpler, no ORM overhead

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

### Plans Completed (2026-01-21)
- .planning/phases/01-foundation-stability/01-01-SUMMARY.md — Bare except replacement
- .planning/phases/01-foundation-stability/01-02-SUMMARY.md — Path traversal prevention
- .planning/phases/01-foundation-stability/01-03-SUMMARY.md — Exception handlers & type hints
- .planning/phases/01-foundation-stability/01-04-SUMMARY.md — Cross-platform temp directories
- .planning/phases/01-foundation-stability/01-05-SUMMARY.md — MP3 path resolution endpoint
- .planning/phases/01-foundation-stability/01-06-SUMMARY.md — Remove hardcoded paths
- .planning/phases/01-foundation-stability/01-07-SUMMARY.md — Audio backend startup validation
- .planning/phases/03-backend-stability/03-01-SUMMARY.md — SQLite job registry
- .planning/phases/03-backend-stability/03-02-SUMMARY.md — Memory leak & race condition fixes
- .planning/phases/03-backend-stability/03-04-SUMMARY.md — Frontend exponential backoff

### Blockers
(None)

### TODOs
- [x] Execute Phase 1 plans - COMPLETE (7/7)
- [x] Verify Phase 1 goal achievement - PASSED
- [ ] Execute Phase 3 plans - IN PROGRESS (3/4)
- [ ] Plan Phase 2: Core UX Polish

### Open Questions
(None)

## Session Continuity

**Last session:** 2026-01-21
**Stopped at:** Completed 03-01-PLAN.md (SQLite Job Registry)
**Resume file:** None

**If context is lost, read these files in order:**
1. .planning/PROJECT.md — Core value and constraints
2. .planning/ROADMAP.md — Phase structure and requirements
3. .planning/STATE.md — Current position (this file)
4. .planning/phases/03-backend-stability/03-01-SUMMARY.md — Last completed plan

---
*State updated: 2026-01-21 — Completed 03-01 SQLite Job Registry*

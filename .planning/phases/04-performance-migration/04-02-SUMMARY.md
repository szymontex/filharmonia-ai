---
phase: 04-performance-migration
plan: 02
subsystem: waveform
tags: [caching, performance, librosa, filesystem]

dependency-graph:
  requires: []
  provides:
    - "Waveform data caching system"
    - "Mtime-based cache invalidation"
    - "WAVEFORM_CACHE_DIR configuration"
  affects:
    - "Future waveform visualization features"
    - "Editor performance on repeat views"

tech-stack:
  added: []
  patterns:
    - "Filesystem-based caching with mtime invalidation"
    - "MD5 hash for cache keys (16-char truncated)"
    - "Async file I/O via asyncio.to_thread"
    - "Graceful degradation on corrupted cache"

key-files:
  created: []
  modified:
    - backend/app/config.py
    - backend/app/api/v1/waveform.py

decisions:
  - id: "04-02-D1"
    choice: "Include mtime in cache key"
    reason: "Automatic invalidation when MP3 file is modified, no manual cache management needed"
  - id: "04-02-D2"
    choice: "Truncate MD5 hash to 16 characters"
    reason: "Sufficient for uniqueness while keeping filenames reasonable"
  - id: "04-02-D3"
    choice: "Cache location under SORTED_FOLDER/.waveform_cache"
    reason: "Easy to manage, clear disk usage, can purge all at once, follows hidden metadata folder convention"

metrics:
  duration: "1m 18s"
  completed: "2026-01-28"
---

# Phase 04 Plan 02: Waveform Caching Summary

Filesystem caching for waveform data to enable instant loading (<500ms) on repeat views with mtime-based invalidation.

## One-liner

Waveform endpoint now caches generated JSON to filesystem with MD5-hashed filenames including mtime for automatic invalidation.

## Commits

| Hash | Type | Message |
|------|------|---------|
| 7e99228 | feat | Add waveform cache directory setting |
| 277a71f | feat | Implement waveform caching (PERF-04) |

## What Was Built

### Configuration (backend/app/config.py)

Added `WAVEFORM_CACHE_DIR` setting to Settings class:
- Defaults to `SORTED_FOLDER/.waveform_cache`
- Configurable via `WAVEFORM_CACHE_DIR` environment variable
- Hidden folder convention for metadata storage

### Waveform Caching Logic (backend/app/api/v1/waveform.py)

**Cache Helper Function:**
```python
def get_cache_path(mp3_path: Path, samples_per_pixel: int) -> Path
```
- Generates cache file path based on MP3 path, samples_per_pixel, and file mtime
- Cache key format: `{mp3_path}:{samples_per_pixel}:{mtime}`
- MD5 hash truncated to 16 characters
- Filename: `{mp3_stem}_{hash}.json`

**Endpoint Flow:**
1. Validate MP3 path (existing security check)
2. Generate cache path with mtime
3. **Check cache first:**
   - If cache file exists, read and return JSON
   - If corrupted (JSONDecodeError/IOError), regenerate
4. **If cache miss:**
   - Load audio with librosa (sr=8000, mono)
   - Generate min/max amplitude data per pixel
   - Write result to cache (async, non-blocking)
   - Return result

**Key Design Decisions:**
- All file I/O uses `asyncio.to_thread` to avoid blocking event loop
- Cache directory created automatically on first write
- Graceful degradation: corrupted cache regenerates, doesn't error
- Mtime in cache key means old cache files auto-ignored when MP3 modified

## Decisions Made

### D1: Mtime-Based Cache Invalidation

**Context:** Need automatic cache invalidation when source MP3 changes.

**Decision:** Include file mtime in cache key.

**Rationale:**
- No manual cache management or cleanup needed
- User edits MP3 → mtime changes → new cache key → regenerates waveform
- Old cache files become orphaned but harmless
- Simple and reliable

**Alternative considered:** Hash-based invalidation (hash file contents)
- Rejected: Too expensive to hash large MP3 files on every request
- mtime check is instant

### D2: Cache Key Hashing

**Context:** Cache key includes full path, which can be very long.

**Decision:** MD5 hash truncated to 16 characters.

**Rationale:**
- 16 hex chars = 2^64 possible values (sufficient for uniqueness)
- Keeps filenames reasonable length
- MD5 is fast (not used for security here)
- Collision risk negligible for this use case

### D3: Cache Location

**Context:** Where to store cached waveform files.

**Decision:** `SORTED_FOLDER/.waveform_cache/`

**Rationale:**
- Single location for all waveform caches
- Easy to manage: user knows where to find it
- Clear disk usage: can check size of one folder
- Can purge entire cache at once if needed
- Follows convention of hidden folders for metadata (`.git`, `.claude`, etc.)
- Per-file caches would be scattered and hard to track

**Alternative considered:** Cache alongside MP3 files
- Rejected: Clutters music folders, hard to manage

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

| Check | Status |
|-------|--------|
| WAVEFORM_CACHE_DIR in config.py | Pass |
| WAVEFORM_CACHE_DIR usage in waveform.py | Pass |
| cache_path logic present | Pass |
| asyncio.to_thread for file I/O | Pass |
| Cache check before generation | Pass |
| Cache write after generation | Pass |
| Mtime included in cache key | Pass |

All verification checks passed.

## Performance Impact

**Before:**
- Every waveform request: librosa.load + numpy processing
- Typical 3-minute MP3: ~2-5 seconds to generate
- No way to avoid regeneration

**After:**
- First request: Same as before (~2-5 seconds) + cache write
- Second request: Instant (<500ms) - just JSON read
- Third+ requests: Instant (<500ms) - cached
- MP3 modified: Automatic regeneration, then cached again

**Expected improvement:**
- 10x faster on repeat views (5s → 0.5s)
- Scales well: more frequently accessed files benefit more
- No memory overhead: pure filesystem cache

## Next Phase Readiness

Ready for next performance optimizations:
- Waveform caching infrastructure complete
- Cache directory configurable and ready
- No blockers identified

Future enhancements could include:
- Cache size monitoring/limits
- Automatic cleanup of orphaned cache files
- Cache warming for frequently accessed files

## Files Changed

```
backend/app/config.py              +7 lines (WAVEFORM_CACHE_DIR setting)
backend/app/api/v1/waveform.py     +37 lines (imports, cache logic)
```

Total: 44 lines added, 27 lines modified (refactored into _load_and_process and _write_cache functions)

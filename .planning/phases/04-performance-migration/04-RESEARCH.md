# Phase 4: Performance & Migration - Research

**Researched:** 2026-01-21
**Domain:** Pandas to Polars migration, CSV performance, waveform caching
**Confidence:** HIGH

## Summary

This phase migrates CSV operations from pandas to Polars for 5-30x performance improvements, fixes several algorithmic inefficiencies (CSV double-reads, N+1 queries, regex recompilation), and implements waveform caching for instant repeat views.

The codebase uses pandas in three files: `csv_parser.py`, `uncertainty.py`, and `batch.py`. The migration is straightforward because the pandas usage is limited to CSV reading/writing and basic DataFrame operations. Polars provides drop-in replacements for all current operations with better performance.

**Primary recommendation:** Migrate all three files to Polars using lazy evaluation (`scan_csv`) where possible, cache waveform data as JSON files alongside MP3s, and pre-compile regex patterns at module level.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| polars | 1.x (latest) | DataFrame operations | 5-30x faster than pandas, lower memory usage, parallel execution |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pyarrow | (already installed) | Arrow memory format | Required by Polars for data interchange |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Polars | pandas 2.0 with Arrow backend | pandas 2.0 faster but still single-threaded, Polars 5-30x faster overall |
| Polars | DuckDB | DuckDB excellent for SQL queries, Polars better for DataFrame API |

**Installation:**
```bash
pip install polars
pip uninstall pandas  # After migration complete
```

## Architecture Patterns

### Pattern 1: Eager vs Lazy Loading

**What:** Polars supports eager (`read_csv`) and lazy (`scan_csv`) evaluation. Lazy mode enables query optimization.

**When to use:**
- Use `pl.read_csv()` for small files or when you need the full DataFrame immediately
- Use `pl.scan_csv().collect()` when filtering/selecting columns - Polars can push predicates down and skip reading unneeded data

**Example:**
```python
# Source: https://docs.pola.rs/user-guide/io/csv/
import polars as pl

# Eager - loads entire file (fine for small CSVs like predictions)
df = pl.read_csv("predictions.csv")

# Lazy - better for filtering (e.g., uncertainty stats)
df = (
    pl.scan_csv("predictions.csv")
    .filter(pl.col("confidence") < 0.7)
    .collect()
)
```

### Pattern 2: Row Access Without Index

**What:** Polars has no index. Use `df[row, col]` or `df.item(row, col)` for single values.

**When to use:** When migrating pandas `.iloc[]` access patterns

**Example:**
```python
# Source: https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.item.html

# Pandas
value = df.iloc[0]["column"]
value = df["column"].iloc[0]

# Polars equivalents
value = df[0, "column"]           # Square bracket indexing
value = df.item(0, "column")      # .item() method
value = df["column"][0]           # Column first, then row
value = df.row(0)[col_idx]        # Get tuple of row values
```

### Pattern 3: Null Handling (No NaN)

**What:** Polars uses `null` for missing data, not `NaN`. Use `is_null()`, `fill_null()`, `drop_nulls()`.

**When to use:** All null checking code

**Example:**
```python
# Source: https://docs.pola.rs/user-guide/expressions/missing-data/

# Pandas
if pd.isna(value):
    value = ""
df.fillna("")
df.dropna()

# Polars
if value is None:  # Polars returns Python None for null
    value = ""
df.fill_null("")
df.drop_nulls()
```

### Pattern 4: Column Selection and Filtering

**What:** Polars uses `select()` and `filter()` with expressions.

**Example:**
```python
# Source: https://docs.pola.rs/user-guide/migration/pandas/

# Pandas
df[df["confidence"] < 0.7]
df[["time", "class"]]

# Polars
df.filter(pl.col("confidence") < 0.7)
df.select(["time", "class"])
# Or with expressions:
df.select(pl.col("time"), pl.col("class"))
```

### Anti-Patterns to Avoid

- **Calling `read_csv().lazy()`:** This loads the entire file first, defeating lazy benefits. Always use `scan_csv()` for lazy mode.
- **Using `with_row_index()` for filtering:** Avoid `with_row_index().filter()` - use `slice()` instead for performance.
- **Confusing `null` and `NaN`:** Polars treats them differently. `fill_null()` does NOT fill NaN values.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Waveform caching | Custom cache logic | JSON file alongside MP3 | Simple, inspectable, survives restarts |
| Regex compilation | Compile per iteration | `re.compile()` at module level | Python caches 512 patterns but module-level is clearer |
| CSV double-read | Two separate reads | Single read + column check | I/O is expensive, memory is cheap |

**Key insight:** All performance issues in this phase are caused by repeated I/O. Fix by caching I/O results.

## Common Pitfalls

### Pitfall 1: Expecting pandas Index

**What goes wrong:** Code assumes `.iloc[0]` or uses `.loc` which don't exist in Polars
**Why it happens:** Mental model from pandas
**How to avoid:** Use `df[row, col]` or `df.item(row, col)` for single values
**Warning signs:** `AttributeError: 'DataFrame' object has no attribute 'iloc'`

### Pitfall 2: NaN vs Null Confusion

**What goes wrong:** `fill_null()` doesn't handle float NaN values
**Why it happens:** Polars separates these concepts; pandas conflates them
**How to avoid:** Use `fill_nan()` for NaN, `fill_null()` for null. Or convert NaN to null first with `fill_nan(None)`.
**Warning signs:** NaN values persist after `fill_null()`

### Pitfall 3: Forgetting to `.collect()` Lazy Frames

**What goes wrong:** Code expects DataFrame but gets LazyFrame, methods fail
**Why it happens:** `scan_csv()` returns LazyFrame, not DataFrame
**How to avoid:** Chain `.collect()` at end of lazy operations
**Warning signs:** `AttributeError: 'LazyFrame' object has no attribute 'iter_rows'`

### Pitfall 4: Column Name Case Sensitivity

**What goes wrong:** Column not found errors when columns exist
**Why it happens:** Polars is case-sensitive by default
**How to avoid:** Normalize column names after read: `df.columns = [c.lower() for c in df.columns]` or use schema
**Warning signs:** `ColumnNotFoundError` when column visibly exists

## Code Examples

### Current Pandas Patterns and Polars Equivalents

#### csv_parser.py - Main CSV Reading

```python
# CURRENT (csv_parser.py:197-199)
df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)
df.columns = [col.strip() for col in df.columns]

# POLARS EQUIVALENT
import polars as pl
df = pl.read_csv(csv_path, encoding='utf-8', quote_char='"')
# Note: Polars automatically strips column whitespace
```

#### csv_parser.py - Row Iteration and Value Access

```python
# CURRENT (csv_parser.py:108-118)
current_class = df.iloc[0][class_col]
start = df.iloc[0][time_col]
name_value = df.iloc[0][name_col]
if pd.isna(name_value):
    current_name = ""
else:
    current_name = str(name_value)

# POLARS EQUIVALENT
current_class = df[0, class_col]
start = df[0, time_col]
name_value = df[0, name_col]
current_name = "" if name_value is None else str(name_value)
```

#### csv_parser.py - Iterating DataFrame

```python
# CURRENT (csv_parser.py:122-164)
while index < len(df):
    if df.iloc[index][class_col] != current_class:
        # ...

# POLARS EQUIVALENT - Convert to list of dicts for iteration
rows = df.to_dicts()
for index, row in enumerate(rows):
    if row[class_col] != current_class:
        # ...

# OR: Use iter_rows for memory efficiency
for row in df.iter_rows(named=True):
    if row[class_col] != current_class:
        # ...
```

#### uncertainty.py - N+1 Query Fix

```python
# CURRENT (uncertainty.py:281-293) - Reads CSV TWICE
df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=1)  # First read
if 'confidence' not in df.columns:
    continue
full_df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # Second read!
if 'model_version' not in full_df.columns:
    model_version = "unknown"
else:
    model_version = full_df['model_version'].iloc[0]

# POLARS FIX - Single read
df = pl.read_csv(csv_file, encoding='utf-8')
if 'confidence' not in df.columns:
    continue
if 'model_version' not in df.columns:
    model_version = "unknown"
else:
    model_version = df[0, 'model_version']
```

#### batch.py - Same Double-Read Pattern

```python
# CURRENT (batch.py:394-400) - Same problem
df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=1)
if 'model_version' not in df.columns:
    csv_model_version = "unknown"
else:
    full_df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # Second read!
    csv_model_version = full_df['model_version'].iloc[0]

# POLARS FIX - Single read
df = pl.read_csv(csv_file, encoding='utf-8')
if 'model_version' not in df.columns:
    csv_model_version = "unknown"
else:
    csv_model_version = df[0, 'model_version']
```

#### files.py - Regex Pre-compilation

```python
# CURRENT (files.py:90) - Compiles regex per file
for csv_file in results_folder.glob("*.csv"):
    match = re.search(r'_(\d{4})-(\d{2})-(\d{2})', csv_file.name)

# FIX - Pre-compile at module level
DATE_PATTERN = re.compile(r'_(\d{4})-(\d{2})-(\d{2})')

for csv_file in results_folder.glob("*.csv"):
    match = DATE_PATTERN.search(csv_file.name)
```

### Waveform Caching Implementation

```python
# waveform.py - Cache waveform data as JSON

import json
import hashlib
from pathlib import Path

WAVEFORM_CACHE_DIR = settings.SORTED_FOLDER / ".waveform_cache"

def get_cache_path(mp3_path: Path, samples_per_pixel: int) -> Path:
    """Generate cache file path based on file path and params"""
    # Include file mtime in cache key for invalidation
    mtime = mp3_path.stat().st_mtime
    cache_key = f"{mp3_path}:{samples_per_pixel}:{mtime}"
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    return WAVEFORM_CACHE_DIR / f"{mp3_path.stem}_{hash_key}.json"

@router.get("/data")
async def get_waveform_data(
    path: str = Query(...),
    samples_per_pixel: int = Query(512)
):
    mp3_path = validate_path_or_raise_http(path, settings.SORTED_FOLDER)
    cache_path = get_cache_path(mp3_path, samples_per_pixel)

    # Check cache first
    if cache_path.exists():
        return JSONResponse(json.loads(cache_path.read_text()))

    # Generate waveform data
    y, sr = librosa.load(str(mp3_path), sr=8000, mono=True)
    # ... existing generation logic ...

    # Cache result
    WAVEFORM_CACHE_DIR.mkdir(exist_ok=True)
    result = {...}  # waveform data dict
    cache_path.write_text(json.dumps(result))

    return JSONResponse(result)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| pandas for all DataFrames | Polars for performance-critical paths | 2024-2025 | 5-30x speedup on CSV operations |
| Eager CSV loading | Lazy loading with `scan_csv` | Polars 0.14+ | Automatic query optimization |
| pandas `NaN` | Polars `null` (separate from NaN) | Polars design | Cleaner null handling |

**Deprecated/outdated:**
- **pandas for large CSV processing:** Polars is now preferred for performance-critical workloads
- **Manual multithreading with pandas:** Polars parallelizes automatically

## Open Questions

1. **Waveform cache location**
   - What we know: Cache should be near MP3 files or in a dedicated folder
   - What's unclear: Should cache be in `.waveform_cache` under SORTED_FOLDER or next to each MP3?
   - Recommendation: Use `.waveform_cache` folder - easier to manage, clear disk usage, can purge all at once

2. **Cache invalidation strategy**
   - What we know: Include file mtime in cache key handles file changes
   - What's unclear: Should cache have TTL or max size?
   - Recommendation: No TTL needed (MP3s don't change), no size limit for single-user app

3. **Async Polars operations**
   - What we know: Polars I/O is synchronous
   - What's unclear: Performance impact of running in asyncio.to_thread
   - Recommendation: Keep wrapping in `asyncio.to_thread()` - maintains non-blocking server behavior

## Sources

### Primary (HIGH confidence)
- [Polars Migration Guide](https://docs.pola.rs/user-guide/migration/pandas/) - API equivalents, key differences
- [Polars CSV I/O](https://docs.pola.rs/user-guide/io/csv/) - read_csv, write_csv, scan_csv usage
- [Polars DataFrame.item](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.item.html) - Single value access
- [Polars Missing Data](https://docs.pola.rs/user-guide/expressions/missing-data/) - null vs NaN handling

### Secondary (MEDIUM confidence)
- [Polars vs Pandas Benchmark](https://pipeline2insights.substack.com/p/pandas-vs-polars-benchmarking-dataframe) - Performance numbers (5-30x)
- [Peaks.js Waveform Caching](https://github.com/bbc/peaks.js) - Pre-computed waveform data patterns

### Tertiary (LOW confidence)
- [Librosa Caching](https://librosa.org/doc/main/cache.html) - Alternative cache approach (not used)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Polars is well-documented with clear migration path
- Architecture: HIGH - Patterns verified against official docs
- Pitfalls: HIGH - Common issues documented in migration guides
- Waveform caching: MEDIUM - Implementation approach is sound but not verified in similar codebases

**Research date:** 2026-01-21
**Valid until:** 60 days (Polars has stable API, pandas-to-polars migration patterns well established)

---

## Appendix: Files to Migrate

| File | Pandas Usage | Migration Complexity |
|------|--------------|---------------------|
| `backend/app/api/v1/csv_parser.py` | read_csv, iloc, isna, DataFrame iteration | MEDIUM - Core parsing logic needs careful migration |
| `backend/app/api/v1/uncertainty.py` | read_csv (4 places), iloc, DataFrame filtering | LOW - Mostly simple reads |
| `backend/app/api/v1/batch.py` | read_csv (2 places), iloc | LOW - Simple reads |

### Performance Issues to Fix

| Issue | Location | Fix |
|-------|----------|-----|
| PERF-01: CSV double-read | batch.py:394-400 | Read once, check columns after |
| PERF-02: N+1 query (same issue) | uncertainty.py:281-293 | Read once, check columns after |
| PERF-03: Regex recompilation | files.py:90 | Pre-compile at module level |
| PERF-04: Waveform caching | waveform.py (new) | Cache JSON to filesystem |

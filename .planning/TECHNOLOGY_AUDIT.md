# Technology Audit: Filharmonia AI

**Date:** 2026-01-20
**Scope:** Full stack review with migration recommendations

---

## Executive Summary

The current stack is **solid and well-chosen** for its purpose. Most technologies are appropriate and do not need replacement. The highest-impact improvements are:

1. **pandas -> polars** for CSV processing (10x+ speedup, easy migration)
2. **Custom waveform -> wavesurfer.js** for better UX (medium effort)
3. **React 18 -> React 19** for better performance (low effort)

The ML stack (PyTorch + AST) and backend framework (FastAPI) are good choices that should remain.

---

## 1. Backend Framework: FastAPI

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | FastAPI 0.115.0 | Litestar | **KEEP CURRENT** |
| **Performance gain** | baseline | ~1.2-1.5x for pure JSON | Marginal |
| **Migration effort** | - | Medium | Not worth it |

### Analysis

**FastAPI advantages in this project:**
- Excellent Pydantic integration (already using v2.11.9)
- Native async support matches audio file I/O patterns
- Large ecosystem, better documentation
- Team familiarity likely

**Litestar benchmarks:**
- msgspec is ~12x faster than Pydantic V2 for serialization
- Higher throughput under heavy load
- BUT: Filharmonia is not serialization-bound; it's I/O and ML-bound

**Why keep FastAPI:**
The bottlenecks in Filharmonia are:
1. Audio file loading (librosa)
2. ML inference (AST model)
3. File system operations

Switching to Litestar would give negligible real-world improvement because the API layer is not the bottleneck. FastAPI's ecosystem and documentation advantage outweighs the marginal performance gain.

**Priority:** Keep current

**Sources:**
- [Litestar vs FastAPI - Better Stack](https://betterstack.com/community/guides/scaling-python/litestar-vs-fastapi/)
- [FastAPI Benchmarks](https://fastapi.tiangolo.com/benchmarks/)

---

## 2. Audio Loading: librosa

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | librosa 0.10.2 | torchaudio / soundfile | **PARTIAL MIGRATION** |
| **Performance gain** | baseline | 2-3x for specific cases | Varies by use case |
| **Migration effort** | - | Easy | Worth it for waveform endpoint |

### Current Usage Analysis

Librosa is used in 5 places:
```python
# analyze.py - Full audio loading for inference (48kHz)
y, sr = librosa.load(str(mp3_path), sr=settings.SAMPLE_RATE)

# waveform.py - Waveform data generation (8kHz)
y, sr = librosa.load(str(mp3_path), sr=8000, mono=True)

# export.py - Training data export (44.1kHz stereo)
y, sr = librosa.load(str(mp3_path), sr=44100, mono=False)

# uncertainty.py - Same as export
y, sr = librosa.load(str(mp3_path), sr=44100, mono=False)

# training.py - Training data loading + mel-spectrogram generation
signal, sr = librosa.load(...)
melspec = librosa.feature.melspectrogram(...)
```

### Benchmark Reality

**Key insight:** librosa uses soundfile under the hood for most formats. The performance difference is primarily from:
1. **Default resampling** - librosa defaults to 22050Hz, which adds processing
2. **Data transposition** - librosa returns `(samples,)` vs `(channels, samples)`

**Actual timing (from faroit/python_audio_loading_benchmark):**
- For WAV files: scipy and torchaudio are fastest
- For MP3 files: torchaudio (sox_io backend) is ~2x faster than librosa
- For FLAC files: torchaudio is fastest

### Recommendations by Use Case

| Use Case | Current | Recommendation | Reason |
|----------|---------|----------------|--------|
| Inference audio | librosa | **Keep** | Already matches training pipeline |
| Waveform data | librosa | **Use torchaudio** | 2x faster for MP3, already have torchaudio |
| Training | librosa | **Use torchaudio** | Already using torchaudio for mel-spectrogram |
| Export | librosa | Keep | Not performance-critical |

**Migration approach:**
```python
# For waveform.py - simple replacement
import torchaudio
waveform, sr = torchaudio.load(str(mp3_path))
if sr != 8000:
    resampler = torchaudio.transforms.Resample(sr, 8000)
    waveform = resampler(waveform)
y = waveform.mean(dim=0).numpy()  # mono
```

**Priority:** Migrate later (waveform endpoint only)

**Sources:**
- [Python Audio Loading Benchmark](https://github.com/faroit/python_audio_loading_benchmark)
- [Comparing Librosa, Soundfile and Torchaudio](https://nasseredd.github.io/blog/speech-and-language-processing/comparing-audio-libraries)

---

## 3. CSV Processing: pandas

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | pandas 2.2.0 | Polars | **MIGRATE** |
| **Performance gain** | baseline | 5-30x faster | Significant |
| **Migration effort** | - | Easy | High value/effort ratio |
| **Memory reduction** | baseline | 2-8x less | Significant |

### Current Usage Analysis

pandas is used for CSV parsing in:
- `csv_parser.py` - Parse prediction CSVs into tracks
- `uncertainty.py` - Read segment data for uncertainty review
- `batch.py` - Batch processing CSV reading

**Typical file sizes:** 1,000-10,000 rows (one row per second of audio)

### Benchmark Data (2026)

| Operation | pandas | Polars | Speedup |
|-----------|--------|--------|---------|
| Read 1GB CSV | ~14s | ~2.8s | **5x** |
| Read 20M rows | ~50s | ~1s | **50x** |
| Read 17M rows | 87s | 7.8s | **11x** |
| Filter 100M rows | 9.45s | 1.89s | **5x** |

**Memory usage:**
- pandas: 1.4GB for 1GB CSV
- Polars: 179MB for same file (**8x reduction**)

### Why Migrate

1. **Parallelism:** Polars auto-parallelizes; pandas is single-threaded
2. **Lazy evaluation:** Process only needed columns
3. **Memory efficiency:** Columnar format, streaming support
4. **API similarity:** Migration is straightforward

### Migration Example

```python
# Before (pandas)
import pandas as pd
df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)

# After (Polars)
import polars as pl
df = pl.read_csv(csv_path, encoding='utf-8', quote_char='"')
```

**Common conversions:**
```python
# pd.isna(value) -> value is None (in Polars)
# df.iloc[i][col] -> df[i, col] or df.row(i)[col_idx]
# df.columns -> df.columns (same)
```

**Priority:** Migrate now (high impact, low effort)

**Sources:**
- [Polars Official Benchmarks](https://pola.rs/posts/benchmarks/)
- [Pandas vs Polars Benchmark 2026](https://medium.com/@mohanaslvn/how-fast-is-polars-really-a-basic-performance-test-20f07f723ff4)
- [DataCamp: pandas 2.0 vs polars](https://www.datacamp.com/tutorial/high-performance-data-manipulation-in-python-pandas2-vs-polars)

---

## 4. Browser Audio Playback: howler.js

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | howler 2.2.4 | Native HTML5 Audio | **ALREADY USING HTML5** |
| **Performance gain** | N/A | N/A | N/A |
| **Migration effort** | - | None needed | - |

### Current Usage Analysis

**Surprise finding:** The codebase does NOT actually use howler.js for playback!

```tsx
// StickyPlayer.tsx - Uses native HTML5 Audio
<audio
  ref={audioRef}
  src={audioUrl}
  preload="auto"
  onPlay={() => setIsPlaying(true)}
  onPause={() => setIsPlaying(false)}
  ...
/>
```

The `howler` package is in dependencies but appears **unused**. The app uses native `<audio>` element.

### Recommendation

1. **Remove howler.js** from dependencies (save 7KB)
2. **Keep native HTML5 Audio** - works perfectly for this use case
3. If advanced features needed later (sprites, 3D audio), howler.js is the right choice

**Priority:** Remove unused dependency

**Sources:**
- [Howler.js](https://howlerjs.com/)
- [Web Audio API Best Practices](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices)

---

## 5. Waveform Visualization: waveform-data

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | waveform-data 4.5.2 + custom canvas | wavesurfer.js | **CONSIDER MIGRATION** |
| **Features gained** | Basic | Regions, timeline, hover, plugins | Significant |
| **Migration effort** | - | Medium | Worth it for UX |

### Current Implementation Analysis

The current approach:
1. Backend generates waveform min/max data via librosa
2. Frontend draws to `<canvas>` manually
3. Custom mouse handling for regions, markers, playhead

**Strengths:**
- Lightweight (no large library)
- Full control over rendering

**Weaknesses:**
- ~500 lines of custom canvas code
- Reimplementing standard features (regions, markers, zoom)
- Duplicated drawing logic in `useEffect` handlers

### Alternatives Comparison

| Feature | Current | wavesurfer.js | peaks.js |
|---------|---------|---------------|----------|
| Waveform rendering | Manual | Built-in | Built-in |
| Regions/markers | Manual | Plugin | Built-in |
| Zoom | Manual | Built-in | Built-in |
| Timeline | None | Plugin | Built-in |
| Spectrogram | None | Plugin | No |
| Pre-computed peaks | Custom | Supported | Required |
| Bundle size | ~5KB | ~50KB | ~60KB |

### Recommendation

**wavesurfer.js v7** is the better choice because:
1. Active development (v7 uses Shadow DOM for CSS isolation)
2. Rich plugin ecosystem (Regions, Timeline, Minimap, Hover)
3. Built-in support for pre-computed peaks
4. Better accessibility

**Migration path:**
```typescript
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/plugins/regions'

const wavesurfer = WaveSurfer.create({
  container: containerRef.current,
  waveColor: '#4b5563',
  progressColor: '#3b82f6',
  peaks: waveformData.data.map(p => [p.min, p.max]),
  duration: waveformData.duration,
})
```

**Priority:** Migrate later (medium effort, good UX improvement)

**Sources:**
- [wavesurfer.js](https://wavesurfer.xyz/)
- [peaks.js](https://github.com/bbc/peaks.js)
- [wavesurfer.js FAQ](https://wavesurfer.xyz/faq/)

---

## 6. Frontend Framework/Libraries

### React Version

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | React 18.3.1 | React 19 | **UPGRADE** |
| **Performance gain** | baseline | 10-20% rendering | Moderate |
| **Migration effort** | - | Low | Easy |

**React 19 advantages:**
- Automatic compiler optimization (no manual `useMemo`/`useCallback`)
- `useActionState` and `useOptimistic` for forms
- Better hydration for SSR
- Cleaner code without memoization boilerplate

**Migration:** Straightforward - React 19 is backward compatible. Upgrade to 18.3 first to see deprecation warnings.

### Other Frontend Libraries

| Library | Current | Recommendation | Notes |
|---------|---------|----------------|-------|
| @tanstack/react-query 5.28.0 | **Keep** | Best-in-class server state | |
| zustand 4.5.2 | **Keep** | Perfect for this scale | |
| axios 1.6.7 | Keep or **fetch** | Native fetch is fine | Axios adds 13KB |
| Tailwind 3.4.1 | **Keep** | Could upgrade to v4 | |
| recharts 2.12.0 | **Keep** | Good for simple charts | |
| lucide-react 0.344.0 | **Keep** | Good icon library | |

**Priority:** Upgrade React (low effort)

**Sources:**
- [React 19 Upgrade Guide](https://react.dev/blog/2024/04/25/react-19-upgrade-guide)
- [React 18 vs 19 Comparison](https://dev.to/manojspace/react-18-vs-react-19-key-differences-and-migration-tips-18op)

---

## 7. Job Registry: SQLite

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | SQLite (via SQLAlchemy) | PostgreSQL / Redis | **KEEP CURRENT** |
| **Performance** | Adequate | Marginally better | Not needed |
| **Migration effort** | - | Medium-High | Not worth it |

### Current Usage Analysis

SQLite is configured but appears minimally used. The job registry seems to be file-based (CSV files in `.claude/` directory).

### Analysis

**SQLite is appropriate when:**
- Single-server deployment
- < 100 concurrent writers
- Data fits in memory
- Simplicity > scalability

**Filharmonia context:**
- Single user (orchestra archivist)
- Batch jobs are sequential
- Job state is simple (pending/running/complete)
- No distributed workers

### Recommendation

**Keep SQLite** because:
1. Zero operational overhead
2. Adequate performance for single-user workload
3. No benefit from PostgreSQL complexity
4. Redis adds infrastructure for minimal gain

**If scaling needed later:**
- PostgreSQL with `SKIP LOCKED` for job queues
- Or `huey` with SQLite backend

**Priority:** Keep current

**Sources:**
- [Choose Postgres Queue Technology](https://adriano.fyi/posts/2023-09-24-choose-postgres-queue-technology/)
- [PostgreSQL vs Redis for Queues](https://spin.atomicobject.com/redis-postgresql/)

---

## 8. ML Model: Audio Spectrogram Transformer (AST)

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| **Technology** | AST (MIT/ast-finetuned-audioset) | FastAST / MAST | **KEEP CURRENT** |
| **Accuracy** | 95.6% (ESC-50 benchmark) | Similar or lower | Trade-off |
| **Speed** | ~100ms/segment (GPU) | 2x faster possible | Consider if needed |
| **Migration effort** | - | High (retrain) | Not recommended |

### Current Implementation

The codebase uses AST with:
- Pre-trained weights from `MIT/ast-finetuned-audioset-10-10-0.4593`
- Fine-tuned on 5 classes: APPLAUSE, MUSIC, PUBLIC, SPEECH, TUNING
- Custom training pipeline with torchaudio mel-spectrograms

### Faster Alternatives

| Model | Speed vs AST | Accuracy | Notes |
|-------|--------------|----------|-------|
| **FastAST** | 2x faster | ~Same | Token merging, knowledge distillation |
| **MAST** | 5x fewer MACs | +4.4% on VGGSound | Multi-scale architecture |
| **MAE-AST** | 3x faster pretraining | Better fine-tuning | Masked autoencoding |
| **EfficientNet-based** | Much faster | Lower | CNN, not transformer |

### Recommendation

**Keep AST** because:
1. Model is already trained and working
2. Inference time (~100ms/segment) is acceptable
3. FastAST requires retraining with knowledge distillation
4. Accuracy is more important than speed for archiving

**Consider FastAST if:**
- Real-time classification is needed
- Batch processing becomes bottleneck
- GPU memory is constrained

**Optimization without model change:**
```python
# Already implemented: batch processing
predictions = service.predict_batch(audio_segments)  # More efficient

# Potential: TorchScript compilation
model = torch.jit.script(model)  # 10-20% faster inference
```

**Priority:** Keep current

**Sources:**
- [AST Paper](https://arxiv.org/abs/2104.01778)
- [FastAST](https://arxiv.org/html/2406.07676v1)
- [MAST](https://www.amazon.science/publications/multiscale-audio-spectrogram-transformer-for-efficient-audio-classification)

---

## Summary: Migration Priority Matrix

| Technology | Action | Impact | Effort | Priority |
|------------|--------|--------|--------|----------|
| pandas -> polars | **MIGRATE** | High (5-30x faster) | Easy | **NOW** |
| React 18 -> 19 | **UPGRADE** | Medium (cleaner code) | Easy | **NOW** |
| Remove howler.js | **REMOVE** | Low (save 7KB) | Trivial | **NOW** |
| Custom waveform -> wavesurfer.js | Consider | Medium (better UX) | Medium | **LATER** |
| librosa -> torchaudio (waveform) | Consider | Low (2x faster) | Easy | **LATER** |
| FastAPI | Keep | - | - | - |
| SQLite | Keep | - | - | - |
| AST model | Keep | - | - | - |

---

## Immediate Actions

### 1. Install Polars and Migrate CSV Processing (1-2 hours)

```bash
cd backend
pip install polars
```

Migration pattern for `csv_parser.py`:
```python
# Replace
import pandas as pd
df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)

# With
import polars as pl
df = pl.read_csv(csv_path, encoding='utf-8', quote_char='"')
```

### 2. Upgrade React to v19 (30 minutes)

```bash
cd frontend
pnpm update react@19 react-dom@19
pnpm update @types/react@19 @types/react-dom@19
```

### 3. Remove Unused howler.js (5 minutes)

```bash
cd frontend
pnpm remove howler
pnpm remove @types/howler  # if exists
```

---

## Deferred Actions (Post-MVP)

1. **wavesurfer.js migration** - When adding new waveform features
2. **torchaudio for waveform** - If waveform loading becomes a bottleneck
3. **FastAST evaluation** - If real-time classification is needed
4. **Tailwind v4** - When starting new features

---

*Audit completed: 2026-01-20*

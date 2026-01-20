# Research Summary: Filharmonia AI Polish & Stability Milestone

**Project:** Filharmonia AI - Audio Analysis Application
**Synthesized:** 2026-01-20
**Research Files:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md

---

## Executive Summary

Filharmonia AI is a production audio annotation tool built on a solid foundation (React + FastAPI + PyTorch/AST), but suffering from classic brownfield issues: bare exception blocks, hardcoded paths, file-based job tracking, and a 1200+ line component. The research reveals that **the core architecture is sound** - no technology swap is needed. The path to stability is incremental refactoring focused on error handling, cross-platform path management, and component decomposition.

For cross-platform PyTorch deployment, the recommended strategy is a **tiered inference approach**: native PyTorch with `torch.compile` for GPU (CUDA/ROCm), ONNX Runtime with quantization for CPU optimization, and eager PyTorch as fallback. The existing codebase already handles some tricky patterns correctly (torchaudio preprocessing consistency, `map_location` for checkpoint loading), but has gaps in error handling, path portability, and job lifecycle management.

The biggest risks are: (1) ROCm on Windows is preview-only with limited GPU support - prioritize Linux for AMD, (2) CPU inference without optimization is 10-50x slower than GPU - ONNX quantization is essential for the CPU-only server, and (3) the 11+ bare `except:` blocks make debugging impossible. The critical user experience gaps are missing keyboard shortcuts (spacebar for play/pause) and no undo/redo capability.

---

## Key Findings

### From STACK.md: Technology Recommendations

| Category | Recommendation | Rationale |
|----------|---------------|-----------|
| **Core Framework** | PyTorch 2.9.x + torchaudio | Wheel variants simplify multi-platform install; `torch.compile` mature |
| **GPU (NVIDIA)** | CUDA 12.6-13.0 | Full support for RTX 3000/4000/5000 series |
| **GPU (AMD)** | ROCm 6.4 on Linux | Windows ROCm is preview only; Linux more stable |
| **CPU Optimization** | ONNX Runtime 1.20+ | 20-30% faster than eager, quantization support |
| **Model Export** | optimum-onnx 2.0+ | Best for HuggingFace models, includes quantization |

**Critical Version Requirement:** Pin exact PyTorch versions with CUDA/ROCm suffix (e.g., `torch==2.9.1+cu126`) to avoid dependency hell.

### From FEATURES.md: Table Stakes vs Differentiators

**Must-Have (Table Stakes Gaps):**
1. **Keyboard shortcuts** - Spacebar play/pause is universal audio convention; currently missing
2. **Specific error messages** - Replace bare "Error" with actionable feedback
3. **Undo/redo** - Users fear making changes without reversal; only "Discard all" exists
4. **Waveform caching** - 30-40 min files regenerate waveform on every load

**Should-Have (Differentiators):**
- One-click workflow (Sort + Analyze + Open)
- Quick class cycling with number keys
- Drag boundary handles for segment adjustment

**Defer to v2+:**
- Time estimates, minimap, export format selection, spectrogram view

### From ARCHITECTURE.md: Structural Patterns

**Key Patterns to Implement:**

1. **Error Handling:** Replace bare `except:` with specific exceptions + logging + global exception handler
2. **Job Registry:** SQLite replaces file-based tracking (persists across reboots, no race conditions, cleanup)
3. **Component Decomposition:** Extract `useTrackEditor`, `useAudioPlayer`, `useAutosave` hooks from 1268-line CsvViewer
4. **Cross-Platform Paths:** Use `tempfile.gettempdir()` not hardcoded `/tmp/`; backend resolves all paths

**Strangler Fig Strategy:** Never stop delivery - wrap and replace incrementally, not rewrite.

### From PITFALLS.md: Critical Risks

| Severity | Pitfall | Prevention |
|----------|---------|------------|
| **CRITICAL** | ROCm Windows is preview-only | Prioritize Linux for AMD; always have CPU fallback |
| **CRITICAL** | CPU inference 10-50x slower | Implement ONNX quantization (INT8 = 4x speedup) |
| **CRITICAL** | Checkpoint device mismatch | Always use `map_location='cpu'` when loading |
| **MODERATE** | Version dependency hell | Pin exact versions with platform suffix |
| **MODERATE** | torchaudio vs librosa inconsistency | Document preprocessing; never mix libraries |
| **MINOR** | Sample rate mismatch | Add validation and auto-resampling |

**Already Correct in Codebase:**
- `model.eval()` and `torch.no_grad()` during inference
- `map_location=self.device` in checkpoint loading
- torchaudio used consistently (not mixed with librosa)
- pathlib used for most file operations

---

## Implications for Roadmap

Based on the combined research, here is the recommended phase structure:

### Phase 1: Foundation Stability

**Rationale:** Fix observability and cross-platform issues first. Every subsequent phase benefits from better error messages and portable paths. Low risk, high value.

**Delivers:**
- Replace 11+ bare `except:` blocks with specific exception handling
- Add global FastAPI exception handler with error IDs
- Centralize path configuration using `tempfile.gettempdir()`
- Move path resolution to backend (frontend never constructs file paths)
- Pin dependency versions with platform suffixes
- Add startup validation for audio backends and sample rates

**Features from FEATURES.md:** Specific error messages, graceful degradation

**Pitfalls to Avoid:** #1 Device detection, #5 Version pinning, #8 Path handling

**Research Flag:** Standard patterns - no additional research needed

---

### Phase 2: Core UX Polish

**Rationale:** Address the most glaring UX gaps that make the tool feel unfinished. Keyboard shortcuts are expected in any audio tool; undo/redo is table stakes for editing.

**Delivers:**
- Keyboard shortcuts (spacebar play/pause, Ctrl+S save, Ctrl+Z undo, number keys for class cycling)
- Undo/redo for segment edits (10-20 step buffer)
- Confirmation dialogs for destructive actions
- Progress stage indicators ("Loading... Analyzing... Saving...")

**Features from FEATURES.md:** Keyboard shortcuts (HIGH), Undo/redo (HIGH), Progress improvements (MEDIUM)

**Pitfalls to Avoid:** None specific - this is UX layer

**Research Flag:** Standard patterns - React state management for undo is well-documented

---

### Phase 3: Backend Stability

**Rationale:** Replace file-based job tracking with SQLite, add cleanup. This enables reliable job history and prevents tmp directory accumulation. Medium effort but significant stability improvement.

**Delivers:**
- SQLite job registry (replaces JSON files in /tmp)
- Startup cleanup for stale jobs
- Network retry logic with exponential backoff
- Worker process communication improvements

**Features from FEATURES.md:** Network error recovery (HIGH), Cancellation option (MEDIUM)

**Pitfalls to Avoid:** #11 CUDA fork safety if adding multiprocessing

**Research Flag:** SQLite patterns are standard; may need light research on IPC best practices if using multiprocessing.Queue

---

### Phase 4: Frontend Decomposition

**Rationale:** CsvViewer at 1268 lines is a maintenance liability. Extract hooks and split components. Lower priority because it's maintainability, not user-facing.

**Delivers:**
- Extract `useTrackEditor` hook
- Extract `useAudioPlayer` hook
- Extract `useAutosave` hook
- Split CsvViewer into CsvFileList, TrackTable, TrackRow, AudioControls
- Waveform caching (IndexedDB or filesystem cache)

**Features from FEATURES.md:** Waveform caching (HIGH), Sub-second UI response (MEDIUM)

**Pitfalls to Avoid:** None specific

**Research Flag:** Standard React patterns - no additional research needed

---

### Phase 5: GPU Backend Support

**Rationale:** Multi-backend inference (CUDA/ROCm/CPU). Deferred until foundation is stable because ROCm Windows is preview-only and requires hardware testing.

**Delivers:**
- Unified device detection with logging (distinguishes NVIDIA from AMD)
- `torch.compile` integration for GPU acceleration
- ROCm 6.4 support on Linux
- Backend selection strategy (GPU > ONNX > eager fallback)

**Features from FEATURES.md:** N/A (performance, not features)

**Pitfalls to Avoid:** #1 Device naming (ROCm uses cuda API), #2 Checkpoint format, #3 ROCm Windows limitations

**Research Flag:** NEEDS RESEARCH - Test on actual AMD hardware before implementing. ROCm compatibility varies by GPU generation.

---

### Phase 6: CPU Optimization

**Rationale:** The CPU-only Linux server needs optimized inference. Without this, analysis takes minutes instead of seconds. Requires ONNX export and quantization.

**Delivers:**
- ONNX model export with optimum-onnx
- INT8 quantization for 4x CPU speedup
- ONNX Runtime inference backend
- Threading optimization
- Fallback to eager mode if ONNX fails

**Features from FEATURES.md:** N/A (performance)

**Pitfalls to Avoid:** #4 CPU inference slowness, #6 Preprocessing consistency (ONNX must match torchaudio)

**Research Flag:** NEEDS RESEARCH - Validate ONNX export preserves accuracy; test quantized model quality

---

## Research Flags

| Phase | Needs Research? | Reason |
|-------|-----------------|--------|
| Phase 1: Foundation | No | Standard Python/FastAPI patterns |
| Phase 2: UX Polish | No | Standard React patterns |
| Phase 3: Backend | Light | IPC patterns if using multiprocessing |
| Phase 4: Frontend | No | Standard React hooks/composition |
| Phase 5: GPU Backends | **Yes** | ROCm hardware compatibility varies; need real testing |
| Phase 6: CPU Optimization | **Yes** | ONNX export accuracy validation; quantization quality testing |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | HIGH | Official PyTorch/HuggingFace docs verified; version recommendations solid |
| **Features** | HIGH | Based on established UX best practices and annotation tool standards |
| **Architecture** | HIGH | Patterns from official React/FastAPI docs + Martin Fowler |
| **Pitfalls** | HIGH | Official documentation + AMD release notes; some community sources |

### Gaps to Address During Planning

1. **AMD Hardware Testing:** ROCm compatibility claims need validation on actual RX 7000/9000 hardware
2. **ONNX Accuracy:** Need to verify quantized ONNX model matches PyTorch accuracy before deploying to CPU server
3. **Waveform Caching Strategy:** Research didn't specify optimal cache implementation (IndexedDB vs filesystem vs server-side)
4. **Batch Processing Performance:** Current research focused on single-file; batch performance patterns less clear

---

## Cross-Cutting Themes

**Theme 1: Observability First**
Multiple research areas point to the same conclusion: you can't fix what you can't see. Error handling, logging, device detection logging, job status tracking - all require better visibility before optimization.

**Theme 2: Platform Portability**
Path handling, device detection, audio backends, ROCm vs CUDA - the codebase runs on Windows dev machines, AMD workstations, and a Linux server. Every decision must be cross-platform or have explicit fallbacks.

**Theme 3: Incremental Over Rewrite**
The Strangler Fig pattern appears in architecture research; the anti-features list warns against over-engineering. Both point to the same wisdom: small, deployable changes beat big rewrites.

**Theme 4: CPU is the Bottleneck**
For the CPU-only Linux server deployment, inference performance is a critical gap. The 10-50x slowdown without optimization makes the tool unusable for batch processing. ONNX quantization is essential, not optional.

---

## Tensions and Tradeoffs

**ROCm vs Stability:** Supporting AMD GPUs adds complexity, and Windows ROCm is unstable. Trade-off: Implement ROCm on Linux only; CPU fallback for Windows AMD.

**Undo Complexity vs Scope:** Full undo/redo requires command pattern, state snapshots. Trade-off: Start with single-step undo; defer infinite undo.

**ONNX Accuracy vs Speed:** INT8 quantization speeds up CPU but may lose accuracy. Trade-off: Test accuracy drop; keep FP32 ONNX as fallback if needed.

**SQLite vs Simplicity:** Adding SQLite increases dependencies. Trade-off: Worth it for persistence, cleanup, and queryability. No Redis/Celery needed.

---

## Sources

**Official Documentation (HIGH confidence):**
- PyTorch CUDA/ROCm Semantics
- FastAPI SQL Databases Tutorial
- React.dev Custom Hooks
- HuggingFace Optimum ONNX

**Technical Articles (MEDIUM confidence):**
- AMD ROCm Release Notes
- Martin Fowler - Modularizing React Applications
- Shopify Engineering - Strangler Fig Pattern

**Community/Blogs (verify before relying):**
- Various CPU optimization benchmarks
- ONNX vs PyTorch speed comparisons

See individual research files for complete source lists.

---

*Synthesis completed: 2026-01-20*

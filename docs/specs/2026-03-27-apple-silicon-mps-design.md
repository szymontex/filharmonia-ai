# Apple Silicon MPS Support — Design Spec

## Goal

Add Apple Silicon GPU (MPS) support to Filharmonia AI for inference and training,
with smart auto-tuning of threads and batch sizes. Zero regressions on CUDA/ROCm/CPU.

## Approach

**Approach C: MPS + smart auto-tuning.** MPS is a first-class PyTorch device —
the codebase already uses `model.to(device)` and `tensor.to(device)` everywhere.
Adding MPS means extending device detection, adjusting thread/batch parameters,
and handling MPS-specific quirks.

## Architecture

### Device hierarchy (detection order)

```
1. NVIDIA CUDA  → device_type="cuda_nvidia"  → PyTorch GPU path
2. AMD ROCm     → device_type="cuda_amd"     → PyTorch GPU path (validated)
3. Apple MPS    → device_type="mps"           → PyTorch MPS path (NEW)
4. CPU          → device_type="cpu"           → ONNX INT8 or PyTorch CPU
```

MPS is detected via `torch.backends.mps.is_available()` (PyTorch 1.12+).
Falls back to CPU if MPS is present but fails a quick validation.

### Inference routing (inference_factory.py)

```
CUDA GPU    → ASTInferenceService (PyTorch)
AMD GPU     → ASTInferenceService (PyTorch)
Apple MPS   → ASTInferenceService (PyTorch)      ← NEW
CPU + ONNX  → ASTInferenceONNXService (INT8)
CPU only    → ASTInferenceService (PyTorch eager)
```

MPS routes to PyTorch (not ONNX) because MPS accelerates the model directly.
ONNX CoreML provider is explicitly out of scope — can be added later.

### MPS validation (startup benchmark)

At startup, run a quick 256x256 matmul on MPS vs CPU. If MPS is slower
(broken driver, unsupported op fallback), fall back to CPU + ONNX.
Reuses the existing `validate_gpu_performance()` pattern from ROCm.

### Thread and batch auto-tuning

**Problem:** Current code hardcodes `torch.set_num_threads(2)` and batch_size=32.
On an M4 Pro (12 P-cores) this wastes 10 cores. On a Raspberry Pi it's fine.

**Solution:** Detect platform and set reasonable defaults:

| Platform | Threads | Batch Size | Rationale |
|----------|---------|------------|-----------|
| CUDA GPU | 2 | 32 | GPU does heavy lifting, CPU just feeds data |
| MPS | min(P-cores, 6) | 64 | Unified memory = no transfer cost, more cores available |
| CPU | min(cores/2, 4) | 32 | Leave headroom for web server |

Core count detection:
- macOS: `sysctl -n hw.perflevel0.physicalcpu` (P-cores only)
- Linux: `os.cpu_count()` (total logical cores, divide by 2)
- Fallback: `os.cpu_count() or 4`

Batch size for MPS is larger because Apple Silicon unified memory eliminates
the CPU↔GPU data transfer bottleneck. Segments are already in shared memory.

## Files to change

### 1. `app/core/device_manager.py` — MPS detection + core count

- Add MPS detection after CUDA check, before CPU fallback
- Add `_validate_mps()` with matmul benchmark (reuse pattern from `_validate_rocm`)
- Add `cpu_core_count` property (platform-aware)
- Add `recommended_threads` and `recommended_batch_size` properties
- `is_gpu` returns True for MPS
- `supports_compile` returns False for MPS (Triton not available on Metal)

### 2. `app/services/ast_inference.py` — MPS-aware inference

- Replace hardcoded `torch.set_num_threads(2)` with device_manager.recommended_threads
- Move mel_transform to device for MPS: `self.mel_transform.to(self.device)`
- In `_apply_torch_compile()`: skip for MPS (no Triton), use torch.synchronize
  instead of torch.cuda.synchronize where needed
- In `preprocess_audio_segment()`: keep preprocessing on CPU (numpy input),
  move final tensor to device at inference time (already does this)

### 3. `app/services/inference_factory.py` — MPS routing

- MPS detected → use ASTInferenceService (same as GPU path)
- Log "Apple MPS" instead of "GPU"

### 4. `app/services/ast_training.py` — MPS training

- Model and data already go to `self.device` — works with MPS
- Handle MPS quirk: `pin_memory=False` when device is MPS
- Handle MPS quirk: some ops need float32 (MPS doesn't support float16 for all ops)
- Increase batch_size when on MPS (use recommended_batch_size)

### 5. `app/workers/analyze_worker.py` — thread count

- Replace hardcoded thread limits with device_manager values

### 6. `app/main.py` — startup logging

- Log MPS device info: chip name, core count, memory
- Log chosen thread count and batch size
- Add numpy Accelerate check: `numpy.show_config()`

### 7. `app/api/v1/analyze.py` — batch size

- Use device_manager.recommended_batch_size instead of hardcoded 32

## MPS-specific quirks to handle

1. **No pin_memory:** `DataLoader(pin_memory=True)` causes errors on MPS.
   Fix: `pin_memory = (device.type == "cuda")`

2. **No torch.cuda.synchronize:** MPS uses `torch.mps.synchronize()`.
   Fix: helper function `sync_device(device)`.

3. **Float precision:** Some MPS ops don't support float16.
   Fix: Keep float32 everywhere (already the case in this codebase).

4. **torch.compile:** Not supported on MPS (requires Triton/CUDA).
   Fix: `supports_compile` returns False for MPS.

5. **Fallback ops:** Some PyTorch ops fall back to CPU on MPS silently.
   Fix: Startup benchmark catches this pattern.

## What does NOT change

- CUDA path: untouched
- ROCm path: untouched
- CPU path: untouched
- ONNX path: untouched
- Audio preprocessing pipeline: same ops, same output
- Model architecture: same AST, same weights
- Training hyperparameters: same lr, epochs, early stopping
- API endpoints: same request/response format
- Frontend: no changes

## Testing

1. Start server on Mac → verify MPS detected in logs
2. Analyze an audio file → verify inference uses MPS
3. Check health endpoint → verify device info shows Apple Silicon
4. Compare inference time: before (CPU) vs after (MPS)
5. If training data available: start training, verify MPS is used
6. Verify existing CUDA/CPU paths are not affected (code review)

## Out of scope

- CoreML / Neural Engine integration (separate future effort)
- ONNX CoreML provider
- Mixed precision (float16) on MPS
- Multi-GPU / distributed training

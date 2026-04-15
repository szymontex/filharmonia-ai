---
phase: 06-gpu-a-cpu-optimization
plan: 04
subsystem: inference-device
tags: [rocm, amd, gpu, device-detection, documentation]
dependency-graph:
  requires: [06-01]
  provides: [rocm-detection, rocm-validation, rocm-docs]
  affects: []
tech-stack:
  added: []
  patterns: [silent-fallback-detection, timing-based-validation]
key-files:
  created:
    - docs/ROCM_SETUP.md
  modified:
    - backend/app/core/device_manager.py
    - backend/requirements.txt
decisions:
  - Timing-based GPU validation (256x256 matmul GPU vs CPU comparison) to detect silent ROCm CPU fallback
  - CRITICAL log level for silent fallback warning (not just WARNING) since it indicates broken GPU setup
  - ROCm Windows documented as experimental with graceful degradation
metrics:
  duration: ~3 minutes
  completed: 2026-01-29
---

# Phase 6 Plan 4: ROCm 6.4 Support Summary

ROCm-aware device detection with timing-based silent CPU fallback validation and multi-platform setup documentation.

## What Was Done

### Task 1: ROCm Silent Fallback Detection (69a9210)

Enhanced DeviceManager with ROCm-specific validation:

- `_validate_rocm()` runs after ROCm detection: logs GPU memory, runs trivial matmul sanity check (<100ms threshold), then full performance validation
- `validate_gpu_performance()` compares 256x256 matmul on GPU vs CPU - logs CRITICAL warning if GPU is slower (indicates silent CPU fallback)
- `rocm_version` property exposes `torch.version.hip` for downstream consumers
- All ROCm-specific code wrapped in try/except with CPU fallback

Key code in `backend/app/core/device_manager.py`:
```python
def validate_gpu_performance(self) -> bool:
    # GPU vs CPU 256x256 matmul comparison
    # Returns False and logs CRITICAL if GPU slower than CPU
```

### Task 2: ROCm Setup Documentation (e0f0f91)

Created `docs/ROCM_SETUP.md` covering:
- Linux ROCm 6.4 (production): supported GPUs, installation, verification, troubleshooting
- Windows ROCm (experimental): preview status, supported GPUs, auto-fallback note
- NVIDIA CUDA setup (brief, for completeness)
- CPU-only setup with ONNX INT8 optimization note

Updated `backend/requirements.txt` comments with ROCm 6.2 install instructions and link to docs.

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Commit | Type | Description |
|--------|------|-------------|
| 69a9210 | feat | ROCm silent fallback detection in DeviceManager |
| e0f0f91 | docs | ROCm setup documentation and requirements update |

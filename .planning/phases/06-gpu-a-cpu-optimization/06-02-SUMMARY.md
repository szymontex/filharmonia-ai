# Phase 6 Plan 2: torch.compile GPU Acceleration Summary

**One-liner:** torch.compile applied conditionally on GPU with eager-vs-compiled benchmarking and graceful fallback on failure.

## What Was Done

### Task 1: Add torch.compile with warmup and benchmark logging
- Added `_apply_torch_compile()` method to ASTInferenceService, called at end of `load_model()`
- Checks `self.device_manager.supports_compile` to skip compilation on CPU
- On GPU: benchmarks 3 eager inference calls, applies `torch.compile(mode="default")`, benchmarks 3 compiled calls, logs speedup ratio
- Enables `torch._logging.set_logs(graph_breaks=True)` for debugging
- Entire block wrapped in try/except -- compilation failures log warning and continue with eager mode
- No changes to predict_segment or predict_batch
- **Commit:** f7236b9

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 3-call average for benchmarking | Reduces variance from JIT warmup while keeping startup fast |
| `mode="default"` for torch.compile | Balanced optimization; `max-autotune` too slow for startup warmup |
| Graph break logging via torch._logging | Helps debug compilation issues without external tools |

## Key Files

| File | Change |
|------|--------|
| `backend/app/services/ast_inference.py` | Added `_apply_torch_compile()` with warmup and benchmarking |

## Duration

~2 minutes

## Next Phase Readiness

No blockers. torch.compile is GPU-only; CPU optimization (ONNX INT8) is handled by 06-03.

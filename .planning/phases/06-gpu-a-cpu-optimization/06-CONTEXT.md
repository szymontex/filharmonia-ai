# Phase 6: GPU & CPU Optimization - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Make inference run optimally across different hardware configurations (NVIDIA CUDA GPU, AMD ROCm GPU, CPU) without requiring users to manually configure anything. Includes backend optimization (torch.compile, ONNX, INT8 quantization) and frontend modernization (React 19 upgrade, wavesurfer.js evaluation, confidence threshold auto-tuning).

</domain>

<decisions>
## Implementation Decisions

### Device detection & fallback
- Auto-detect only — no manual user override or configuration
- Once at startup — detect device when server starts, use same device for entire session
- Fallback with warning — if GPU fails (driver issue, OOM), log error, show user warning, automatically retry on CPU
- Device priority order: Claude's discretion (will determine based on typical performance characteristics)

### Optimization strategies
- torch.compile usage: Claude's discretion (will choose based on compatibility and performance tradeoffs)
- CPU ONNX + INT8: Claude's discretion (will choose based on model characteristics and typical workloads)
- Model format strategy: Claude's discretion (will choose based on disk space, startup time, and flexibility tradeoffs)
- Performance validation: Claude's discretion (will choose appropriate validation strategy - benchmarks, logging, or both)

### ROCm support scope
- Linux vs Windows: Claude's discretion (user cannot test, so defer to practical assessment of Windows ROCm maturity)
- ROCm failure handling: Claude's discretion (will choose based on typical failure modes and user experience)
- Installation documentation: Claude's discretion (will choose appropriate documentation level)
- Performance parity: Claude's discretion (will choose based on practical tradeoffs and testing capabilities)

### Frontend modernization
- React 19 timing: Claude's discretion (will choose based on dependency analysis and risk management)
- wavesurfer.js migration: Claude's discretion (evaluate current implementation, decide if migration is worthwhile)
- Auto-tuning behavior: Per-recording learning (adjust threshold based on user edits for each recording)
- Auto-tuning notification: Claude's discretion (will choose based on UX principles and user control preferences)

### Claude's Discretion
- Device priority order when multiple GPUs available
- torch.compile application strategy (always, conditional, or optional)
- ONNX + INT8 usage pattern (default, fallback, or conditional)
- Model format caching strategy
- Performance validation approach
- ROCm platform support (Linux-only vs Linux + Windows preview)
- ROCm failure handling
- ROCm documentation level
- ROCm optimization effort (parity vs working vs best-effort)
- React 19 upgrade timing relative to GPU work
- wavesurfer.js migration decision
- Auto-tuning notification strategy

</decisions>

<specifics>
## Specific Ideas

- User mentioned they cannot test ROCm configurations ("decyduj, nawet nie mam jak tego przetestowac wiec ...") — defer ROCm decisions to practical testing capabilities
- Per-recording learning for confidence threshold: auto-adjust based on user corrections within each recording (not global across all recordings)
- Auto-detect at startup, no manual device selection: users shouldn't need to configure hardware choices
- Fallback with warning: transparency when GPU fails and CPU is used

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-gpu-a-cpu-optimization*
*Context gathered: 2026-01-29*

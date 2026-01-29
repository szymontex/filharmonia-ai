# Phase 6: GPU & CPU Optimization - Research

**Researched:** 2026-01-29
**Domain:** PyTorch inference optimization (GPU acceleration, CPU optimization, hardware detection)
**Confidence:** MEDIUM-HIGH

## Summary

This phase optimizes PyTorch AST model inference across multiple hardware configurations (NVIDIA CUDA, AMD ROCm, CPU) with automatic device detection and hardware-specific optimizations. The research covers three main optimization strategies: (1) `torch.compile` for GPU acceleration, (2) ONNX Runtime with INT8 quantization for CPU inference, and (3) unified device detection that handles both NVIDIA and AMD GPUs transparently.

PyTorch 2.5.1 (current version in codebase) supports torch.compile as a mature JIT compiler with 20-40% GPU speedups and minimal code changes. ROCm 6.4+ provides production-ready PyTorch support for AMD Radeon RX 7000/9000 series GPUs on both Linux and Windows, using the same CUDA API surface for compatibility. ONNX Runtime with INT8 quantization provides 3-5x CPU speedup, though quantization requires careful calibration to avoid accuracy degradation.

Frontend modernization (React 18.3 → 19) requires attention to breaking changes but has straightforward migration paths via codemods. Confidence threshold auto-tuning should follow per-recording learning patterns using user correction feedback loops.

**Primary recommendation:** Implement automatic device detection at startup with GPU priority (CUDA/ROCm > CPU), use torch.compile conditionally (GPU-only, skip on CPU), export separate ONNX INT8 model for CPU-only inference path, and handle silent ROCm fallback detection to prevent performance regressions.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.5.1+ | Core inference framework | Already in use, torch.compile available in 2.0+, mature ROCm support |
| torch.compile | Built-in (2.0+) | JIT compilation for GPU | Official PyTorch optimization, 20-40% speedup, minimal code changes |
| ONNX Runtime | 1.18+ | CPU-optimized inference | Industry standard for production CPU inference, INT8 quantization support |
| Optimum | 1.23+ | ONNX export for transformers | HuggingFace official exporter, handles AST models, includes optimization levels |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| onnx | 1.16+ | ONNX model format | Required for ONNX export, opset 14+ for transformers |
| optimum[onnxruntime] | 1.23+ | ONNX Runtime integration | Simplified ORTModel API, automatic optimization |
| React | 19.0+ | Frontend framework | Upgrade from 18.3, performance improvements, required for future ecosystem |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torch.compile | TorchScript | TorchScript is deprecated, torch.compile handles dynamic control flow better |
| ONNX Runtime | TensorRT | TensorRT is NVIDIA-only, ONNX Runtime works across vendors |
| Optimum | Manual torch.onnx.export | Optimum handles transformer-specific patterns, includes validation |
| Per-recording learning | Global threshold | Global threshold ignores recording-specific characteristics |

**Installation:**
```bash
# Backend (Python)
pip install optimum[onnxruntime]==1.23.0 onnx>=1.16.0

# Frontend (Node.js)
npm install react@^19.0.0 react-dom@^19.0.0
npm install --save-dev @types/react@^19.0.0 @types/react-dom@^19.0.0
```

## Architecture Patterns

### Recommended Project Structure
```
backend/app/
├── core/
│   └── device_manager.py      # Centralized device detection, fallback handling
├── services/
│   ├── ast_inference.py       # GPU inference (torch.compile wrapper)
│   ├── ast_inference_onnx.py  # CPU inference (ONNX Runtime)
│   └── inference_factory.py   # Route to GPU or CPU implementation
└── utils/
    └── benchmark.py           # Performance validation utilities

frontend/src/
├── utils/
│   └── confidenceThreshold.ts # Per-recording threshold learning
└── hooks/
    └── useConfidenceAdjust.ts # Track user corrections, adjust thresholds
```

### Pattern 1: Unified Device Detection at Startup
**What:** Detect hardware capabilities once at application startup, log details, establish device priority
**When to use:** Server initialization, before model loading
**Example:**
```python
# Source: PyTorch HIP (ROCm) documentation
# https://docs.pytorch.org/docs/stable/notes/hip.html

import torch
import logging

class DeviceManager:
    def __init__(self):
        self.device = None
        self.device_type = None  # "cuda_nvidia", "cuda_amd", or "cpu"
        self.device_name = None

    def detect_device(self):
        """Detect device once at startup, log details"""
        if torch.cuda.is_available():
            if torch.version.hip:
                # AMD ROCm
                self.device = torch.device('cuda')
                self.device_type = "cuda_amd"
                self.device_name = torch.cuda.get_device_name(0)
                logging.info(f"[GPU] AMD ROCm detected: {self.device_name}")
            elif torch.version.cuda:
                # NVIDIA CUDA
                self.device = torch.device('cuda')
                self.device_type = "cuda_nvidia"
                self.device_name = torch.cuda.get_device_name(0)
                logging.info(f"[GPU] NVIDIA CUDA detected: {self.device_name}")
        else:
            # CPU fallback
            self.device = torch.device('cpu')
            self.device_type = "cpu"
            logging.warning("[CPU] No GPU detected, using CPU inference")

        return self.device, self.device_type

# Singleton pattern
_device_manager = None
def get_device_manager():
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
        _device_manager.detect_device()
    return _device_manager
```

### Pattern 2: Conditional torch.compile (GPU-only)
**What:** Apply torch.compile only on GPU, skip on CPU to avoid compilation overhead
**When to use:** Model initialization after device detection
**Example:**
```python
# Source: PyTorch torch.compile tutorial
# https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

class ASTInferenceService:
    def __init__(self, device_manager):
        self.device_manager = device_manager
        self.model = None

    def load_model(self, model_path):
        # Load model architecture
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=5,
            ignore_mismatched_sizes=True
        )

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device_manager.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Move to device
        self.model = self.model.to(self.device_manager.device)
        self.model.eval()

        # Apply torch.compile ONLY on GPU
        if self.device_manager.device_type in ["cuda_nvidia", "cuda_amd"]:
            logging.info("[OPTIMIZE] Applying torch.compile for GPU acceleration")
            self.model = torch.compile(self.model, mode="default")
            # Warmup: trigger compilation
            self._warmup_inference()
        else:
            logging.info("[OPTIMIZE] Skipping torch.compile on CPU")

    def _warmup_inference(self):
        """Run warmup inference to trigger torch.compile ahead of time"""
        # Create dummy input matching production shape
        dummy_input = torch.randn(1, 1024, 128).to(self.device_manager.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        logging.info("[OPTIMIZE] torch.compile warmup complete")
```

### Pattern 3: ONNX Export with Optimum for CPU Inference
**What:** Export PyTorch model to ONNX with INT8 quantization for fast CPU inference
**When to use:** One-time export script, creates optimized model for CPU-only deployments
**Example:**
```python
# Source: HuggingFace Optimum documentation
# https://huggingface.co/docs/optimum-onnx/en/onnx/usage_guides/export_a_model

from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import AutoConfig

def export_ast_to_onnx_int8(model_path, output_dir):
    """Export AST model to ONNX with INT8 quantization"""
    # Export with O2 optimization (transformers-specific fusions)
    # Note: O4 (fp16) requires GPU, use O2 for CPU
    main_export(
        model_path,
        output=output_dir,
        task="audio-classification",
        optimize="O2",  # Basic + transformers fusions
        device="cpu",
        opset=14  # Min opset for transformers
    )

    # Load and quantize
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_model_path = output_dir / "model.onnx"
    quantized_model_path = output_dir / "model_int8.onnx"

    quantize_dynamic(
        model_input=str(onnx_model_path),
        model_output=str(quantized_model_path),
        weight_type=QuantType.QInt8,  # INT8 weights
        optimize_model=True,
        per_channel=True,  # Better accuracy for transformers
    )

    logging.info(f"[EXPORT] INT8 ONNX model saved to {quantized_model_path}")
```

### Pattern 4: Inference Factory (Route to GPU or CPU Backend)
**What:** Factory pattern that routes inference requests to GPU (torch.compile) or CPU (ONNX) backend
**When to use:** Startup initialization, single entry point for inference
**Example:**
```python
# Application startup pattern
class InferenceFactory:
    @staticmethod
    def create_inference_service():
        device_manager = get_device_manager()

        if device_manager.device_type in ["cuda_nvidia", "cuda_amd"]:
            # GPU path: PyTorch with torch.compile
            from app.services.ast_inference import ASTInferenceService
            service = ASTInferenceService(device_manager)
            service.load_model(settings.AST_MODEL_PATH)
            return service
        else:
            # CPU path: ONNX Runtime INT8
            from app.services.ast_inference_onnx import ASTInferenceONNXService
            service = ASTInferenceONNXService()
            service.load_model(settings.AST_ONNX_INT8_PATH)
            return service
```

### Pattern 5: Per-Recording Threshold Learning with User Feedback
**What:** Track user corrections per recording, adjust confidence threshold based on false positive/negative patterns
**When to use:** Frontend track editing, when user accepts/rejects predictions
**Example:**
```typescript
// Source: Online learning feedback loop patterns
// https://engineering.fb.com/2026/01/14/ml-applications/adapting-the-facebook-reels-recsys-ai-model-based-on-user-feedback/

interface RecordingThreshold {
  recordingId: string;
  threshold: number;
  falsePositives: number;  // Predictions user deleted
  falseNegatives: number;  // User added labels ML missed
  corrections: number;      // Total user corrections
}

class ConfidenceThresholdLearner {
  private thresholds = new Map<string, RecordingThreshold>();
  private readonly DEFAULT_THRESHOLD = 0.7;
  private readonly LEARNING_RATE = 0.05;

  getThreshold(recordingId: string): number {
    const record = this.thresholds.get(recordingId);
    return record?.threshold ?? this.DEFAULT_THRESHOLD;
  }

  recordUserCorrection(
    recordingId: string,
    correctionType: 'delete' | 'add',
    confidence?: number
  ) {
    let record = this.thresholds.get(recordingId);

    if (!record) {
      record = {
        recordingId,
        threshold: this.DEFAULT_THRESHOLD,
        falsePositives: 0,
        falseNegatives: 0,
        corrections: 0
      };
    }

    record.corrections++;

    if (correctionType === 'delete') {
      // User deleted prediction: threshold too low
      record.falsePositives++;
      record.threshold = Math.min(0.95, record.threshold + this.LEARNING_RATE);
    } else {
      // User added label: threshold too high
      record.falseNegatives++;
      record.threshold = Math.max(0.5, record.threshold - this.LEARNING_RATE);
    }

    this.thresholds.set(recordingId, record);

    // Persist to localStorage for session continuity
    this.persist();
  }

  private persist() {
    const data = Array.from(this.thresholds.entries());
    localStorage.setItem('recordingThresholds', JSON.stringify(data));
  }
}
```

### Anti-Patterns to Avoid
- **Manual device selection by users:** Auto-detection is more reliable, users shouldn't configure GPU vs CPU
- **Silent ROCm CPU fallback:** ROCm can silently fall back to CPU for unsupported operations, causing 100x slowdown without warning
- **Global confidence threshold:** Different recordings have different characteristics, per-recording learning is essential
- **Applying torch.compile on CPU:** Compilation overhead exceeds runtime savings on CPU, use ONNX instead
- **Using TorchScript instead of torch.compile:** TorchScript is deprecated and handles dynamic control flow poorly
- **Forgetting warmup for torch.compile:** First inference is slow (5-60s compilation), warmup at startup prevents user-facing latency

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ONNX export for transformers | Custom torch.onnx.export | Optimum CLI/API | Handles AST-specific patterns, includes validation, supports optimization levels |
| INT8 quantization | Manual quantization logic | onnxruntime.quantization | Handles calibration, multiple quantization schemes (dynamic/static), per-channel support |
| Device detection | Parse CUDA version strings | torch.version.hip / torch.version.cuda | Official API, handles ROCm's CUDA compatibility layer correctly |
| Inference benchmarking | Manual timeit loops | torch.utils.benchmark.Timer | Handles GPU synchronization, warmup, statistical analysis automatically |
| React 19 migration | Manual code changes | Official codemods (react/19/migration-recipe) | Automated transforms for 90% of breaking changes, tested by React team |
| Graph break detection | Trial and error | torch._logging.set_logs(graph_breaks=True) | Shows all graph breaks with stack traces, fullgraph=True mode for strict enforcement |

**Key insight:** PyTorch and ONNX ecosystems have mature, well-tested tools for optimization. Custom solutions miss edge cases (e.g., ROCm's silent CPU fallback, ONNX quantization saturation issues, graph break subtleties) and require ongoing maintenance as PyTorch evolves.

## Common Pitfalls

### Pitfall 1: Silent ROCm CPU Fallback
**What goes wrong:** When ROCm lacks a kernel for an operation, PyTorch silently falls back to CPU execution, causing 10-100x slowdown with no warning or error
**Why it happens:** ROCm's operator coverage is nearly complete but edge cases remain (e.g., 3D convolutions, specific sparse ops). PyTorch prioritizes correctness over performance warnings
**How to avoid:**
- Log device placement for critical operations: `logging.debug(f"Tensor device: {tensor.device}")`
- Monitor inference latency, alert on unexpected slowdown (e.g., >2x median)
- Use benchmark script to validate GPU utilization (`nvidia-smi` / `rocm-smi`)
**Warning signs:** Inference suddenly 10x slower on ROCm vs CUDA with same model, GPU utilization drops to near zero during inference

### Pitfall 2: torch.compile Graph Breaks Destroying Performance
**What goes wrong:** Data-dependent control flow causes graph breaks, fragmenting the computation graph and losing optimization opportunities
**Why it happens:** torch.compile traces through Python code but cannot handle tensor-value conditionals (`if tensor.sum() > 0:`) or `.item()` calls
**How to avoid:**
- Use `torch.compile(..., fullgraph=True)` during development to catch breaks early
- Enable graph break logging: `torch._logging.set_logs(graph_breaks=True)`
- Replace Python conditionals with tensor ops: `torch.where()`, boolean indexing, `torch.cond()`
- Avoid `.item()`, `.data_ptr()`, `.numpy()` in forward pass
**Warning signs:** Model runs correctly but shows no speedup from torch.compile, logs show multiple graph breaks

### Pitfall 3: ONNX INT8 Quantization Accuracy Degradation
**What goes wrong:** Quantized model loses 5-10% accuracy, making predictions unreliable
**Why it happens:** Saturation on x86-64 CPUs (VPMADDUBSW instruction), poor calibration data, sensitive numerical computations
**How to avoid:**
- Use S8S8 format (signed int8, default) for best accuracy/performance balance
- Try U8U8 format if saturation issues occur (accuracy drops >5%)
- Use per-channel quantization for better accuracy: `per_channel=True`
- Validate on representative data: compare quantized vs original model outputs
- If accuracy loss is unacceptable, keep FP32 ONNX model (still faster than PyTorch eager)
**Warning signs:** Validation accuracy drops >3% after quantization, specific classes perform much worse

### Pitfall 4: React 19 TypeScript Breaking Changes
**What goes wrong:** `useRef()` requires an argument, `ReactElement.props` defaults to `unknown`, JSX namespace changes break builds
**Why it happens:** React 19 improved type safety, removed implicit `any` types, standardized ref handling
**How to avoid:**
- Run TypeScript codemod BEFORE upgrading: `npx types-react-codemod@latest preset-19 ./src`
- Update all `useRef()` calls to include initial value: `useRef(null)` or `useRef(undefined)`
- Test with React 18.3 first (includes deprecation warnings)
- Review all ref callback functions for implicit returns: `ref={current => instance = current}` → `ref={current => {instance = current}}`
**Warning signs:** TypeScript errors about missing arguments, `unknown` type errors on props access, JSX namespace errors

### Pitfall 5: torch.compile Compilation Overhead on First Inference
**What goes wrong:** First API request takes 5-60 seconds while torch.compile compiles the model, causing timeout or poor UX
**Why it happens:** torch.compile is a JIT compiler that compiles on first execution, not at load time
**How to avoid:**
- Run warmup inference immediately after model loading: `_warmup_inference()`
- Use dummy input matching production shape: `torch.randn(1, 1024, 128)`
- For faster startup, consider regional compilation (compiles smaller blocks)
- Cache compiled models if possible (reduces cold start on server restart)
**Warning signs:** First request times out, subsequent requests are fast, logs show "TorchDynamo" messages on first inference

### Pitfall 6: Wrong Device Priority Order Causing Suboptimal Performance
**What goes wrong:** CPU selected when AMD GPU is available, or CUDA fails but doesn't fall back to CPU
**Why it happens:** Device detection logic doesn't handle all cases, priority order not defined
**How to avoid:**
- Use priority order: CUDA (NVIDIA or AMD) > CPU
- Implement fallback with user notification: log error if GPU fails, retry on CPU
- Don't distinguish NVIDIA vs AMD in priority (both use CUDA API, similar performance)
- Test startup behavior with GPU disabled to verify CPU fallback works
**Warning signs:** GPU available but CPU inference used, startup fails when GPU has issue

## Code Examples

Verified patterns from official sources:

### Device Detection with ROCm/CUDA Distinction
```python
# Source: PyTorch HIP (ROCm) semantics documentation
# https://docs.pytorch.org/docs/stable/notes/hip.html

import torch
import logging

def detect_and_log_device():
    """Detect device and log detailed information"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)

        if torch.version.hip:
            device_type = "ROCm (AMD)"
            version_info = torch.version.hip
        elif torch.version.cuda:
            device_type = "CUDA (NVIDIA)"
            version_info = torch.version.cuda
        else:
            device_type = "Unknown GPU"
            version_info = "N/A"

        logging.info(f"[GPU] {device_type} detected")
        logging.info(f"  Device: {device_name}")
        logging.info(f"  Version: {version_info}")
        logging.info(f"  Device count: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        logging.warning("[CPU] No GPU detected, using CPU inference")

    return device
```

### Inference Benchmarking with Proper GPU Synchronization
```python
# Source: PyTorch Performance Tuning Guide
# https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

import torch
import time

def benchmark_inference(model, input_tensor, device, num_warmup=50, num_iterations=100):
    """Benchmark inference with proper warmup and GPU synchronization"""
    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model(input_tensor)
            # Synchronize after each iteration for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    throughput = 1.0 / avg_time

    return {
        'avg_latency_ms': avg_time * 1000,
        'throughput_per_sec': throughput,
        'total_iterations': num_iterations
    }
```

### ONNX Dynamic Quantization for CPU
```python
# Source: ONNX Runtime Quantization documentation
# https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

def quantize_ast_model(model_path: Path, output_path: Path):
    """Apply dynamic INT8 quantization to ONNX model"""
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,  # INT8 for weights
        optimize_model=True,          # Apply graph optimizations
        per_channel=True,             # Per-channel quantization (better accuracy)
        reduce_range=False,           # Set True if saturation issues occur
        extra_options={
            'ActivationSymmetric': False,  # Asymmetric for CPU (better accuracy)
            'WeightSymmetric': True,       # Symmetric weights
        }
    )

    logging.info(f"[QUANTIZE] INT8 model saved to {output_path}")

    # Validate model
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    logging.info("[QUANTIZE] Model validation passed")
```

### React 19 Migration Pattern for useRef
```typescript
// Source: React 19 Upgrade Guide
// https://react.dev/blog/2024/04/25/react-19-upgrade-guide

// Before (React 18)
function Component() {
  const ref = useRef();  // No argument - BREAKS in React 19
  return <div ref={ref} />;
}

// After (React 19)
function Component() {
  const ref = useRef<HTMLDivElement>(null);  // Explicit null argument
  return <div ref={ref} />;
}

// Mutable refs (all refs are mutable in React 19)
function Counter() {
  const count = useRef<number>(0);  // TypeScript generic

  const increment = () => {
    count.current += 1;  // Now works - all refs are mutable
  };

  return <button onClick={increment}>Count: {count.current}</button>;
}
```

### Logging Graph Breaks in torch.compile
```python
# Source: PyTorch torch.compile troubleshooting
# https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html

import torch

# Enable graph break logging
torch._logging.set_logs(graph_breaks=True)

# Compile with fullgraph=True to enforce no graph breaks (raises error if breaks occur)
model = torch.compile(model, fullgraph=True)

# Or use default mode and log breaks
model = torch.compile(model, mode="default")

# Graph breaks will be logged to stderr with stack traces
# Example output:
# [WARNING] Graph break: calling method __len__ on <class 'list'>
# Stack trace shows where the break occurred
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TorchScript (torch.jit) | torch.compile | PyTorch 2.0 (2023) | Better dynamic control flow handling, 20-40% faster compilation, easier debugging |
| Manual torch.onnx.export | Optimum library | Optimum 1.0 (2022) | Transformers-specific patterns, automatic validation, optimization levels |
| String refs in React | useRef hook | React 16.3 (2018), removed in 19 (2024) | Type-safe, functional components, no this context issues |
| ReactDOM.render | ReactDOM.createRoot | React 18 (2022), required in 19 (2024) | Concurrent features, automatic batching, Suspense support |
| TF32 disabled by default | TF32 enabled by default | PyTorch 1.12 (2022) | 3x faster matmuls on Ampere+ GPUs with minimal accuracy loss (NVIDIA only, not ROCm) |
| U8S8 quantization default | S8S8 quantization default | ONNX Runtime 1.10 (2021) | Avoids saturation issues on AVX2/AVX512 CPUs, better accuracy |

**Deprecated/outdated:**
- **TorchScript (torch.jit.trace, torch.jit.script):** Officially deprecated in favor of torch.compile and torch.export
- **react-test-renderer/shallow:** Removed in React 19, use @testing-library/react instead
- **PropTypes and defaultProps:** Deprecated in React 18, removed in 19, use TypeScript and ES6 defaults
- **ROCm 5.x:** ROCm 6.4+ required for PyTorch 2.5+, earlier versions have limited transformer support
- **ONNX opset <14:** Transformers require opset 14+ for full compatibility

## Open Questions

Things that couldn't be fully resolved:

1. **ROCm Windows Preview Stability**
   - What we know: ROCm 6.4.4 adds Windows support for Radeon RX 7000/9000 series (January 2026)
   - What's unclear: Production readiness, stability vs Linux, whether "preview" means beta quality
   - Recommendation: Mark Windows ROCm as "experimental" in docs, prioritize Linux ROCm support, graceful degradation to CPU on Windows if ROCm fails

2. **torch.compile Performance on AMD ROCm vs NVIDIA CUDA**
   - What we know: torch.compile works on ROCm, AMD has specific blog post about it
   - What's unclear: Performance parity (do both achieve similar 20-40% speedups?), specific optimizations needed
   - Recommendation: Benchmark both platforms if available, document actual speedups, don't assume NVIDIA numbers apply to AMD

3. **AST Model ONNX Export Compatibility**
   - What we know: Optimum supports audio-classification task, AST is a HuggingFace transformers model
   - What's unclear: Whether MIT AST pretrained model exports cleanly, if custom fine-tuning affects export
   - Recommendation: Test ONNX export early in implementation, validate output matches PyTorch, keep PyTorch fallback if export fails

4. **React 19 Dependency Compatibility**
   - What we know: React 19 released December 2024, ecosystem catching up
   - What's unclear: Whether @tanstack/react-query, zustand, react-router-dom have React 19-compatible versions
   - Recommendation: Check each dependency's React 19 support before upgrading, consider staged upgrade (React 18.3 → 19 → dependencies)

5. **wavesurfer.js vs Current Implementation**
   - What we know: Current implementation uses waveform-data library, wavesurfer.js is popular alternative
   - What's unclear: Whether current implementation has limitations, if migration provides meaningful benefits
   - Recommendation: Evaluate current waveform rendering performance, only migrate if clear benefits (features, performance, maintenance)

## Sources

### Primary (HIGH confidence)
- PyTorch official documentation - torch.compile tutorial: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- PyTorch official documentation - HIP (ROCm) semantics: https://docs.pytorch.org/docs/stable/notes/hip.html
- ONNX Runtime official documentation - Quantization guide: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- HuggingFace Optimum documentation - Export models to ONNX: https://huggingface.co/docs/optimum-onnx/en/onnx/usage_guides/export_a_model
- React official documentation - React 19 Upgrade Guide: https://react.dev/blog/2024/04/25/react-19-upgrade-guide

### Secondary (MEDIUM confidence)
- PyTorch blog - torch.compile and Diffusers performance: https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
- AMD ROCm blog - torch.compile on AMD GPUs: https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html
- ROCm documentation - PyTorch compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html
- ONNX Runtime documentation - oneDNN execution provider: https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html
- Meta Engineering blog - Adapting RecSys based on user feedback: https://engineering.fb.com/2026/01/14/ml-applications/adapting-the-facebook-reels-recsys-ai-model-based-on-user-feedback/

### Tertiary (LOW confidence - flagged for validation)
- GitHub issue - Silent ROCm CPU fallback in Qwen2.5-VL: https://github.com/pytorch/pytorch/issues/169857
- Medium article - 8 TorchCompile Pitfalls: https://medium.com/@Modexa/8-torchcompile-pitfalls-and-how-to-dodge-them-3364cd7352ce
- Medium article - PyTorch to Quantized ONNX Model: https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27
- WCCFTech article - ROCm 6.4.4 Windows support announcement: https://wccftech.com/amd-rocm-6-4-4-pytorch-support-windows-radeon-9000-radeon-7000-gpus-ryzen-ai-apus/

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official PyTorch 2.5.1 documentation, ONNX Runtime 1.18+ docs, Optimum 1.23+ docs
- Architecture: MEDIUM-HIGH - Patterns verified in official docs, device detection confirmed, ONNX export tested in HuggingFace
- Pitfalls: MEDIUM - Silent ROCm fallback from GitHub issues (LOW), graph breaks from official docs (HIGH), quantization issues from official docs (HIGH)
- React 19: HIGH - Official React upgrade guide, codemods tested, TypeScript migrations documented
- ROCm Windows: LOW - Based on news articles and release notes, not production testing

**Research date:** 2026-01-29
**Valid until:** 2026-03-01 (30 days - PyTorch ecosystem is stable but ROCm support evolving rapidly)

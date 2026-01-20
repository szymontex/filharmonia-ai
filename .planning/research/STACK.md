# Technology Stack: Cross-Platform PyTorch Deployment

**Project:** Filharmonia AI - Audio Analysis
**Researched:** 2026-01-20
**Focus:** Production-ready multi-backend inference (CUDA/ROCm/CPU)

## Executive Summary

The 2025/2026 PyTorch ecosystem has matured significantly for cross-platform deployment. The recommended approach for Filharmonia AI is a **tiered strategy**:

1. **Primary:** Native PyTorch with `torch.compile` for CUDA and ROCm
2. **CPU Optimization:** ONNX Runtime with quantized models for fast CPU inference
3. **Fallback:** Native PyTorch eager mode for maximum compatibility

This approach provides 3-5x speedups on GPU via `torch.compile`, 20-30% speedups on CPU via ONNX Runtime, while maintaining a single codebase.

---

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | 2.9.x | Model training & primary inference | Latest stable with torch.compile maturity, wheel variants for easy multi-platform install |
| torchaudio | 2.9.x | Audio preprocessing | Native GPU-accelerated mel-spectrogram, consistent with training |
| transformers | 4.50+ | AST model architecture | HuggingFace AST with latest optimizations |

**Rationale:** PyTorch 2.9 introduces wheel variants that simplify installation across CUDA/ROCm/CPU. The `torch.compile` feature is now production-ready for both NVIDIA and AMD GPUs.

### GPU Backends

| Backend | Version | Target Hardware | Installation |
|---------|---------|-----------------|--------------|
| CUDA | 12.6/12.8/13.0 | NVIDIA RTX/Tesla/Quadro | `--index-url https://download.pytorch.org/whl/cu126` |
| ROCm | 6.4 (stable) / 7.1 (nightly) | AMD Radeon RX 7000/9000 | `--index-url https://download.pytorch.org/whl/rocm6.4` |

**CUDA Support (HIGH confidence - Official PyTorch docs):**
- CUDA 12.6: Mature, recommended for RTX 3000/4000 series
- CUDA 12.8: Recommended for RTX 5000 series (Blackwell)
- CUDA 13.0: Cutting edge, latest features

**ROCm Support (MEDIUM confidence - AMD docs + testing reports):**
- ROCm 6.4: Stable release, AMD-tested wheels
- ROCm 7.1: Nightly, broader GPU architecture support
- Supported architectures: gfx1100 (RX 7900), gfx1200 (RX 9000 series), MI300X

### CPU Optimization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| ONNX Runtime | 1.20+ | Production CPU inference | 20-30% faster than eager PyTorch, quantization support |
| optimum-onnx | 2.0+ | Model export & optimization | Easy export from HuggingFace models |

**CPU Optimization Rationale:**
- Intel Extension for PyTorch was **discontinued** after version 2.8 (upstreamed to PyTorch core)
- ONNX Runtime provides the best cross-platform CPU performance with quantization
- OpenVINO is an alternative for Intel-only deployments but less portable

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| librosa | 0.10.x | Audio analysis utilities | Feature extraction, audio loading |
| soundfile | 0.12+ | Audio I/O | Loading WAV files efficiently |
| numpy | 1.26+ | Array operations | CPU preprocessing |
| accelerate | 1.10+ | Device management | Automatic device placement |

---

## Multi-Backend Architecture

### Device Detection Pattern (Recommended)

```python
import torch
from enum import Enum
from dataclasses import dataclass

class AcceleratorType(Enum):
    CUDA = "cuda"      # NVIDIA GPU
    ROCM = "rocm"      # AMD GPU (uses cuda API)
    CPU = "cpu"        # CPU fallback

@dataclass
class DeviceInfo:
    accelerator: AcceleratorType
    device: torch.device
    device_name: str
    supports_compile: bool

def detect_best_device() -> DeviceInfo:
    """
    Detect best available compute device.
    ROCm uses torch.cuda API - both NVIDIA and AMD appear as 'cuda'.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)

        # Distinguish NVIDIA from AMD by device name
        if 'AMD' in device_name or 'Radeon' in device_name:
            accelerator = AcceleratorType.ROCM
            # torch.compile works on ROCm but less mature
            supports_compile = True
        else:
            accelerator = AcceleratorType.CUDA
            supports_compile = True

        return DeviceInfo(
            accelerator=accelerator,
            device=device,
            device_name=device_name,
            supports_compile=supports_compile
        )

    return DeviceInfo(
        accelerator=AcceleratorType.CPU,
        device=torch.device('cpu'),
        device_name="CPU",
        supports_compile=True  # torch.compile works on CPU too
    )
```

**Key Insight:** PyTorch's ROCm implementation uses the `torch.cuda` API. Both NVIDIA and AMD GPUs appear under `torch.cuda.is_available()`. Distinguish by checking the device name.

### Inference Service Architecture

```python
from abc import ABC, abstractmethod

class InferenceBackend(ABC):
    @abstractmethod
    def load_model(self, model_path: Path) -> None: ...

    @abstractmethod
    def predict(self, features: torch.Tensor) -> torch.Tensor: ...

class PyTorchBackend(InferenceBackend):
    """Native PyTorch with optional torch.compile"""

    def __init__(self, device_info: DeviceInfo, use_compile: bool = True):
        self.device = device_info.device
        self.use_compile = use_compile and device_info.supports_compile
        self.model = None

    def load_model(self, model_path: Path):
        from transformers import ASTForAudioClassification

        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=5,
            ignore_mismatched_sizes=True
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Apply torch.compile for GPU acceleration
        if self.use_compile and self.device.type == 'cuda':
            self.model = torch.compile(self.model, mode='reduce-overhead')

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(features.to(self.device)).logits

class ONNXBackend(InferenceBackend):
    """ONNX Runtime for optimized CPU inference"""

    def __init__(self, use_quantized: bool = True):
        import onnxruntime as ort

        # Select execution providers
        self.providers = ['CPUExecutionProvider']
        self.session = None
        self.use_quantized = use_quantized

    def load_model(self, model_path: Path):
        import onnxruntime as ort

        # Load quantized ONNX model for CPU
        onnx_path = model_path.with_suffix('.onnx')
        if self.use_quantized:
            onnx_path = model_path.with_name(
                model_path.stem + '_quantized.onnx'
            )

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=self.providers
        )

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        # ONNX Runtime expects numpy
        inputs = {
            self.session.get_inputs()[0].name: features.numpy()
        }
        outputs = self.session.run(None, inputs)
        return torch.from_numpy(outputs[0])
```

### Backend Selection Strategy

```python
def create_inference_backend(
    model_path: Path,
    force_backend: str = None
) -> InferenceBackend:
    """
    Create best inference backend for current hardware.

    Priority:
    1. GPU + torch.compile (CUDA or ROCm)
    2. ONNX Runtime quantized (CPU, if ONNX model exists)
    3. Native PyTorch eager (fallback)
    """
    device_info = detect_best_device()

    if force_backend:
        if force_backend == 'onnx':
            return ONNXBackend(use_quantized=True)
        elif force_backend == 'pytorch':
            return PyTorchBackend(device_info, use_compile=False)

    # GPU available: use PyTorch with torch.compile
    if device_info.accelerator in (AcceleratorType.CUDA, AcceleratorType.ROCM):
        return PyTorchBackend(device_info, use_compile=True)

    # CPU: check for ONNX model
    onnx_path = model_path.with_suffix('.onnx')
    if onnx_path.exists():
        return ONNXBackend(use_quantized=True)

    # Fallback: native PyTorch
    return PyTorchBackend(device_info, use_compile=False)
```

---

## ONNX Export for CPU Optimization

### Export AST Model to ONNX

```python
from optimum.onnxruntime import ORTModelForAudioClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

def export_ast_to_onnx(
    pytorch_model_path: Path,
    output_dir: Path,
    quantize: bool = True
):
    """
    Export fine-tuned AST model to ONNX with optional quantization.

    Quantized INT8 models are ~4x smaller and ~30% faster on CPU.
    """
    from transformers import ASTForAudioClassification
    import torch

    # Load PyTorch model
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save for Optimum export
    temp_dir = output_dir / "temp_hf_model"
    model.save_pretrained(temp_dir)

    # Export to ONNX using Optimum
    ort_model = ORTModelForAudioClassification.from_pretrained(
        temp_dir,
        export=True
    )
    ort_model.save_pretrained(output_dir)

    if quantize:
        # Apply dynamic INT8 quantization
        quantizer = ORTQuantizer.from_pretrained(output_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,  # Dynamic quantization
            per_channel=False
        )
        quantizer.quantize(
            save_dir=output_dir,
            quantization_config=qconfig
        )

    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)
```

### ONNX Runtime Execution Providers

| Provider | Platform | When to Use |
|----------|----------|-------------|
| `CPUExecutionProvider` | All | Default, always available |
| `CUDAExecutionProvider` | NVIDIA | GPU inference with ONNX |
| `ROCmExecutionProvider` | AMD (deprecated) | Use native PyTorch instead |
| `OpenVINOExecutionProvider` | Intel | Intel-optimized CPU inference |

**Recommendation:** For GPU inference, use native PyTorch with `torch.compile` rather than ONNX Runtime's GPU providers. ONNX Runtime's ROCm provider is deprecated. Use ONNX Runtime primarily for CPU optimization.

---

## Installation Commands

### Production Installation (requirements.txt approach)

```bash
# requirements-base.txt (platform-agnostic)
transformers>=4.50.0
torchaudio>=2.9.0
librosa>=0.10.2
soundfile>=0.12.1
numpy>=1.26.0
accelerate>=1.10.0
```

```bash
# Install for NVIDIA CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-base.txt

# Install for AMD ROCm 6.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
pip install -r requirements-base.txt

# Install for CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime>=1.20.0 optimum-onnx>=2.0.0
pip install -r requirements-base.txt
```

### Unified Installation (using torchruntime - experimental)

```bash
# torchruntime auto-detects hardware and installs correct backend
pip install torchruntime
pip install -r requirements-base.txt
```

**Note:** `torchruntime` is a third-party package that simplifies multi-platform installation but is less battle-tested than manual wheel selection.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| GPU Compilation | torch.compile | TensorRT | TensorRT requires NVIDIA-specific export, doesn't work on AMD |
| CPU Inference | ONNX Runtime | Intel Extension for PyTorch | IPEX discontinued after v2.8, features upstreamed to PyTorch |
| CPU Inference | ONNX Runtime | OpenVINO | Intel-only, less portable across AMD/Intel CPUs |
| Model Export | optimum-onnx | torch.onnx.export | optimum handles HuggingFace models better, includes quantization |
| Multi-backend | Native PyTorch | JAX/Flax | Would require rewriting model code, no HuggingFace AST support |

---

## Performance Expectations

Based on research (MEDIUM confidence - community benchmarks):

| Backend | Hardware | Relative Speed | Notes |
|---------|----------|----------------|-------|
| PyTorch eager | NVIDIA RTX 4090 | 1.0x (baseline) | Standard inference |
| torch.compile | NVIDIA RTX 4090 | 2.5-3.5x | First call has compile overhead |
| torch.compile | AMD RX 7900 XTX | 2.0-2.6x | Less mature but functional |
| PyTorch eager | CPU (Intel i9) | 0.15x | Very slow for transformers |
| ONNX Runtime | CPU (Intel i9) | 0.20x | ~30% faster than eager |
| ONNX Quantized | CPU (Intel i9) | 0.25-0.30x | INT8, slight accuracy loss |

**For 30-40 minute audio files (~700 segments):**
- GPU: 30-60 seconds total processing
- CPU (eager): 10-15 minutes
- CPU (ONNX quantized): 7-10 minutes

---

## Environment Variables

```bash
# Thread control (prevent CPU overload during inference)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ROCm-specific
export HIP_VISIBLE_DEVICES=0  # Limit to specific AMD GPU
export PYTORCH_ROCM_ARCH=gfx1100  # Target architecture

# CUDA-specific
export CUDA_VISIBLE_DEVICES=0  # Limit to specific NVIDIA GPU

# torch.compile cache (speeds up subsequent runs)
export TORCHINDUCTOR_CACHE_DIR=/path/to/cache
```

---

## Containerization

### Dockerfile Strategy

```dockerfile
# Base image selection based on target platform
ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Copy application
COPY . .

# Runtime configuration
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
```

**Build Commands:**
```bash
# For NVIDIA
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime -t filharmonia:cuda .

# For AMD ROCm
docker build --build-arg BASE_IMAGE=rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_2.8.0 -t filharmonia:rocm .

# For CPU
docker build --build-arg BASE_IMAGE=python:3.12-slim -t filharmonia:cpu .
```

---

## Sources

**HIGH Confidence (Official Documentation):**
- [PyTorch Get Started](https://pytorch.org/get-started/locally/) - Installation matrix, versions
- [ROCm PyTorch Compatibility](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html) - ROCm versions, GPU architectures
- [HuggingFace Optimum ONNX](https://huggingface.co/docs/optimum/en/onnxruntime/usage_guides/models) - ONNX export and inference

**MEDIUM Confidence (AMD/Intel Technical Articles):**
- [PyTorch 2.9 Wheel Variants](https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html) - ROCm installation
- [Intel Extension for PyTorch End of Life](https://intel.github.io/intel-extension-for-pytorch/) - Discontinuation notice
- [torch.compile on AMD](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html) - Performance benchmarks

**LOW Confidence (Community/Blogs - verify before relying on):**
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/) - Ecosystem overview
- [ONNX vs PyTorch Speed](https://dev-kit.io/blog/machine-learning/onnx-vs-pytorch-speed-comparison) - Performance comparisons

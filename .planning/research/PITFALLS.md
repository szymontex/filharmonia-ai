# Domain Pitfalls: PyTorch Cross-Platform GPU Deployment

**Domain:** ML audio analysis application with PyTorch/Transformers
**Target Hardware:** NVIDIA CUDA, AMD ROCm, CPU-only Linux server
**Researched:** 2026-01-20
**Confidence:** HIGH (verified via official documentation and current sources)

---

## Critical Pitfalls

Mistakes that cause rewrites, major instability, or fundamental architecture problems.

---

### Pitfall 1: Hardcoding Device Names Instead of Using Unified Detection

**What goes wrong:** Code uses `torch.device('cuda')` everywhere, assuming CUDA is always available. On AMD ROCm systems, developers mistakenly try `torch.device('rocm')` or `torch.device('hip')` - neither of which are valid PyTorch device strings. ROCm uses the `cuda` device name by design.

**Why it happens:** Developers assume each GPU backend needs its own device string. PyTorch for HIP/ROCm intentionally reuses the existing `torch.cuda` interfaces to minimize code changes.

**Consequences:**
- `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`
- Application crashes on AMD systems
- Code branches become unmaintainable

**Current code vulnerability:**
```python
# In ast_inference.py line 26:
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
This pattern works but doesn't provide visibility into WHAT GPU backend is active.

**Prevention:**
```python
def get_device():
    """Unified device detection with logging"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Log what backend is actually running (CUDA vs ROCm)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return device
    return torch.device("cpu")
```

**Detection (warning signs):**
- Code contains `if 'rocm' in...` or `if 'hip' in...` device checks
- Different code paths for AMD vs NVIDIA
- Device-related crashes only on certain machines

**Phase to address:** Phase 1 (Foundation) - establish unified device detection early.

**Sources:**
- [HIP (ROCm) semantics - PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [PyTorch ROCm Documentation](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)

---

### Pitfall 2: Saving Checkpoints with Device-Specific Tensors

**What goes wrong:** Models saved on GPU contain tensors tagged to `cuda:0`. When loaded on a CPU-only machine or different GPU, loading fails with "CUDA not available" or tensor device mismatch errors.

**Why it happens:** `torch.save()` preserves the device location of tensors. Developers forget to use `map_location` when loading, or save the entire model object instead of just the state_dict.

**Consequences:**
- `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`
- Models trained on GPU cannot run on CPU server
- Cross-machine deployment fails

**Current code vulnerability:**
```python
# In ast_inference.py line 52:
checkpoint = torch.load(model_path, map_location=self.device)
```
This is CORRECT - but many training scripts save incorrectly.

**Prevention:**
```python
# When SAVING (training code):
torch.save({
    'model_state_dict': model.state_dict(),  # NOT the model itself
    'epoch': epoch,
    'val_acc': val_acc,
}, checkpoint_path)

# When LOADING (inference code):
checkpoint = torch.load(model_path, map_location='cpu')  # Always load to CPU first
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)  # Then move to target device
```

**Detection:**
- `.pt` or `.pth` files that only work on specific machines
- Errors mentioning "CUDA device" when loading on CPU
- Checkpoint files unexpectedly large (may contain optimizer state)

**Phase to address:** Phase 1 (Foundation) - fix checkpoint format before adding new backends.

**Sources:**
- [torch.load Documentation](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [Tips for Loading from Checkpoint - PyTorch Tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)

---

### Pitfall 3: ROCm on Windows Is Preview-Only with Limited GPU Support

**What goes wrong:** Developers assume AMD GPU support on Windows is as mature as CUDA. In reality, ROCm on Windows is a preview release supporting only specific GPU generations (RX 7000/9000 series, Ryzen AI).

**Why it happens:** AMD announced ROCm Windows support at Computex 2025, but the "entire ROCm stack is not yet supported on Windows." It's a preview release with ongoing development.

**Consequences:**
- Older AMD GPUs (RX 6000 series) may not work on Windows
- `HSA_OVERRIDE_GFX_VERSION` workarounds don't work on Windows
- Inconsistent performance, memory issues on some configurations
- Requires specific driver versions (25.20.01.17 for ROCm 7.1.1)

**Known issues from AMD release notes:**
- "Intermittent script failure due to out-of-memory error" on Ryzen AI Max
- "Corruption may appear while running Stable Diffusion 3" on some APUs
- 32GB minimum system RAM recommended, 64GB for larger models

**Prevention:**
1. **Check GPU compatibility** before promising ROCm Windows support
2. **Prioritize Linux** for ROCm - it's more mature and tested
3. **Have CPU fallback** always ready
4. **Pin driver versions** in deployment documentation
5. **Test on actual hardware** - don't assume compatibility

**Detection:**
- Random crashes or memory errors on AMD Windows
- Performance inconsistency between runs
- Driver version mismatches

**Phase to address:** Phase 2 (GPU Backends) - investigate actual hardware compatibility before implementing.

**Sources:**
- [AMD ROCm 6.4.4 Release Notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html)
- [ROCm Windows Support Matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
- [The Road to ROCm on Radeon](https://www.amd.com/en/blogs/2025/the-road-to-rocm-on-radeon-for-windows-and-linux.html)

---

### Pitfall 4: CPU Inference Without Optimization is 10-50x Slower

**What goes wrong:** Running PyTorch inference on CPU with default settings results in unacceptably slow performance - minutes instead of seconds for audio analysis.

**Why it happens:** PyTorch CPU inference is not optimized out of the box. Without quantization, threading optimization, or ONNX conversion, CPU inference is dramatically slower than GPU.

**Consequences:**
- Linux server (CPU-only) becomes unusable for real-time analysis
- Users wait minutes instead of seconds
- Application appears broken on non-GPU machines

**Current code attempts mitigation:**
```python
# In ast_inference.py lines 14-18:
torch.set_num_threads(2)  # Use only 2 threads for inference
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
```
This prevents CPU blocking but doesn't improve speed.

**Prevention - CPU optimization ladder:**

1. **Level 1: Threading optimization**
   ```python
   # Match CPU cores (but not too many)
   torch.set_num_threads(min(4, os.cpu_count() or 2))
   ```

2. **Level 2: Eval mode and no_grad (already done)**
   ```python
   model.eval()
   with torch.no_grad():
       outputs = model(input)
   ```

3. **Level 3: INT8 Quantization (4x speedup)**
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

4. **Level 4: ONNX Runtime (2.5x+ speedup)**
   ```python
   # Export once
   torch.onnx.export(model, dummy_input, "model.onnx")
   # Use ONNX Runtime for inference
   import onnxruntime as ort
   session = ort.InferenceSession("model.onnx")
   ```

5. **Level 5: OpenVINO for Intel CPUs (additional 2x)**

**Detection:**
- Inference takes >10 seconds for single audio file
- CPU usage spikes to 100% for extended periods
- Users complain about "slow" on non-GPU machines

**Phase to address:** Phase 3 (CPU Optimization) - dedicated phase for this.

**Sources:**
- [Optimizing PyTorch Model Inference on CPU](https://towardsdatascience.com/optimizing-pytorch-model-inference-on-cpu/)
- [Boosting PyTorch Inference on CPU: Quantization to Multithreading](https://towardsdatascience.com/boosting-pytorch-inference-on-cpu-from-post-training-quantization-to-multithreading-6820ac7349bb/)
- [Model Inference Optimization Checklist - PyTorch Serve](https://docs.pytorch.org/serve/performance_checklist.html)

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or significant debugging time.

---

### Pitfall 5: PyTorch/Transformers Version Dependency Hell

**What goes wrong:** Different machines have different versions of PyTorch, transformers, and CUDA. Code works on development machine but fails on others with cryptic errors.

**Why it happens:**
- `pip install torch` installs different versions depending on when it's run
- Transformers library has tight coupling with PyTorch versions
- CUDA toolkit versions must match PyTorch build

**Example errors:**
- `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`
- `RuntimeError: CUDA error: no kernel image is available for execution on the device`
- `KeyError` during model loading from mismatched tokenizer/model versions

**Consequences:**
- "Works on my machine" syndrome
- Hours of debugging environment issues
- Production deployments fail silently

**Prevention:**
```bash
# Pin EXACT versions in requirements.txt
torch==2.3.1+cu121  # Include CUDA version in pin
torchaudio==2.3.1+cu121
transformers==4.40.0
```

```python
# Or use dependency lock files
pip install pip-tools
pip-compile requirements.in --generate-hashes
```

**Platform-specific wheels:**
```bash
# CUDA 12.1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# ROCm 5.7
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/rocm5.7

# CPU only
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

**Detection:**
- Different behavior on different machines
- Errors containing version numbers or attribute errors
- Import failures that only happen on some systems

**Phase to address:** Phase 1 (Foundation) - lock versions before adding complexity.

**Sources:**
- [Transformers/PyTorch Compatibility Issues](https://github.com/huggingface/transformers/issues/29763)
- [Handling PyTorch GPU Compatibility Hell](https://discuss.pytorch.org/t/how-to-deal-with-pytorch-gpu-compatibility-hell/224182)

---

### Pitfall 6: torchaudio vs librosa Audio Processing Inconsistencies

**What goes wrong:** Models trained with librosa preprocessing fail when deployed with torchaudio (or vice versa) due to subtle differences in spectrogram computation.

**Why it happens:**
- **Default sample rate:** librosa resamples to 22050Hz by default; torchaudio preserves original
- **MelSpectrogram parameters:** torchaudio uses `norm=None, htk=True` by default (non-librosa defaults)
- **dB scale conversion:** librosa returns dB-scaled spectrograms; torchaudio returns linear power

**Current code is CORRECT:**
```python
# ast_inference.py uses torchaudio consistently
self.mel_transform = T.MelSpectrogram(...)  # torchaudio
logmel = torch.log(melspec + 1e-9)  # Manual log scale
```
The code comments explicitly note: "Uses IDENTICAL preprocessing as training (torchaudio, not librosa!)"

**Consequences if violated:**
- Model accuracy drops dramatically (30-50% accuracy loss)
- Predictions appear random
- Very hard to debug - model loads fine, just gives wrong answers

**Prevention:**
1. **Document preprocessing pipeline** explicitly in model card
2. **Use same library for training and inference** - don't mix
3. **Save preprocessing parameters** with model checkpoint:
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'preprocessing': {
           'sample_rate': 48000,
           'n_fft': 2048,
           'hop_length': 512,
           'n_mels': 128,
           'library': 'torchaudio'
       }
   }, checkpoint_path)
   ```
4. **Add preprocessing validation** that compares outputs

**Detection:**
- Model accuracy is much worse in production than training
- Predictions seem random or biased toward one class
- Code contains both `import librosa` and `import torchaudio`

**Phase to address:** Phase 1 (Foundation) - validate current preprocessing, add to model metadata.

**Sources:**
- [Comparing Librosa, Soundfile and Torchaudio](https://nasseredd.github.io/blog/speech-and-language-processing/comparing-audio-libraries)
- [MelSpectrogram inconsistency with librosa - torchaudio Issue](https://github.com/pytorch/audio/issues/1058)

---

### Pitfall 7: Forgetting eval() Mode and no_grad() Context

**What goes wrong:** Model runs in training mode during inference, causing:
- Dropout layers randomly dropping values (non-deterministic outputs)
- BatchNorm using batch statistics instead of running statistics
- Gradient tracking consuming extra memory

**Why it happens:** Developers forget that PyTorch models default to training mode. The `model.eval()` call is easy to forget.

**Current code is CORRECT:**
```python
# ast_inference.py lines 55-57:
self.model.eval()  # Correct
self.model = self.model.to(self.device)

# Lines 129-130:
with torch.no_grad():  # Correct
    outputs = self.model(features).logits
```

**Consequences if violated:**
- Different predictions on same input (non-deterministic)
- CUDA out-of-memory errors from gradient accumulation
- 20-30% slower inference

**Prevention:**
```python
def predict(self, input):
    self.model.eval()  # Always before inference
    with torch.no_grad():  # Always wrap inference
        return self.model(input)
```

**Detection:**
- Same input gives different outputs
- Memory usage grows over time
- Predictions are less confident than expected

**Phase to address:** Already handled in current code - maintain during refactoring.

**Sources:**
- [How to PyTorch in Production](https://medium.com/data-science/how-to-pytorch-in-production-743cb6aac9d4)

---

### Pitfall 8: Cross-Platform Path Handling

**What goes wrong:** Windows uses backslashes (`\`), Linux/Mac use forward slashes (`/`). Hardcoded paths break across platforms.

**Why it happens:** String concatenation for paths works on development machine but fails elsewhere.

**Current code uses pathlib correctly:**
```python
# ast_inference.py line 10:
from pathlib import Path

# Line 39:
model_path = settings.AST_MODEL_PATH  # Should be Path object
```

**But common mistakes include:**
```python
# BAD:
model_path = "models/ast/" + filename  # Breaks on Windows

# GOOD:
model_path = Path("models") / "ast" / filename  # Works everywhere
```

**Consequences:**
- `FileNotFoundError` on Windows when paths have forward slashes in strings
- Model loading fails on different OS
- Audio file paths fail cross-platform

**Prevention:**
1. **Always use `pathlib.Path`** for file operations
2. **Never use string concatenation** for paths
3. **Test on both Windows and Linux** in CI

**Detection:**
- `FileNotFoundError` only on certain platforms
- Paths containing both `\` and `/`
- Path-related errors in logs

**Phase to address:** Phase 1 (Foundation) - audit all path handling.

**Sources:**
- [Windows FAQ - PyTorch](https://docs.pytorch.org/docs/stable/notes/windows.html)
- [How I Solved PyTorch's Cross-Platform Nightmare](https://svana.name/2025/09/how-i-solved-pytorchs-cross-platform-nightmare/)

---

## Minor Pitfalls

Mistakes that cause annoyance but are relatively easy to fix.

---

### Pitfall 9: Audio Sample Rate Mismatches

**What goes wrong:** Input audio is at different sample rate than model expects. Model still runs but gives poor predictions.

**Current code expects:**
```python
# 48kHz sample rate (from settings.SAMPLE_RATE)
# 2.97s segments = 142560 samples
```

**Prevention:**
```python
def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != EXPECTED_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, EXPECTED_SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform
```

**Detection:**
- Audio files from different sources give poor predictions
- Sample rate mentioned in error messages
- Audio duration calculations are wrong

**Phase to address:** Phase 1 (Foundation) - add sample rate validation and auto-resampling.

---

### Pitfall 10: Windows torchaudio Backend Issues

**What goes wrong:** torchaudio on Windows historically had issues with certain audio formats and backends (SoX was not supported).

**Current status:** Modern torchaudio uses different backends per platform:
- Windows: `soundfile` backend (no SoX)
- Linux: Can use `sox`, `soundfile`, or `sox_io`

**Prevention:**
1. **Stick to widely-supported formats:** WAV, MP3, FLAC
2. **Check backend on startup:**
   ```python
   print(f"torchaudio backend: {torchaudio.get_audio_backend()}")
   ```
3. **Test audio loading** on each target platform

**Detection:**
- `RuntimeError: No audio I/O backend is available`
- Audio format errors only on Windows
- OGG files failing to load

**Phase to address:** Phase 1 (Foundation) - validate audio backends, document supported formats.

---

### Pitfall 11: CUDA Fork Safety Initialization

**What goes wrong:** In multiprocessing scenarios, calling `torch.cuda.is_available()` before fork causes child processes to fail with CUDA initialization errors.

**Why it happens:** CUDA driver gets initialized by `is_available()` check, and this state doesn't transfer correctly across fork().

**Prevention:**
```bash
# Set before importing PyTorch
export PYTORCH_NVML_BASED_CUDA_CHECK=1
```

Or use spawn instead of fork for multiprocessing:
```python
import multiprocessing
multiprocessing.set_start_method('spawn')
```

**Detection:**
- CUDA errors only when using multiprocessing
- "CUDA initialization error" in child processes
- Works fine in single-process mode

**Phase to address:** Phase 4 (Stability) - if implementing parallel processing.

**Sources:**
- [CUDA Semantics - PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html)

---

## Phase-Specific Warnings

Summary of which pitfalls to watch for in each phase:

| Phase | Topic | Likely Pitfalls | Priority Mitigation |
|-------|-------|-----------------|---------------------|
| Phase 1 | Foundation | #1 Device detection, #2 Checkpoint format, #5 Version pinning, #6 Audio preprocessing, #8 Path handling | Lock versions, audit device handling |
| Phase 2 | GPU Backends | #3 ROCm Windows limitations, #1 Device names | Test on actual AMD hardware before implementing |
| Phase 3 | CPU Optimization | #4 CPU slowness | Implement optimization ladder (quantization, ONNX) |
| Phase 4 | Stability | #5 Dependency management, #11 Fork safety | CI/CD for all platforms |

---

## Validation Checklist

Before considering the milestone complete:

- [ ] Same model checkpoint works on all three target machines
- [ ] Device detection reports correct backend (CUDA/ROCm/CPU)
- [ ] CPU inference completes in reasonable time (<30s for audio file)
- [ ] Dependencies are pinned with exact versions
- [ ] Audio preprocessing is validated against training pipeline
- [ ] All paths use pathlib, tested on Windows and Linux
- [ ] No "works on my machine" issues between development environments

---

## Sources Summary

**Official Documentation:**
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch HIP/ROCm Semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [PyTorch Windows FAQ](https://docs.pytorch.org/docs/stable/notes/windows.html)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Hugging Face Transformers AST](https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer)

**Verified Community Resources:**
- [Model Inference Optimization Checklist - PyTorch Serve](https://docs.pytorch.org/serve/performance_checklist.html)
- [AMD ROCm Release Notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html)

**Research/Blog Posts (MEDIUM confidence):**
- [Optimizing PyTorch CPU Inference - TDS](https://towardsdatascience.com/optimizing-pytorch-model-inference-on-cpu/)
- [Cross-Platform PyTorch Solutions](https://svana.name/2025/09/how-i-solved-pytorchs-cross-platform-nightmare/)

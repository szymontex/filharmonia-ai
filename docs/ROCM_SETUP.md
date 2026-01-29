# ROCm Setup Guide

This guide covers GPU setup for Filharmonia AI across different hardware platforms.

## Linux ROCm 6.4 (Production)

### Supported GPUs

- AMD Radeon RX 7000 series (RDNA 3)
- AMD Radeon RX 9000 series (RDNA 4)
- AMD Instinct MI series (data center)

### Installation

1. **Install ROCm 6.4+** following the official guide:
   https://rocm.docs.amd.com/

2. **Install PyTorch with ROCm support:**

   ```bash
   pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
   ```

3. **Verify installation:**

   ```bash
   python -c "import torch; print('HIP:', torch.version.hip, 'CUDA available:', torch.cuda.is_available())"
   ```

   Expected output: `HIP: 6.2.xxxxx CUDA available: True`

   (ROCm exposes GPUs through the CUDA compatibility layer, so `torch.cuda.is_available()` returns `True` for AMD GPUs.)

### Troubleshooting

- **`torch.cuda.is_available()` returns `False`:** Ensure ROCm drivers are installed and your GPU is in the supported list. Run `rocminfo` to verify the driver sees your GPU.
- **Inference is unexpectedly slow:** Filharmonia AI runs a performance validation at startup. Check logs for "ROCm may be falling back to CPU silently" — this means the GPU path is not working correctly. Reinstall ROCm drivers.
- **`libamdhip64.so` not found:** Add ROCm libraries to your path: `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH`

## Windows ROCm (Experimental)

> **Status:** Preview support as of ROCm 6.4.4 (January 2026). May be unstable.

### Supported GPUs

- AMD Radeon RX 7000 series (RDNA 3)
- AMD Radeon RX 9000 series (RDNA 4)
- AMD Ryzen AI APUs with integrated RDNA graphics

### Installation

1. Download and install the AMD ROCm Windows installer from:
   https://rocm.docs.amd.com/

2. Install PyTorch with ROCm support (same command as Linux):

   ```bash
   pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
   ```

### Important Notes

- Windows ROCm support is experimental and may produce incorrect results or crashes.
- Filharmonia AI will automatically detect and validate GPU performance at startup.
- If the GPU is not performing correctly, the application falls back to CPU transparently.
- Check application logs for any ROCm warnings.

## NVIDIA CUDA Setup

For NVIDIA GPUs (GTX 1000 series and newer):

```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`

## CPU-Only Setup

For systems without a supported GPU:

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

For best CPU performance, export the ONNX INT8 quantized model which provides 3x+ speedup over eager PyTorch inference. See the ONNX export script in the backend tools directory.

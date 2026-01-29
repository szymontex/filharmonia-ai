"""
Unified device detection for GPU (NVIDIA CUDA / AMD ROCm) and CPU.
Singleton pattern — detection runs once at startup.
"""
import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Detects and manages compute device (NVIDIA CUDA, AMD ROCm, or CPU)."""

    def __init__(self):
        self.device: torch.device = torch.device("cpu")
        self.device_type: str = "cpu"
        self.device_name: str = "CPU"
        self._detected: bool = False
        self._gpu_validated: bool = True

    def detect_device(self) -> None:
        """Detect available compute device. Safe to call multiple times (no-op after first)."""
        if self._detected:
            return
        self._detected = True

        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()

                # Distinguish AMD ROCm from NVIDIA CUDA
                if hasattr(torch.version, "hip") and torch.version.hip is not None:
                    self.device_type = "cuda_amd"
                    version = torch.version.hip
                    logger.info(
                        "GPU detected: AMD ROCm %s — %s (%d device(s))",
                        version, self.device_name, device_count,
                    )
                    self._validate_rocm()
                else:
                    self.device_type = "cuda_nvidia"
                    version = torch.version.cuda or "unknown"
                    logger.info(
                        "GPU detected: NVIDIA CUDA %s — %s (%d device(s))",
                        version, self.device_name, device_count,
                    )
            else:
                self._set_cpu()
        except Exception as exc:
            logger.error("GPU detection failed, falling back to CPU: %s", exc)
            self._set_cpu()

    def _validate_rocm(self) -> None:
        """Run ROCm-specific validation: log device info and check for silent CPU fallback."""
        try:
            # Log GPU memory for debugging
            props = torch.cuda.get_device_properties(0)
            total_mem_gb = props.total_mem / (1024 ** 3)
            logger.info(
                "ROCm GPU memory: %.1f GB (%s)", total_mem_gb, self.device_name
            )

            # Quick sanity check — trivial matmul should be <100ms on real GPU
            torch.cuda.synchronize()
            start = time.perf_counter()
            a = torch.randn(10, 10, device="cuda")
            b = torch.randn(10, 10, device="cuda")
            _ = a @ b
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            if elapsed_ms > 100:
                logger.warning(
                    "ROCm trivial matmul took %.0f ms (expected <100 ms) — "
                    "possible silent CPU fallback",
                    elapsed_ms,
                )

            # Full performance validation
            self._gpu_validated = self.validate_gpu_performance()
        except Exception as exc:
            logger.error("ROCm validation failed, falling back to CPU: %s", exc)
            self._set_cpu()

    def validate_gpu_performance(self) -> bool:
        """Compare GPU vs CPU on a 256x256 matmul. Returns True if GPU is faster.

        If GPU is slower than CPU, logs a CRITICAL warning about possible
        silent CPU fallback (common with misconfigured ROCm).
        """
        try:
            size = 256

            # GPU timing
            torch.cuda.synchronize()
            gpu_start = time.perf_counter()
            ga = torch.randn(size, size, device="cuda")
            gb = torch.randn(size, size, device="cuda")
            _ = ga @ gb
            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - gpu_start) * 1000

            # CPU timing
            cpu_start = time.perf_counter()
            ca = torch.randn(size, size, device="cpu")
            cb = torch.randn(size, size, device="cpu")
            _ = ca @ cb
            cpu_ms = (time.perf_counter() - cpu_start) * 1000

            logger.info(
                "GPU performance check: GPU=%.1f ms, CPU=%.1f ms (256x256 matmul)",
                gpu_ms, cpu_ms,
            )

            if gpu_ms > cpu_ms:
                logger.critical(
                    "ROCm GPU (%.1f ms) is SLOWER than CPU (%.1f ms) — "
                    "ROCm may be falling back to CPU silently. "
                    "Check driver installation and GPU compatibility.",
                    gpu_ms, cpu_ms,
                )
                return False

            return True
        except Exception as exc:
            logger.error("GPU performance validation failed: %s", exc)
            return False

    def _set_cpu(self) -> None:
        self.device = torch.device("cpu")
        self.device_type = "cpu"
        self.device_name = "CPU"
        logger.warning("No GPU detected — using CPU for inference")

    @property
    def is_gpu(self) -> bool:
        return self.device_type != "cpu"

    @property
    def supports_compile(self) -> bool:
        return self.is_gpu

    @property
    def rocm_version(self) -> Optional[str]:
        """Return ROCm/HIP version string, or None if not running on ROCm."""
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return torch.version.hip
        return None


# ── Singleton ──────────────────────────────────────────────────────────
_instance: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Return the singleton DeviceManager, detecting device on first call."""
    global _instance
    if _instance is None:
        _instance = DeviceManager()
        _instance.detect_device()
    return _instance

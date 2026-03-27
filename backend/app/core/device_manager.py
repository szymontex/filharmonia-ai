"""
Unified device detection for GPU (NVIDIA CUDA / AMD ROCm / Apple MPS) and CPU.
Singleton pattern — detection runs once at startup.
"""
import logging
import os
import platform
import subprocess
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Detects and manages compute device (NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU)."""

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

            elif self._detect_mps():
                pass  # MPS setup done inside _detect_mps

            else:
                self._set_cpu()
        except Exception as exc:
            logger.error("Device detection failed, falling back to CPU: %s", exc)
            self._set_cpu()

    # ── Apple MPS ─────────────────────────────────────────────────────

    def _detect_mps(self) -> bool:
        """Detect and validate Apple MPS (Metal Performance Shaders). Returns True if usable."""
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return False

        try:
            self.device = torch.device("mps")
            self.device_type = "mps"
            self.device_name = self._get_apple_chip_name()

            logger.info(
                "GPU detected: Apple MPS — %s (%s)",
                self.device_name, self._get_apple_memory_info(),
            )

            # Validate MPS actually works (some ops may silently fall back to CPU)
            if not self._validate_mps():
                logger.warning("MPS validation failed — falling back to CPU")
                self._set_cpu()
                return False

            return True
        except Exception as exc:
            logger.warning("MPS detection failed: %s", exc)
            return False

    def _validate_mps(self) -> bool:
        """Quick benchmark: MPS must be faster than CPU on a 256x256 matmul."""
        try:
            size = 256

            # MPS timing (warmup + measure)
            warmup = torch.randn(size, size, device="mps") @ torch.randn(size, size, device="mps")
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            mps_start = time.perf_counter()
            a = torch.randn(size, size, device="mps")
            b = torch.randn(size, size, device="mps")
            _ = a @ b
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            mps_ms = (time.perf_counter() - mps_start) * 1000

            # CPU timing
            cpu_start = time.perf_counter()
            ca = torch.randn(size, size, device="cpu")
            cb = torch.randn(size, size, device="cpu")
            _ = ca @ cb
            cpu_ms = (time.perf_counter() - cpu_start) * 1000

            logger.info(
                "MPS performance check: MPS=%.1f ms, CPU=%.1f ms (256x256 matmul)",
                mps_ms, cpu_ms,
            )

            if mps_ms > cpu_ms * 3:
                logger.warning(
                    "MPS (%.1f ms) is significantly slower than CPU (%.1f ms) — "
                    "possible driver issue or unsupported operation fallback",
                    mps_ms, cpu_ms,
                )
                return False

            return True
        except Exception as exc:
            logger.warning("MPS validation benchmark failed: %s", exc)
            return False

    @staticmethod
    def _get_apple_chip_name() -> str:
        """Get Apple Silicon chip name (e.g., 'Apple M1 Pro')."""
        if platform.system() != "Darwin":
            return "Apple Silicon"
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            name = result.stdout.strip()
            return name if name else "Apple Silicon"
        except Exception:
            return "Apple Silicon"

    @staticmethod
    def _get_apple_memory_info() -> str:
        """Get total system memory (unified with GPU on Apple Silicon)."""
        if platform.system() != "Darwin":
            return "unknown memory"
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            mem_bytes = int(result.stdout.strip())
            mem_gb = mem_bytes / (1024 ** 3)
            return f"{mem_gb:.0f} GB unified memory"
        except Exception:
            return "unknown memory"

    # ── AMD ROCm ──────────────────────────────────────────────────────

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

    # ── CPU fallback ──────────────────────────────────────────────────

    def _set_cpu(self) -> None:
        self.device = torch.device("cpu")
        self.device_type = "cpu"
        self.device_name = "CPU"
        logger.warning("No GPU detected — using CPU for inference")

    # ── Properties ────────────────────────────────────────────────────

    @property
    def is_gpu(self) -> bool:
        return self.device_type != "cpu"

    @property
    def is_mps(self) -> bool:
        return self.device_type == "mps"

    @property
    def is_cuda(self) -> bool:
        return self.device_type in ("cuda_nvidia", "cuda_amd")

    @property
    def supports_compile(self) -> bool:
        import sys
        # torch.compile requires Triton which is only available on CUDA (not MPS, not Windows)
        if sys.platform == "win32":
            return False
        if not self.is_cuda:
            return False
        # Triton requires CUDA Capability >= 7.0 (Volta+)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:
                return False
        return True

    @property
    def recommended_threads(self) -> int:
        """Recommended torch thread count for this platform."""
        if self.is_cuda:
            # GPU does heavy lifting; CPU just feeds data
            return 2

        # For MPS and CPU: use available cores wisely
        perf_cores = self._get_performance_core_count()

        if self.is_mps:
            # MPS: use more CPU threads for preprocessing (GPU handles inference)
            return min(perf_cores, 6)
        else:
            # CPU-only: leave headroom for web server and OS
            return min(max(perf_cores // 2, 2), 4)

    @property
    def recommended_batch_size(self) -> int:
        """Recommended inference batch size for this platform."""
        if self.is_cuda:
            return 32
        if self.is_mps:
            # Unified memory = no transfer overhead, can use larger batches
            return 64
        return 32

    @staticmethod
    def _get_performance_core_count() -> int:
        """Get performance core count (P-cores on Apple Silicon, total/2 on others)."""
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                    capture_output=True, text=True, timeout=5,
                )
                count = int(result.stdout.strip())
                if count > 0:
                    return count
            except Exception:
                pass

        # Fallback: half of total logical cores
        total = os.cpu_count() or 4
        return max(total // 2, 2)

    @property
    def rocm_version(self) -> Optional[str]:
        """Return ROCm/HIP version string, or None if not running on ROCm."""
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return torch.version.hip
        return None

    def sync_device(self) -> None:
        """Synchronize the current device (for accurate timing)."""
        if self.is_cuda:
            torch.cuda.synchronize()
        elif self.is_mps and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


# ── Singleton ──────────────────────────────────────────────────────────
_instance: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Return the singleton DeviceManager, detecting device on first call."""
    global _instance
    if _instance is None:
        _instance = DeviceManager()
        _instance.detect_device()
    return _instance

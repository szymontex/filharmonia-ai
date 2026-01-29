"""
Unified device detection for GPU (NVIDIA CUDA / AMD ROCm) and CPU.
Singleton pattern — detection runs once at startup.
"""
import logging
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


# ── Singleton ──────────────────────────────────────────────────────────
_instance: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Return the singleton DeviceManager, detecting device on first call."""
    global _instance
    if _instance is None:
        _instance = DeviceManager()
        _instance.detect_device()
    return _instance

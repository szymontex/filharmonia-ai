"""
Inference factory — routes to optimal backend based on device and model availability.

CUDA/ROCm GPU  -> ASTInferenceService (PyTorch GPU)
Apple MPS      -> ASTInferenceService (PyTorch MPS — Apple Silicon GPU)
CPU + ONNX     -> ASTInferenceONNXService (INT8 quantized, ~3x faster than eager)
CPU + no ONNX  -> ASTInferenceService (eager PyTorch fallback)
"""
import logging

from app.config import settings
from app.core.device_manager import get_device_manager

logger = logging.getLogger(__name__)


def create_inference_service():
    """Create the optimal inference service for the current environment.

    Returns:
        ASTInferenceService or ASTInferenceONNXService instance.
    """
    dm = get_device_manager()
    onnx_exists = settings.AST_ONNX_MODEL_PATH.exists()

    # GPU path (CUDA, ROCm, or MPS) — always use PyTorch
    if dm.is_gpu:
        from app.services.ast_inference import ASTInferenceService

        if dm.is_mps:
            logger.info(
                "Inference backend: PyTorch MPS (%s, batch_size=%d)",
                dm.device_name, dm.recommended_batch_size,
            )
        else:
            reason = "GPU detected" + (", ONNX model ignored (GPU faster)" if onnx_exists else "")
            logger.info("Inference backend: PyTorch GPU (%s)", reason)
        return ASTInferenceService()

    # CPU path — prefer ONNX INT8 if available
    if onnx_exists:
        try:
            from app.services.ast_inference_onnx import ASTInferenceONNXService

            logger.info("Inference backend: ONNX INT8 CPU (model found at %s)", settings.AST_ONNX_MODEL_PATH.name)
            return ASTInferenceONNXService()
        except Exception as exc:
            logger.warning(
                "Failed to create ONNX service, falling back to PyTorch CPU: %s", exc
            )
            from app.services.ast_inference import ASTInferenceService

            return ASTInferenceService()

    # CPU without ONNX — eager PyTorch fallback
    from app.services.ast_inference import ASTInferenceService

    logger.info("Inference backend: PyTorch CPU (no ONNX model found)")
    return ASTInferenceService()

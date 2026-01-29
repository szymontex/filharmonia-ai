"""
Export PyTorch AST model to ONNX format with INT8 dynamic quantization.

Usage:
    cd backend && python -m scripts.export_onnx
    cd backend && python -m scripts.export_onnx --model-path /path/to/model.pth
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import ASTForAudioClassification

# Add backend to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_pytorch_model(model_path: Path) -> torch.nn.Module:
    """Load the PyTorch AST model from checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=5,
        ignore_mismatched_sizes=True,
    )
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded PyTorch model from %s", model_path.name)
    if "val_acc" in checkpoint:
        logger.info("  Validation accuracy: %.2f%%", checkpoint["val_acc"])
    return model


def export_to_onnx(model: torch.nn.Module, output_path: Path) -> Path:
    """Export model to ONNX format."""
    fp32_path = output_path.with_suffix(".fp32.onnx")
    dummy_input = torch.randn(1, 1024, 128)

    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        opset_version=14,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    logger.info("Exported FP32 ONNX model: %s", fp32_path.name)
    return fp32_path


def quantize_int8(fp32_path: Path, int8_path: Path) -> Path:
    """Apply dynamic INT8 quantization."""
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        optimize_model=True,
    )
    logger.info("Quantized to INT8: %s", int8_path.name)
    return int8_path


def validate_and_report(fp32_path: Path, int8_path: Path, model: torch.nn.Module):
    """Validate ONNX model and report compression + accuracy."""
    # Validate
    onnx_model = onnx.load(str(int8_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed")

    # File sizes
    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)
    ratio = fp32_size / int8_size if int8_size > 0 else 0
    logger.info(
        "Size: FP32=%.1fMB, INT8=%.1fMB, compression=%.1fx",
        fp32_size, int8_size, ratio,
    )

    # Accuracy comparison
    import onnxruntime as ort

    dummy = torch.randn(1, 1024, 128)
    with torch.no_grad():
        pt_logits = model(dummy).logits.numpy()

    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input_values": dummy.numpy()})[0]

    max_diff = np.max(np.abs(pt_logits - onnx_logits))
    logger.info("Max logits difference (PyTorch vs ONNX INT8): %.4f", max_diff)
    if max_diff < 0.1:
        logger.info("Accuracy check PASSED (atol=0.1)")
    else:
        logger.warning("Accuracy check WARNING: max diff %.4f > 0.1", max_diff)

    # Cleanup FP32 intermediate
    fp32_path.unlink()
    logger.info("Removed intermediate FP32 model")


def main():
    parser = argparse.ArgumentParser(description="Export AST model to ONNX INT8")
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to PyTorch checkpoint (default: from config)",
    )
    args = parser.parse_args()

    model_path = args.model_path or settings.AST_MODEL_PATH
    output_path = settings.AST_ONNX_MODEL_PATH

    logger.info("=== ONNX Export Pipeline ===")
    logger.info("Input:  %s", model_path)
    logger.info("Output: %s", output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_pytorch_model(model_path)
    fp32_path = export_to_onnx(model, output_path)
    quantize_int8(fp32_path, output_path)
    validate_and_report(fp32_path=fp32_path, int8_path=output_path, model=model)

    logger.info("=== Export complete: %s ===", output_path.name)


if __name__ == "__main__":
    main()

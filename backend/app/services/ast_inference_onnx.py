"""
ONNX Runtime inference service for CPU with INT8 quantization.
Drop-in replacement for ASTInferenceService on CPU-only systems.
"""
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.transforms as T

from app.config import settings

logger = logging.getLogger(__name__)


class ASTInferenceONNXService:
    """ONNX Runtime inference service — same interface as ASTInferenceService."""

    def __init__(self):
        self.session = None

        # Create mel-spectrogram transform (IDENTICAL to ASTInferenceService)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=settings.SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )

    def load_model(self, model_path: Path = None):
        """Load ONNX model via InferenceSession."""
        if model_path is None:
            model_path = settings.AST_ONNX_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX INT8 model loaded: %s", model_path.name)

        self._benchmark_speedup()

    def _benchmark_speedup(self):
        """Compare ONNX INT8 vs eager PyTorch CPU latency."""
        try:
            dummy = np.random.randn(1, 1024, 128).astype(np.float32)

            # ONNX timing (3 runs, average)
            onnx_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                self.session.run(None, {"input_values": dummy})
                onnx_times.append(time.perf_counter() - t0)
            onnx_ms = (sum(onnx_times) / len(onnx_times)) * 1000

            # PyTorch CPU timing
            from app.services.ast_inference import ASTInferenceService

            pt_service = ASTInferenceService()
            # Force CPU
            pt_service.device = torch.device("cpu")
            pt_service.load_model()
            dummy_tensor = torch.from_numpy(dummy)

            pt_times = []
            with torch.no_grad():
                for _ in range(3):
                    t0 = time.perf_counter()
                    pt_service.model(dummy_tensor)
                    pt_times.append(time.perf_counter() - t0)
            pt_ms = (sum(pt_times) / len(pt_times)) * 1000

            ratio = pt_ms / onnx_ms if onnx_ms > 0 else 0
            logger.info(
                "ONNX INT8 speedup: %.1fx vs CPU PyTorch (%.1fms vs %.1fms)",
                ratio, onnx_ms, pt_ms,
            )
            if ratio < 3.0:
                logger.warning(
                    "ONNX INT8 speedup %.1fx is below 3x target", ratio
                )

            # Discard temporary PyTorch instance
            del pt_service
        except Exception as exc:
            logger.warning("Benchmark comparison failed: %s", exc)

    def preprocess_audio_segment(self, audio_segment: np.ndarray) -> torch.Tensor:
        """
        Convert raw audio segment to AST input format.
        IDENTICAL preprocessing to ASTInferenceService.
        """
        waveform = torch.from_numpy(audio_segment).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        melspec = self.mel_transform(waveform)
        logmel = torch.log(melspec + 1e-9)
        logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-9)
        logmel = logmel.squeeze(0)
        logmel = logmel.T

        if logmel.shape[0] < 1024:
            pad_width = 1024 - logmel.shape[0]
            logmel = torch.nn.functional.pad(logmel, (0, 0, 0, pad_width))
        elif logmel.shape[0] > 1024:
            logmel = logmel[:1024, :]

        tensor = logmel.unsqueeze(0)
        return tensor

    def predict_segment(self, audio_segment: np.ndarray) -> str:
        """Predict class for a single audio segment via ONNX."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        features = self.preprocess_audio_segment(audio_segment)
        input_np = features.numpy()

        logits = self.session.run(None, {"input_values": input_np})[0]
        predicted_idx = int(np.argmax(logits, axis=1)[0])
        return settings.LABELS[predicted_idx]

    def predict_batch(self, audio_segments: list) -> list:
        """Predict classes for multiple segments via ONNX."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        features_list = [self.preprocess_audio_segment(seg) for seg in audio_segments]
        features_batch = torch.cat(features_list, dim=0)
        input_np = features_batch.numpy()

        logits = self.session.run(None, {"input_values": input_np})[0]

        # Softmax for confidence scores
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        predicted_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        predictions = [
            (settings.LABELS[idx], float(conf))
            for idx, conf in zip(predicted_indices, confidences)
        ]
        return predictions

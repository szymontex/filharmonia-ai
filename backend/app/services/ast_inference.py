"""
PyTorch AST Inference Service
Replaces Keras CNN with Audio Spectrogram Transformer
"""
import torch
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T
from pathlib import Path
from transformers import ASTForAudioClassification
from app.config import settings


class ASTInferenceService:
    """AST model inference service"""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create mel-spectrogram transform (IDENTICAL to training!)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=settings.SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

    def load_model(self, model_path: Path = None):
        """Load AST model from checkpoint"""
        if model_path is None:
            model_path = settings.AST_MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"AST model not found: {model_path}")

        # Load pretrained AST architecture
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=5,
            ignore_mismatched_sizes=True
        )

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Set to evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)

        print(f"[OK] AST model loaded: {model_path.name}")
        print(f"  Device: {self.device}")
        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")

    def preprocess_audio_segment(self, audio_segment: np.ndarray) -> torch.Tensor:
        """
        Convert raw audio segment to AST input format
        Uses IDENTICAL preprocessing as training (torchaudio, not librosa!)

        Args:
            audio_segment: Raw audio (48kHz, 2.97s = 142560 samples)

        Returns:
            Tensor of shape (1, 1024, 128) - mel-spectrogram for AST
        """
        # Convert numpy to torch tensor
        waveform = torch.from_numpy(audio_segment).float()

        # Add channel dimension if needed: (samples,) -> (1, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Generate mel-spectrogram using torchaudio (IDENTICAL to training!)
        melspec = self.mel_transform(waveform)

        # Convert to log scale (IDENTICAL to training!)
        logmel = torch.log(melspec + 1e-9)

        # Normalize to [0, 1] (IDENTICAL to training!)
        logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-9)

        # Remove channel dimension: (1, 128, time) -> (128, time)
        logmel = logmel.squeeze(0)

        # Transpose: (128 mels, time) -> (time, 128 mels)
        logmel = logmel.T

        # Pad or trim to 1024 time frames (AST requirement)
        if logmel.shape[0] < 1024:
            # Pad with zeros
            pad_width = 1024 - logmel.shape[0]
            logmel = torch.nn.functional.pad(logmel, (0, 0, 0, pad_width))
        elif logmel.shape[0] > 1024:
            # Trim
            logmel = logmel[:1024, :]

        # Add batch dimension: (1024, 128) -> (1, 1024, 128)
        tensor = logmel.unsqueeze(0)

        return tensor

    def predict_segment(self, audio_segment: np.ndarray) -> str:
        """
        Predict class for a single 2.97s audio segment

        Args:
            audio_segment: Raw audio (48kHz, 2.97s)

        Returns:
            Predicted class name (APPLAUSE, MUSIC, PUBLIC, SPEECH, TUNING)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        features = self.preprocess_audio_segment(audio_segment)
        features = features.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(features).logits
            predicted_idx = torch.argmax(outputs, dim=1).item()

        # Convert index to class name
        predicted_class = settings.LABELS[predicted_idx]

        return predicted_class

    def predict_batch(self, audio_segments: list) -> list:
        """
        Predict classes for multiple segments (batch processing)

        Args:
            audio_segments: List of raw audio segments

        Returns:
            List of (predicted_class_name, confidence_score) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess all segments
        features_list = [self.preprocess_audio_segment(seg) for seg in audio_segments]
        features_batch = torch.cat(features_list, dim=0)  # (batch_size, 1024, 128)
        features_batch = features_batch.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(features_batch).logits

            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Get predicted class and its confidence
            confidences, predicted_indices = torch.max(probs, dim=1)

            predicted_indices = predicted_indices.cpu().numpy()
            confidences = confidences.cpu().numpy()

        # Convert indices to class names with confidence scores
        predictions = [
            (settings.LABELS[idx], float(conf))
            for idx, conf in zip(predicted_indices, confidences)
        ]

        return predictions


# Singleton instance
_service = None

def get_ast_inference_service() -> ASTInferenceService:
    """Get singleton AST inference service"""
    global _service
    if _service is None:
        _service = ASTInferenceService()
        # Try to load model on startup (fail silently if not available yet)
        try:
            _service.load_model()
        except FileNotFoundError:
            print("⚠ AST model not found - will need to train first")
    return _service

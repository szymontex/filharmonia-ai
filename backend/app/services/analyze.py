"""
Audio Analysis Service
Uses PyTorch AST (Audio Spectrogram Transformer) for inference
"""
import os
import time
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
import librosa
import eyed3
from app.config import settings
from app.services.ast_inference import get_ast_inference_service

class AnalyzeService:
    def __init__(self):
        self.ast_service = get_ast_inference_service()

    def load_model(self):
        """Load AST model (compatibility method)"""
        self.ast_service.load_model()

    def analyze_file(self, mp3_path: Path, output_csv: Path = None, on_progress=None, check_cancelled=None) -> dict:
        """
        Analyze one MP3 file using PyTorch AST

        Args:
            mp3_path: Path to MP3 file
            output_csv: Optional output CSV path
            on_progress: Callback function(current, total, percent)
            check_cancelled: Callback to check if cancelled

        Returns:
            dict with csv_path, segments_analyzed, duration_seconds
        """
        # Load audio
        y, sr = librosa.load(str(mp3_path), sr=settings.SAMPLE_RATE)

        # Segment audio into 2.97s chunks
        frame_length = int(settings.FRAME_DURATION_SEC * settings.SAMPLE_RATE)
        num_segments = (len(y) + frame_length - 1) // frame_length  # Ceil division

        # Przygotuj output CSV
        if output_csv is None:
            # Extract date from folder structure: SORTED/YYYY/MM/DD/file.mp3
            try:
                rel_path = mp3_path.relative_to(settings.SORTED_FOLDER)
                parts = rel_path.parts
                if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                    concert_date = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                else:
                    concert_date = time.strftime('%Y-%m-%d')
            except ValueError:
                # Fallback if not in SORTED folder structure
                concert_date = time.strftime('%Y-%m-%d')

            # Extract time from ID3 tag
            time_str = ""
            try:
                audiofile = eyed3.load(str(mp3_path))
                if audiofile and audiofile.tag and audiofile.tag.title:
                    # Parse: "Untitled %m/%d/%Y %H:%M:%S"
                    record_date = datetime.strptime(audiofile.tag.title, 'Untitled %m/%d/%Y %H:%M:%S')
                    time_str = f"_{record_date.hour:02d}-{record_date.minute:02d}"
            except:
                pass  # If can't extract time, leave empty

            output_csv = settings.SORTED_FOLDER / "ANALYSIS_RESULTS" / f"predictions_{mp3_path.stem}_{concert_date}{time_str}.csv"

        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file during analysis to avoid showing incomplete CSV
        output_csv_tmp = output_csv.with_suffix('.csv.tmp')

        # Clean up any existing temp file from previous incomplete analysis
        if output_csv_tmp.exists():
            output_csv_tmp.unlink()

        # Get active model version for CSV tracking
        from app.services.model_registry import get_active_model_id
        model_version = get_active_model_id()

        # Analyze segments using PyTorch AST with batch processing
        BATCH_SIZE = 32  # Process 32 segments at once for better GPU utilization

        with open(output_csv_tmp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['segment_time', 'predicted_class', 'confidence', 'model_version'])

            # Process in batches
            for batch_start in range(0, num_segments, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_segments)

                # Check if cancelled
                if check_cancelled and check_cancelled():
                    if output_csv_tmp.exists():
                        output_csv_tmp.unlink()
                    raise InterruptedError("Analysis cancelled by user")

                # Extract batch of segments
                batch_segments = []
                for i in range(batch_start, batch_end):
                    start = i * frame_length
                    end = min((i + 1) * frame_length, len(y))
                    audio_segment = y[start:end]

                    # Pad if shorter
                    if len(audio_segment) < frame_length:
                        audio_segment = np.pad(audio_segment, (0, frame_length - len(audio_segment)))

                    batch_segments.append(audio_segment)

                # Predict batch using AST (now returns list of (class, confidence) tuples)
                predictions = self.ast_service.predict_batch(batch_segments)

                # Write results
                for i, (predicted_class, confidence) in enumerate(predictions):
                    segment_idx = batch_start + i
                    segment_time_sec = segment_idx * settings.FRAME_DURATION_SEC
                    formatted_time = time.strftime('%H:%M:%S', time.gmtime(segment_time_sec))
                    writer.writerow([formatted_time, predicted_class, f"{confidence:.4f}", model_version])

                # Progress callback (after each batch)
                if on_progress:
                    progress = batch_end / num_segments * 100
                    on_progress(batch_end, num_segments, progress)

        # Rename temporary file to final CSV
        # Windows requires removing destination file first (unlike Linux)
        if output_csv.exists():
            output_csv.unlink()
        output_csv_tmp.rename(output_csv)

        return {
            'csv_path': str(output_csv),
            'segments_analyzed': num_segments,
            'duration_seconds': len(y) / sr
        }

# Singleton instance
_service = None

def get_analyze_service():
    global _service
    if _service is None:
        _service = AnalyzeService()
    return _service

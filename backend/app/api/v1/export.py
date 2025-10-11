"""
Export API - export audio segments to training data
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import librosa
import soundfile as sf
from pathlib import Path
import csv
from datetime import datetime
from app.config import settings

router = APIRouter(prefix="/export", tags=["export"])

class ExportSegment(BaseModel):
    start: float
    stop: float
    predicted_class: str
    segment_index: int
    segment_time: str

class ExportRequest(BaseModel):
    csv_path: str
    mp3_path: str
    segments: List[ExportSegment]

class ExportResponse(BaseModel):
    exported: List[dict]
    skipped: List[dict]
    errors: List[dict]
    summary: dict

def get_tracking_csv_path() -> Path:
    """Get path to exported segments tracking CSV"""
    return settings.TRAINING_DATA_FOLDER.parent / "exported_segments.csv"

def is_segment_exported(csv_path: str, segment_index: int) -> bool:
    """Check if segment was already exported"""
    tracking_csv = get_tracking_csv_path()
    if not tracking_csv.exists():
        return False

    with open(tracking_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['csv_path'] == csv_path and int(row['segment_index']) == segment_index:
                return True
    return False

def log_export(csv_path: str, segment_index: int, segment_time: str, class_name: str, exported_path: str):
    """Log exported segment to tracking CSV"""
    tracking_csv = get_tracking_csv_path()

    # Create file with headers if doesn't exist
    if not tracking_csv.exists():
        tracking_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['csv_path', 'segment_index', 'segment_time', 'class', 'exported_path', 'export_date'])

    # Append new export
    with open(tracking_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            csv_path,
            segment_index,
            segment_time,
            class_name,
            exported_path,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

def generate_export_filename(song_name: str, timestamp: str, class_name: str, index: int) -> str:
    """Generate filename for exported WAV"""
    # Convert timestamp HH:MM:SS to HH-MM-SS for filename safety
    safe_timestamp = timestamp.replace(':', '-')
    return f"{song_name}_{safe_timestamp}_{class_name}_{index:03d}.wav"

def delete_export(csv_path: str, segment_index: int) -> bool:
    """Delete exported segment - remove from tracking CSV and delete WAV file"""
    tracking_csv = get_tracking_csv_path()
    if not tracking_csv.exists():
        return False

    # Find the export entry
    exported_path = None
    rows_to_keep = []

    with open(tracking_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['csv_path'] == csv_path and int(row['segment_index']) == segment_index:
                exported_path = row['exported_path']
            else:
                rows_to_keep.append(row)

    if exported_path is None:
        return False  # Not found

    # Rewrite tracking CSV without this entry
    with open(tracking_csv, 'w', newline='', encoding='utf-8') as f:
        if rows_to_keep:
            fieldnames = ['csv_path', 'segment_index', 'segment_time', 'class', 'exported_path', 'export_date']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)

    # Delete WAV file
    wav_file = Path(exported_path)
    if wav_file.exists():
        wav_file.unlink()

    return True

@router.post("/training-data", response_model=ExportResponse)
async def export_training_data(request: ExportRequest):
    """
    Export selected segments to TRAINING DATA/DATA/{CLASS}/ folders
    """
    if not request.segments:
        raise HTTPException(status_code=400, detail="No segments provided")

    mp3_path = Path(request.mp3_path)
    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail=f"MP3 file not found: {mp3_path}")

    # Extract song name from path
    song_name = mp3_path.stem

    # Load full audio once (STEREO to match existing training data)
    try:
        y, sr = librosa.load(str(mp3_path), sr=44100, mono=False)  # Keep stereo, 44.1kHz for training data consistency
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load audio: {str(e)}")

    exported = []
    skipped = []
    errors = []

    for idx, segment in enumerate(request.segments):
        try:
            # Check if already exported
            if is_segment_exported(request.csv_path, segment.segment_index):
                skipped.append({
                    'segment_index': segment.segment_index,
                    'time': segment.segment_time,
                    'reason': 'Already exported'
                })
                continue

            # Calculate sample indices
            start_sample = int(segment.start * sr)
            stop_sample = int(segment.stop * sr)

            # Extract segment (handle both mono and stereo)
            if len(y.shape) == 1:
                # Mono
                segment_audio = y[start_sample:stop_sample]
            else:
                # Stereo - extract both channels
                segment_audio = y[:, start_sample:stop_sample]

            # Target folder
            target_folder = settings.TRAINING_DATA_FOLDER / segment.predicted_class
            target_folder.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            filename = generate_export_filename(song_name, segment.segment_time, segment.predicted_class, idx + 1)
            output_path = target_folder / filename

            # Save as WAV (44.1kHz, stereo/mono as loaded)
            # Transpose for soundfile if stereo (soundfile expects samples x channels)
            if len(segment_audio.shape) > 1:
                segment_audio_to_save = segment_audio.T  # Transpose from (channels, samples) to (samples, channels)
            else:
                segment_audio_to_save = segment_audio

            sf.write(str(output_path), segment_audio_to_save, sr)

            # Log export
            log_export(request.csv_path, segment.segment_index, segment.segment_time, segment.predicted_class, str(output_path))

            exported.append({
                'segment_index': segment.segment_index,
                'time': segment.segment_time,
                'class': segment.predicted_class,
                'path': str(output_path),
                'duration': f"{segment_audio.shape[-1]/sr:.2f}s"  # Use last dimension for duration (works for both mono and stereo)
            })

        except Exception as e:
            errors.append({
                'segment_index': segment.segment_index,
                'time': segment.segment_time,
                'error': str(e)
            })

    summary = {
        'total_requested': len(request.segments),
        'exported': len(exported),
        'skipped': len(skipped),
        'errors': len(errors)
    }

    return ExportResponse(
        exported=exported,
        skipped=skipped,
        errors=errors,
        summary=summary
    )

@router.get("/check-exported")
async def check_exported(csv_path: str):
    """
    Get list of segment indices that were already exported from this CSV
    """
    tracking_csv = get_tracking_csv_path()
    if not tracking_csv.exists():
        return {"exported_indices": []}

    exported_indices = []
    with open(tracking_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['csv_path'] == csv_path:
                exported_indices.append(int(row['segment_index']))

    return {"exported_indices": exported_indices}

@router.get("/all-exported-csvs")
async def get_all_exported_csvs():
    """
    Get list of all CSV paths that have exported segments
    """
    tracking_csv = get_tracking_csv_path()
    if not tracking_csv.exists():
        return {"csv_paths": []}

    csv_paths = set()
    with open(tracking_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_paths.add(row['csv_path'])

    return {"csv_paths": list(csv_paths)}

@router.delete("/segment")
async def delete_exported_segment(csv_path: str, segment_index: int):
    """
    Delete an exported segment - removes from tracking and deletes WAV file

    Args:
        csv_path: Path to the CSV file
        segment_index: Index of the segment in the CSV

    Returns:
        Success status and message
    """
    success = delete_export(csv_path, segment_index)

    if not success:
        raise HTTPException(status_code=404, detail="Exported segment not found")

    return {
        "success": True,
        "message": f"Segment {segment_index} export deleted successfully"
    }

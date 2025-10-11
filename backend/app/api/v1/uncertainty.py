"""
Uncertainty Review API - Active Learning for Training Data Generation
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import pandas as pd
import csv
from datetime import datetime
from app.config import settings
import librosa
import soundfile as sf
import numpy as np

router = APIRouter(prefix="/uncertainty", tags=["uncertainty"])


def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS to seconds"""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


class UncertainSegment(BaseModel):
    csv_path: str
    mp3_path: str
    segment_index: int
    segment_time: str
    predicted_class: str
    confidence: float


class ExportRangeRequest(BaseModel):
    csv_path: str
    mp3_path: str
    segment_index: int
    start_time: str  # HH:MM:SS
    end_time: str    # HH:MM:SS
    user_label: str  # Verified class by user


def get_reviewed_segments_path() -> Path:
    """Get path to reviewed segments tracking CSV"""
    return settings.TRAINING_DATA_FOLDER / "reviewed_segments.csv"


def is_segment_reviewed(csv_path: str, segment_index: int) -> bool:
    """Check if segment was already reviewed"""
    tracking_csv = get_reviewed_segments_path()
    if not tracking_csv.exists():
        return False

    with open(tracking_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['csv_path'] == csv_path and int(row['segment_index']) == segment_index:
                return True
    return False


def mark_segment_reviewed(csv_path: str, segment_index: int, user_label: str, exported_path: str):
    """Log that user reviewed this segment"""
    tracking_csv = get_reviewed_segments_path()

    # Create file with headers if doesn't exist
    if not tracking_csv.exists():
        tracking_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['csv_path', 'segment_index', 'reviewed_date', 'user_label', 'exported_path'])

    # Append new review
    with open(tracking_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            csv_path,
            segment_index,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_label,
            exported_path
        ])


def mark_segment_skipped(csv_path: str, segment_index: int):
    """Mark segment as skipped (not exported, just ignored)"""
    tracking_csv = get_reviewed_segments_path()

    # Create file with headers if doesn't exist
    if not tracking_csv.exists():
        tracking_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['csv_path', 'segment_index', 'reviewed_date', 'user_label', 'exported_path'])

    # Append as skipped
    with open(tracking_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            csv_path,
            segment_index,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'SKIPPED',
            ''
        ])


def derive_mp3_path_from_csv(csv_path: Path) -> Path:
    """
    Derive MP3 path from CSV filename

    Example:
    predictions_SONG042_2025-09-27_14-30.csv
    → SORTED/2025/09/27/SONG042.MP3
    """
    csv_name = csv_path.stem  # predictions_SONG042_2025-09-27_14-30
    parts = csv_name.split('_')

    if len(parts) < 4:
        # Old format: predictions_SONG042_2025-09-27
        song_name = parts[1]  # SONG042
        date_str = parts[2]   # 2025-09-27
    else:
        # New format with time: predictions_SONG042_2025-09-27_14-30
        song_name = parts[1]  # SONG042
        date_str = parts[2]   # 2025-09-27

    # Parse date
    year, month, day = date_str.split('-')

    # Construct MP3 path
    mp3_path = settings.SORTED_FOLDER / year / month / day / f"{song_name}.MP3"

    return mp3_path


@router.get("/segments")
async def get_uncertain_segments(
    min_confidence: float = Query(0.0, description="Minimum confidence (0.0-1.0)"),
    max_confidence: float = Query(0.7, description="Maximum confidence (0.0-1.0)"),
    category: str = Query(None, description="Filter by category (APPLAUSE, MUSIC, PUBLIC, SPEECH, TUNING)"),
    limit: int = Query(20, description="Maximum number of segments to return"),
    skip_reviewed: bool = Query(True, description="Skip already reviewed segments")
):
    """
    Find segments by confidence range and/or category

    Returns segments sorted by confidence (lowest first)
    Only returns segments from active model version (skips edited CSVs)

    Examples:
    - Uncertain TUNING: min_confidence=0, max_confidence=0.7, category=TUNING
    - All TUNING: min_confidence=0, max_confidence=1.0, category=TUNING
    - High confidence MUSIC: min_confidence=0.9, max_confidence=1.0, category=MUSIC
    """
    from app.services.model_registry import get_active_model_id, is_csv_edited

    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"
    if not results_folder.exists():
        return {"segments": [], "total_found": 0}

    # Get active model version
    active_model_id = get_active_model_id()

    uncertain = []

    for csv_file in results_folder.glob("predictions_*.csv"):
        try:
            # Skip manually edited CSVs
            if is_csv_edited(str(csv_file)):
                continue

            # Read CSV
            df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)

            # Check if CSV has confidence column (new format)
            if 'confidence' not in df.columns:
                continue  # Skip old CSVs without confidence

            # Check if CSV has model_version column
            if 'model_version' not in df.columns:
                continue  # Skip old CSVs without model_version tracking

            # Skip if CSV was generated by different model version
            csv_model_version = df['model_version'].iloc[0]
            if csv_model_version != active_model_id:
                continue

            # Normalize column names
            time_col = None
            class_col = None
            conf_col = None

            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['time', 'segment_time']:
                    time_col = col
                elif col_lower in ['predicted_class', 'class']:
                    class_col = col
                elif col_lower == 'confidence':
                    conf_col = col

            if not all([time_col, class_col, conf_col]):
                continue

            # Filter by confidence range
            filtered = df[(df[conf_col] >= min_confidence) & (df[conf_col] <= max_confidence)]

            # Filter by category if specified
            if category:
                filtered = filtered[filtered[class_col] == category]

            for idx, row in filtered.iterrows():
                # Skip if already reviewed
                if skip_reviewed and is_segment_reviewed(str(csv_file), int(idx)):
                    continue

                # Derive MP3 path
                mp3_path = derive_mp3_path_from_csv(csv_file)

                if not mp3_path.exists():
                    continue

                uncertain.append({
                    "csv_path": str(csv_file),
                    "mp3_path": str(mp3_path),
                    "segment_index": int(idx),
                    "segment_time": row[time_col],
                    "predicted_class": row[class_col],
                    "confidence": float(row[conf_col])
                })

                if len(uncertain) >= limit:
                    break

            if len(uncertain) >= limit:
                break

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    # Sort by confidence (lowest first - most uncertain)
    uncertain.sort(key=lambda x: x['confidence'])

    # Randomize order to avoid always starting with same files
    import random
    random.shuffle(uncertain)

    return {
        "segments": uncertain,
        "total_found": len(uncertain),
        "active_model": active_model_id
    }


@router.get("/stats")
async def get_uncertainty_stats():
    """
    Get statistics about uncertain segments and reviews
    Shows breakdown per model version
    """
    from app.services.model_registry import get_active_model_id, is_csv_edited

    active_model_id = get_active_model_id()
    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"

    # Stats by model version
    stats_by_model = {}
    edited_count = 0

    if results_folder.exists():
        for csv_file in results_folder.glob("predictions_*.csv"):
            try:
                # Check if manually edited
                if is_csv_edited(str(csv_file)):
                    edited_count += 1
                    continue

                # Read CSV header
                df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=1)

                # Skip if no confidence column
                if 'confidence' not in df.columns:
                    continue

                # Get model version - read full CSV once
                full_df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)

                if 'model_version' not in full_df.columns:
                    model_version = "unknown"
                else:
                    model_version = full_df['model_version'].iloc[0]

                # Initialize stats for this model
                if model_version not in stats_by_model:
                    stats_by_model[model_version] = {
                        "csvs": 0,
                        "uncertain_segments": 0
                    }

                stats_by_model[model_version]["csvs"] += 1

                # Count uncertain segments (confidence < 0.7)
                conf_col = None
                for col in full_df.columns:
                    if col.lower() == 'confidence':
                        conf_col = col
                        break

                if conf_col:
                    uncertain_count = len(full_df[full_df[conf_col] < 0.7])
                    stats_by_model[model_version]["uncertain_segments"] += uncertain_count

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

    # Count reviewed segments
    tracking_csv = get_reviewed_segments_path()
    total_reviewed = 0
    if tracking_csv.exists():
        with open(tracking_csv, 'r', encoding='utf-8') as f:
            total_reviewed = sum(1 for _ in csv.DictReader(f))

    # Get active model stats
    active_stats = stats_by_model.get(active_model_id, {"csvs": 0, "uncertain_segments": 0})

    return {
        "active_model": active_model_id,
        "active_model_stats": {
            "csvs": active_stats["csvs"],
            "uncertain_segments": active_stats["uncertain_segments"],
            "reviewed": total_reviewed,
            "remaining": active_stats["uncertain_segments"] - total_reviewed
        },
        "all_models": stats_by_model,
        "edited_csvs_count": edited_count
    }


@router.post("/export-range")
async def export_user_selected_range(request: ExportRangeRequest):
    """
    Export user-selected time range to training data

    User manually selects start and end time on waveform player,
    then this exports exactly that range as WAV to training data.
    """
    mp3_path = Path(request.mp3_path)
    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail=f"MP3 file not found: {mp3_path}")

    # Convert times to seconds
    start_sec = time_to_seconds(request.start_time)
    end_sec = time_to_seconds(request.end_time)

    if end_sec <= start_sec:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    duration_sec = end_sec - start_sec

    # Load audio (STEREO for training data consistency)
    try:
        y, sr = librosa.load(str(mp3_path), sr=44100, mono=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load audio: {str(e)}")

    # Calculate sample indices
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    # Extract segment (handle both mono and stereo)
    if len(y.shape) == 1:
        # Mono
        audio_segment = y[start_sample:end_sample]
    else:
        # Stereo - extract both channels
        audio_segment = y[:, start_sample:end_sample]

    # Target folder (TRAINING_DATA_FOLDER already points to .../DATA)
    target_folder = settings.TRAINING_DATA_FOLDER / request.user_label
    target_folder.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    song_name = mp3_path.stem
    start_time_safe = request.start_time.replace(':', '-')
    end_time_safe = request.end_time.replace(':', '-')
    filename = f"{song_name}_{start_time_safe}_to_{end_time_safe}_{request.user_label}_verified.wav"
    output_path = target_folder / filename

    # Save as WAV (44.1kHz, stereo/mono as loaded)
    # Transpose for soundfile if stereo (soundfile expects samples x channels)
    if len(audio_segment.shape) > 1:
        audio_segment_to_save = audio_segment.T  # Transpose from (channels, samples) to (samples, channels)
    else:
        audio_segment_to_save = audio_segment

    sf.write(str(output_path), audio_segment_to_save, sr)

    # Mark as reviewed with actual segment index
    mark_segment_reviewed(request.csv_path, request.segment_index, request.user_label, str(output_path))

    return {
        "success": True,
        "exported_path": str(output_path),
        "duration_sec": duration_sec,
        "start_time": request.start_time,
        "end_time": request.end_time,
        "user_label": request.user_label
    }


class SkipFileRequest(BaseModel):
    csv_path: str


class UndoExportRequest(BaseModel):
    csv_path: str
    segment_index: int


@router.post("/skip-file")
async def skip_entire_file(request: SkipFileRequest):
    """
    Mark all uncertain segments from a specific CSV file as skipped

    This prevents them from appearing in future uncertainty reviews
    """
    csv_file = Path(request.csv_path)
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_file}")

    # Read CSV to find all uncertain segments
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")

    # Check if CSV has confidence column
    if 'confidence' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV does not have confidence column")

    # Find column names
    conf_col = None
    for col in df.columns:
        if col.lower() == 'confidence':
            conf_col = col
            break

    if not conf_col:
        raise HTTPException(status_code=400, detail="Confidence column not found")

    # Find all uncertain segments (confidence < 0.7) that aren't already reviewed
    threshold = 0.7
    low_conf = df[df[conf_col] < threshold]

    skipped_count = 0
    for idx, row in low_conf.iterrows():
        # Skip if already reviewed
        if is_segment_reviewed(str(csv_file), int(idx)):
            continue

        # Mark as skipped
        mark_segment_skipped(str(csv_file), int(idx))
        skipped_count += 1

    return {
        "success": True,
        "csv_path": str(csv_file),
        "segments_skipped": skipped_count
    }


@router.post("/undo-export")
async def undo_last_export(request: UndoExportRequest):
    """
    Undo export - delete WAV file and remove from reviewed_segments.csv
    """
    tracking_csv = get_reviewed_segments_path()
    if not tracking_csv.exists():
        raise HTTPException(status_code=404, detail="No reviewed segments found")

    # Read all reviewed segments
    rows = []
    deleted_row = None
    with open(tracking_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['csv_path'] == request.csv_path and
                int(row['segment_index']) == request.segment_index):
                deleted_row = row
            else:
                rows.append(row)

    if not deleted_row:
        raise HTTPException(status_code=404, detail="Export not found in tracking CSV")

    # Delete WAV file if exists and path is not empty
    wav_path = deleted_row.get('exported_path', '')
    if wav_path and wav_path != '':
        wav_file = Path(wav_path)
        if wav_file.exists():
            wav_file.unlink()
            print(f"[UNDO] Deleted WAV: {wav_file}")
        else:
            print(f"[UNDO] WAV not found (maybe SKIPPED): {wav_file}")

    # Rewrite CSV without deleted row
    with open(tracking_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['csv_path', 'segment_index', 'reviewed_date', 'user_label', 'exported_path'])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "success": True,
        "deleted_wav": wav_path,
        "csv_path": request.csv_path,
        "segment_index": request.segment_index
    }

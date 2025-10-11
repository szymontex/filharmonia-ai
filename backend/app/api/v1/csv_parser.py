"""
CSV Parser API - parse predictions CSV into tracks
"""
import pandas as pd
from fastapi import APIRouter, Query, Body
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import os

router = APIRouter(prefix="/csv", tags=["csv"])

class Track(BaseModel):
    id: str
    selected: bool
    name: str
    predicted_class: str  # Renamed to avoid "class" keyword conflict
    start: str
    stop: str
    duration: str

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "track_0",
                "selected": False,
                "name": "",
                "predicted_class": "MUSIC",
                "start": "00:00:05",
                "stop": "00:12:30",
                "duration": "12'25\""
            }
        }

class ParseResponse(BaseModel):
    tracks: List[Track]
    total_segments: int

class SaveRequest(BaseModel):
    path: str
    tracks: List[Track]

class AutosaveCheckResponse(BaseModel):
    has_autosave: bool
    autosave_newer: bool
    autosave_path: Optional[str] = None
    autosave_time: Optional[str] = None
    original_time: Optional[str] = None

def get_duration(start: str, stop: str) -> str:
    """Calculate duration as M'S" format"""
    try:
        # Parse HH:MM:SS
        start_parts = list(map(int, start.split(':')))
        stop_parts = list(map(int, stop.split(':')))

        start_sec = start_parts[0] * 3600 + start_parts[1] * 60 + start_parts[2]
        stop_sec = stop_parts[0] * 3600 + stop_parts[1] * 60 + stop_parts[2]

        diff_sec = stop_sec - start_sec
        minutes = diff_sec // 60
        seconds = diff_sec % 60

        return f"{minutes}'{seconds}\""
    except:
        return "0'0\""

def extract_tracks(df: pd.DataFrame, threshold: int = 5) -> List[Track]:
    """
    Extract tracks from CSV with threshold filtering
    Port from main2.py logic
    Supports both old format (segment_time,predicted_class) and new format (time,predicted_class,name)
    """
    tracks = []

    if len(df) == 0:
        return tracks

    # Normalize column names - handle both 'time' and 'segment_time'
    time_col = None
    class_col = None
    name_col = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['time', 'segment_time']:
            time_col = col
        elif col_lower in ['predicted_class', 'class']:
            class_col = col
        elif col_lower == 'name':
            name_col = col

    if time_col is None or class_col is None:
        raise ValueError(f"CSV must have time and predicted_class columns. Found: {df.columns.tolist()}")

    has_name_column = name_col is not None

    current_class = df.iloc[0][class_col]
    start = df.iloc[0][time_col]
    last_valid_class = current_class
    track_counter = 0

    # Handle NaN values in name column
    if has_name_column and name_col is not None:
        name_value = df.iloc[0][name_col]
        current_name = "" if pd.isna(name_value) else str(name_value)
    else:
        current_name = ""
    last_valid_name = current_name

    index = 1
    while index < len(df):
        if df.iloc[index][class_col] != current_class:
            segment_start = index

            # Count consecutive segments of new class
            while index < len(df) and df.iloc[index][class_col] == df.iloc[segment_start][class_col]:
                index += 1

            segment_length = index - segment_start

            # Threshold filtering: ignore short segments
            if segment_length < threshold:
                current_class = last_valid_class
            else:
                # Create track for previous segment
                stop = df.iloc[segment_start - 1][time_col]
                duration = get_duration(start, stop)

                tracks.append(Track(
                    id=f"track_{track_counter}",
                    selected=False,
                    name=current_name,
                    predicted_class=current_class,
                    start=start,
                    stop=stop,
                    duration=duration
                ))

                track_counter += 1
                start = df.iloc[segment_start][time_col]
                current_class = df.iloc[segment_start][class_col]

                # Handle NaN values in name column
                if has_name_column and name_col is not None:
                    name_value = df.iloc[segment_start][name_col]
                    current_name = "" if pd.isna(name_value) else str(name_value)
                else:
                    current_name = ""

                last_valid_class = current_class
                last_valid_name = current_name
        else:
            index += 1

    # Add final track
    stop = df.iloc[-1][time_col]
    duration = get_duration(start, stop)
    tracks.append(Track(
        id=f"track_{track_counter}",
        selected=False,
        name=current_name,
        predicted_class=current_class,
        start=start,
        stop=stop,
        duration=duration
    ))

    return tracks

@router.get("/parse", response_model=ParseResponse)
async def parse_csv(
    path: str = Query(..., description="Path to CSV file"),
    threshold: int = Query(5, description="Minimum segment length to consider as new track")
):
    """
    Parse predictions CSV and extract tracks with threshold filtering
    """
    csv_path = Path(path)

    if not csv_path.exists():
        return ParseResponse(tracks=[], total_segments=0)

    # Read CSV with proper encoding - handle quoted fields (for names with commas)
    df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)  # QUOTE_ALL=1, QUOTE_MINIMAL=0
    df.columns = [col.strip() for col in df.columns]  # Strip whitespace

    # Extract tracks
    tracks = extract_tracks(df, threshold=threshold)

    return ParseResponse(
        tracks=tracks,
        total_segments=len(df)
    )

def get_autosave_path(original_path: str) -> str:
    """Get autosave path for a given CSV path"""
    p = Path(original_path)
    return str(p.parent / f"{p.stem}_autosave{p.suffix}")

def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to total seconds"""
    parts = list(map(int, time_str.split(':')))
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def seconds_to_time(seconds: int) -> str:
    """Convert total seconds to HH:MM:SS"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def escape_csv_field(field: str) -> str:
    """Escape CSV field - quote if contains comma, quote, or newline"""
    if ',' in field or '"' in field or '\n' in field:
        # Escape quotes by doubling them, then wrap in quotes
        return f'"{field.replace(chr(34), chr(34)+chr(34))}"'
    return field

def tracks_to_csv_content(tracks: List[Track]) -> str:
    """Convert tracks back to CSV format - generate one line per second with track names"""
    if not tracks:
        return "time,predicted_class,name\n"

    lines = ["time,predicted_class,name\n"]

    current_sec = 0  # Start from 00:00:00

    for i, track in enumerate(tracks):
        start_sec = time_to_seconds(track.start)
        stop_sec = time_to_seconds(track.stop)

        # Fill gap with previous track's class if there's a gap
        if i > 0 and current_sec < start_sec:
            prev_track = tracks[i-1]
            for sec in range(current_sec, start_sec):
                time_str = seconds_to_time(sec)
                escaped_name = escape_csv_field(prev_track.name)
                lines.append(f"{time_str},{prev_track.predicted_class},{escaped_name}\n")

        # Generate lines for this track
        for sec in range(start_sec, stop_sec + 1):
            time_str = seconds_to_time(sec)
            escaped_name = escape_csv_field(track.name)
            lines.append(f"{time_str},{track.predicted_class},{escaped_name}\n")

        current_sec = stop_sec + 1

    return "".join(lines)

@router.get("/check-autosave", response_model=AutosaveCheckResponse)
async def check_autosave(path: str = Query(..., description="Path to original CSV file")):
    """
    Check if autosave exists and is newer than original
    """
    original_path = Path(path)
    autosave_path = Path(get_autosave_path(path))

    if not autosave_path.exists():
        return AutosaveCheckResponse(
            has_autosave=False,
            autosave_newer=False
        )

    # Compare modification times
    original_mtime = original_path.stat().st_mtime if original_path.exists() else 0
    autosave_mtime = autosave_path.stat().st_mtime

    # Compare file contents - if identical, no need to notify user
    if original_path.exists():
        with open(original_path, 'r', encoding='utf-8') as f1:
            original_content = f1.read()
        with open(autosave_path, 'r', encoding='utf-8') as f2:
            autosave_content = f2.read()

        if original_content == autosave_content:
            # Files are identical - silently delete autosave and report no autosave
            autosave_path.unlink()
            return AutosaveCheckResponse(
                has_autosave=False,
                autosave_newer=False
            )

    return AutosaveCheckResponse(
        has_autosave=True,
        autosave_newer=autosave_mtime > original_mtime,
        autosave_path=str(autosave_path),
        autosave_time=datetime.fromtimestamp(autosave_mtime).isoformat(),
        original_time=datetime.fromtimestamp(original_mtime).isoformat() if original_path.exists() else None
    )

@router.post("/autosave")
async def autosave_csv(request: SaveRequest = Body(...)):
    """
    Save tracks to autosave file
    """
    autosave_path = get_autosave_path(request.path)
    csv_content = tracks_to_csv_content(request.tracks)

    Path(autosave_path).write_text(csv_content, encoding='utf-8')

    return {"success": True, "autosave_path": autosave_path}

def mark_csv_as_edited(csv_path: str):
    """Mark CSV as manually edited by logging to edited_csvs.txt"""
    from app.config import settings

    edited_log = settings.FILHARMONIA_BASE / ".claude" / "edited_csvs.txt"
    edited_log.parent.mkdir(parents=True, exist_ok=True)

    # Read existing edited files
    edited_files = set()
    if edited_log.exists():
        with open(edited_log, 'r', encoding='utf-8') as f:
            edited_files = set(line.strip() for line in f if line.strip())

    # Add current file if not already there
    if csv_path not in edited_files:
        with open(edited_log, 'a', encoding='utf-8') as f:
            f.write(f"{csv_path}\n")

@router.post("/save")
async def save_csv(request: SaveRequest = Body(...)):
    """
    Save tracks to original CSV file (overwrite)
    """
    csv_content = tracks_to_csv_content(request.tracks)

    Path(request.path).write_text(csv_content, encoding='utf-8')

    # Remove autosave file if it exists
    autosave_path = Path(get_autosave_path(request.path))
    if autosave_path.exists():
        autosave_path.unlink()

    # Mark as edited
    mark_csv_as_edited(request.path)

    return {"success": True, "path": request.path}

@router.get("/edited-list")
async def get_edited_csvs():
    """
    Get list of manually edited CSV files
    """
    from app.config import settings

    edited_log = settings.FILHARMONIA_BASE / ".claude" / "edited_csvs.txt"

    if not edited_log.exists():
        return {"edited_files": []}

    with open(edited_log, 'r', encoding='utf-8') as f:
        edited_files = [line.strip() for line in f if line.strip()]

    return {"edited_files": edited_files}

@router.delete("/discard-autosave")
async def discard_autosave(path: str = Query(..., description="Path to original CSV file")):
    """
    Delete autosave file
    """
    autosave_path = Path(get_autosave_path(path))

    if autosave_path.exists():
        autosave_path.unlink()
        return {"success": True, "message": "Autosave deleted"}

    return {"success": False, "message": "No autosave file found"}

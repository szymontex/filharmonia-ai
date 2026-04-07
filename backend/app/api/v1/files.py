"""
File Browser API
"""
import logging
import re
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import List
from pydantic import BaseModel
import eyed3
from app.config import settings
from app.core.security import validate_path_or_raise_http

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for performance (PERF-03)
# Simple date extraction: matches _YYYY-MM-DD in filenames
DATE_PATTERN = re.compile(r'_(\d{4})-(\d{2})-(\d{2})')
# Full predictions filename: predictions_{songName}_{YYYY-MM-DD}[_{HH-MM}].csv
PREDICTIONS_PATTERN = re.compile(r'predictions_(.+?)_(\d{4})-(\d{2})-(\d{2})(?:_\d{2}-\d{2})?\.csv')

router = APIRouter(prefix="/files", tags=["files"])

class FileInfo(BaseModel):
    path: str
    name: str
    size: int
    date: str  # From folder structure YYYY/MM/DD
    type: str  # "mp3" or "csv"
    time: str = ""  # HH:MM from ID3 tag (optional)

# Cache for sorted files listing (avoid repeated network scans)
_sorted_files_cache: dict = {"files": None, "timestamp": 0, "loading": False}
_SORTED_CACHE_TTL = 1800  # 30 minutes - network share doesn't change often

def _scan_sorted_files_sync() -> list:
    """Scan sorted folder using scandir - fastest option for network shares."""
    import os
    files = []
    sorted_root = str(settings.SORTED_FOLDER)
    skip_dirs = {'ANALYSIS_RESULTS', '.waveform_cache'}

    def scan_dir(dirpath, depth=0, year="", month="", day=""):
        """Recursive scandir - avoids os.walk overhead and extra stat calls."""
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name in skip_dirs or entry.name.startswith('.'):
                            continue
                        # Track year/month/day from folder depth
                        if depth == 0 and entry.name.isdigit():
                            scan_dir(entry.path, 1, year=entry.name)
                        elif depth == 1 and entry.name.isdigit():
                            scan_dir(entry.path, 2, year=year, month=entry.name)
                        elif depth == 2 and entry.name.isdigit():
                            scan_dir(entry.path, 3, year=year, month=month, day=entry.name)
                        else:
                            scan_dir(entry.path, depth, year=year, month=month, day=day)
                    elif entry.name.upper().endswith('.MP3'):
                        date_str = "Unknown"
                        if year and month and day:
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        try:
                            fsize = entry.stat().st_size
                        except OSError:
                            fsize = 0
                        files.append(FileInfo(
                            path=entry.path,
                            name=entry.name,
                            size=fsize,
                            date=date_str,
                            type="mp3",
                            time=""
                        ))
        except PermissionError:
            pass

    t0 = time.time()
    scan_dir(sorted_root)
    files.sort(key=lambda x: x.date, reverse=True)
    elapsed = time.time() - t0
    logger.info("Sorted files scan: %d MP3s in %.1fs", len(files), elapsed)
    return files

def _warm_cache_background():
    """Warm the cache in a background thread at startup."""
    import threading
    def _do():
        try:
            _sorted_files_cache["loading"] = True
            files = _scan_sorted_files_sync()
            _sorted_files_cache["files"] = files
            _sorted_files_cache["timestamp"] = time.time()
        finally:
            _sorted_files_cache["loading"] = False
    t = threading.Thread(target=_do, daemon=True)
    t.start()

# Start warming cache immediately on module load
_warm_cache_background()

@router.get("/sorted", response_model=List[FileInfo])
async def list_sorted_files():
    """
    List all MP3 files from SORTED/ folder.
    Cache is warmed at startup. Returns instantly after first scan.
    """
    import asyncio

    now = time.time()
    if _sorted_files_cache["files"] is not None and (now - _sorted_files_cache["timestamp"]) < _SORTED_CACHE_TTL:
        return _sorted_files_cache["files"]

    # Cache expired or not ready yet - scan in thread
    if _sorted_files_cache["loading"]:
        # Already scanning in background, return what we have (or empty)
        return _sorted_files_cache["files"] or []

    loop = asyncio.get_event_loop()
    files = await loop.run_in_executor(None, _scan_sorted_files_sync)

    _sorted_files_cache["files"] = files
    _sorted_files_cache["timestamp"] = time.time()

    return files

def invalidate_sorted_cache():
    """Clear the sorted files cache so the next request rescans."""
    _sorted_files_cache["files"] = None
    _sorted_files_cache["timestamp"] = 0

@router.post("/sorted/refresh")
async def refresh_sorted_cache():
    """Force refresh the sorted files cache"""
    invalidate_sorted_cache()
    return {"message": "Cache cleared, next request will rescan"}

@router.get("/analysis-results", response_model=List[FileInfo])
async def list_analysis_results():
    """
    List all CSV files from ANALYSIS_RESULTS/
    """
    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"

    if not results_folder.exists():
        return []

    files = []

    for csv_file in results_folder.glob("*.csv"):
        # Skip temporary files being written during analysis
        if csv_file.suffix == '.tmp' or csv_file.name.endswith('.csv.tmp'):
            continue

        # Extract recording date from filename: predictions_SONG042_2025-09-27.csv
        date_str = "Unknown"
        match = DATE_PATTERN.search(csv_file.name)
        if match:
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

        files.append(FileInfo(
            path=str(csv_file),
            name=csv_file.name,
            size=csv_file.stat().st_size,
            date=date_str,  # Recording date from filename
            type="csv"
        ))

    # Sort by recording date (newest first)
    files.sort(key=lambda x: x.date, reverse=True)

    return files

@router.delete("/delete-csv")
async def delete_csv(path: str = Query(..., description="Path to CSV file to delete")):
    """
    Delete a CSV file and its autosave if exists
    """
    # Validate path is within SORTED_FOLDER (includes ANALYSIS_RESULTS)
    csv_path = validate_path_or_raise_http(path, settings.SORTED_FOLDER)

    # Delete main file
    csv_path.unlink()

    # Delete autosave if exists
    autosave_path = Path(str(csv_path).replace('.csv', '_autosave.csv'))
    if autosave_path.exists():
        autosave_path.unlink()

    return {"success": True, "message": "CSV deleted successfully"}


class Mp3PathResponse(BaseModel):
    mp3_path: str
    recording_date: str
    exists: bool


@router.get("/mp3-for-csv", response_model=Mp3PathResponse)
async def get_mp3_for_csv(csv_path: str = Query(..., description="Path to CSV file")):
    """
    Resolve MP3 file path from CSV prediction file path.

    CSV filename format: predictions_{songName}_{YYYY-MM-DD}[_{HH-MM}].csv
    Returns: MP3 path in SORTED_FOLDER/{year}/{month}/{day}/{songName}.MP3
    """
    # Remove _autosave suffix if present
    clean_path = csv_path.replace('_autosave', '')

    # Parse CSV filename: predictions_{songName}_{YYYY-MM-DD}[_{HH-MM}].csv
    match = PREDICTIONS_PATTERN.search(clean_path)

    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV filename. Expected format: predictions_{{songName}}_{{YYYY-MM-DD}}.csv"
        )

    song_name, year, month, day = match.groups()

    # Build MP3 path: SORTED_FOLDER/year/month/day/songName.MP3
    mp3_path = settings.SORTED_FOLDER / year / month / day / f"{song_name}.MP3"

    # Check if exists (also try lowercase .mp3)
    exists = mp3_path.exists()
    if not exists:
        mp3_path_lower = mp3_path.with_suffix('.mp3')
        if mp3_path_lower.exists():
            mp3_path = mp3_path_lower
            exists = True

    return Mp3PathResponse(
        mp3_path=str(mp3_path),
        recording_date=f"{year}-{month}-{day}",
        exists=exists
    )


class CsvMetadataResponse(BaseModel):
    csv_path: str
    mp3_path: str
    recording_date: str
    song_name: str
    mp3_exists: bool
    csv_exists: bool


@router.get("/csv-metadata", response_model=CsvMetadataResponse)
async def get_csv_metadata(csv_path: str = Query(..., description="Path to CSV file")):
    """
    Get full metadata for a CSV file including resolved MP3 path.

    Useful for frontend to get all needed info in a single call.
    """
    csv_path_obj = Path(csv_path)
    clean_name = csv_path_obj.name.replace('_autosave', '')

    # Parse filename
    match = PREDICTIONS_PATTERN.search(clean_name)

    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV filename format"
        )

    song_name, year, month, day = match.groups()

    # Build MP3 path
    mp3_path = settings.SORTED_FOLDER / year / month / day / f"{song_name}.MP3"
    mp3_exists = mp3_path.exists()

    if not mp3_exists:
        mp3_path_lower = mp3_path.with_suffix('.mp3')
        if mp3_path_lower.exists():
            mp3_path = mp3_path_lower
            mp3_exists = True

    return CsvMetadataResponse(
        csv_path=csv_path,
        mp3_path=str(mp3_path),
        recording_date=f"{year}-{month}-{day}",
        song_name=song_name,
        mp3_exists=mp3_exists,
        csv_exists=csv_path_obj.exists()
    )

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

router = APIRouter(prefix="/files", tags=["files"])

class FileInfo(BaseModel):
    path: str
    name: str
    size: int
    date: str  # From folder structure YYYY/MM/DD
    type: str  # "mp3" or "csv"
    time: str = ""  # HH:MM from ID3 tag (optional)

@router.get("/sorted", response_model=List[FileInfo])
async def list_sorted_files():
    """
    List all MP3 files from SORTED/ folder
    """
    files = []

    # Search both .mp3 and .MP3 extensions
    mp3_files = list(settings.SORTED_FOLDER.rglob("*.mp3")) + list(settings.SORTED_FOLDER.rglob("*.MP3"))

    for mp3_file in mp3_files:
        # Parse date from folder structure: SORTED/YYYY/MM/DD/file.mp3
        rel_path = mp3_file.relative_to(settings.SORTED_FOLDER)
        parts = rel_path.parts

        date_str = "Unknown"
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            date_str = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"

        # Extract time from ID3 tag
        time_str = ""
        try:
            audiofile = eyed3.load(str(mp3_file))
            if audiofile and audiofile.tag and audiofile.tag.title:
                record_date = datetime.strptime(audiofile.tag.title, 'Untitled %m/%d/%Y %H:%M:%S')
                time_str = f"{record_date.hour:02d}:{record_date.minute:02d}"
        except Exception as e:
            logger.debug(f"Could not read ID3 tag from {mp3_file}: {e}")

        files.append(FileInfo(
            path=str(mp3_file),
            name=mp3_file.name,
            size=mp3_file.stat().st_size,
            date=date_str,
            type="mp3",
            time=time_str
        ))

    # Sort by date (newest first)
    files.sort(key=lambda x: x.date, reverse=True)

    return files

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
        match = re.search(r'_(\d{4})-(\d{2})-(\d{2})', csv_file.name)
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
    match = re.search(r'predictions_(.+?)_(\d{4})-(\d{2})-(\d{2})(?:_\d{2}-\d{2})?\.csv', clean_path)

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
    match = re.search(r'predictions_(.+?)_(\d{4})-(\d{2})-(\d{2})(?:_\d{2}-\d{2})?\.csv', clean_name)

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

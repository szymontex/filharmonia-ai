"""
File Browser API
"""
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import List
from pydantic import BaseModel
import eyed3
from app.config import settings

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

    for mp3_file in settings.SORTED_FOLDER.rglob("*.mp3"):
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
        except:
            pass

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
        import re
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
    csv_path = Path(path)

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Delete main file
    csv_path.unlink()

    # Delete autosave if exists
    autosave_path = Path(str(csv_path).replace('.csv', '_autosave.csv'))
    if autosave_path.exists():
        autosave_path.unlink()

    return {"success": True, "message": "CSV deleted successfully"}

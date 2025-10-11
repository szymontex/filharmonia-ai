"""
Batch Analysis API
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import threading
from app.config import settings
from app.services.analyze import get_analyze_service

router = APIRouter(prefix="/analyze", tags=["analyze"])

class BatchRequest(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    mp3_paths: Optional[List[str]] = None  # Alternative: explicit list of paths

class BatchResponse(BaseModel):
    job_id: str
    files_queued: int
    files: List[str]
    message: str

# Job tracking (in-memory for now, later: Redis/DB)
_jobs = {}

def get_unanalyzed_files(year: Optional[int] = None, month: Optional[int] = None) -> List[Path]:
    """
    Find MP3 files that don't have corresponding CSV in ANALYSIS_RESULTS
    Matches by stem + date to avoid false positives (e.g., SONG005 from 2023 vs 2025)
    """
    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"
    analyzed_files = set()  # Set of (stem, date) tuples

    if results_folder.exists():
        for csv_file in results_folder.glob("predictions_*.csv"):
            # Extract from predictions_SONG059_2025-10-03_20-18-48.csv or predictions_SONG059_2025-10-03.csv
            parts = csv_file.stem.split('_')
            if len(parts) >= 4:  # predictions, SONG059, 2025-10-03, ...
                mp3_stem = parts[1]  # SONG059
                date_str = parts[2]  # 2025-10-03
                analyzed_files.add((mp3_stem, date_str))

    # Find all MP3s matching criteria
    mp3_files = []

    if year and month:
        # Specific month
        folder = settings.SORTED_FOLDER / str(year) / f"{month:02d}"
        if folder.exists():
            for day_folder in folder.iterdir():
                if day_folder.is_dir():
                    mp3_files.extend(day_folder.glob("*.MP3"))
    elif year:
        # Whole year
        folder = settings.SORTED_FOLDER / str(year)
        if folder.exists():
            mp3_files.extend(folder.rglob("*.MP3"))
    else:
        # All files
        mp3_files.extend(settings.SORTED_FOLDER.rglob("*.MP3"))

    # Filter out already analyzed (by stem + date)
    unanalyzed = []
    for mp3_file in mp3_files:
        # Extract date from folder structure: SORTED/YYYY/MM/DD/file.MP3
        rel_path = mp3_file.relative_to(settings.SORTED_FOLDER)
        parts = rel_path.parts
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            date_str = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            if (mp3_file.stem, date_str) not in analyzed_files:
                unanalyzed.append(mp3_file)
        else:
            # Fallback for files not in YYYY/MM/DD structure
            if mp3_file.stem not in {stem for stem, _ in analyzed_files}:
                unanalyzed.append(mp3_file)

    return unanalyzed

def run_batch_analysis(job_id: str, mp3_files: List[Path]):
    """
    Background task to analyze multiple files
    """
    _jobs[job_id] = {
        "status": "running",
        "total": len(mp3_files),
        "completed": 0,
        "failed": 0,
        "current_file": None,
        "current_file_progress": 0.0,
        "results": [],
        "errors": [],
        "cancelled": False
    }

    service = get_analyze_service()

    for i, mp3_file in enumerate(mp3_files):
        # Check if cancelled
        if _jobs[job_id].get("cancelled", False):
            _jobs[job_id]["status"] = "cancelled"
            _jobs[job_id]["current_file"] = None
            return

        _jobs[job_id]["current_file"] = mp3_file.name
        _jobs[job_id]["current_file_progress"] = 0.0

        def progress_callback(segment, total_segments, file_progress):
            """Update progress for current file"""
            _jobs[job_id]["current_file_progress"] = file_progress

            # Global progress: (completed files + current file progress) / total files
            global_progress = (i + file_progress / 100) / len(mp3_files) * 100
            _jobs[job_id]["progress"] = round(global_progress, 1)

        def is_cancelled():
            """Check if job was cancelled"""
            return _jobs[job_id].get("cancelled", False)

        try:
            result = service.analyze_file(mp3_file, on_progress=progress_callback, check_cancelled=is_cancelled)
            _jobs[job_id]["results"].append({
                "mp3": str(mp3_file),
                "csv": result["csv_path"],
                "segments": result["segments_analyzed"]
            })
            _jobs[job_id]["completed"] += 1
            _jobs[job_id]["current_file_progress"] = 100.0
        except InterruptedError as e:
            # Analysis was cancelled - stop the batch
            _jobs[job_id]["errors"].append({
                "mp3": str(mp3_file),
                "error": "Cancelled by user"
            })
            _jobs[job_id]["status"] = "cancelled"
            _jobs[job_id]["current_file"] = None
            return
        except Exception as e:
            _jobs[job_id]["errors"].append({
                "mp3": str(mp3_file),
                "error": str(e)
            })
            _jobs[job_id]["failed"] += 1

    _jobs[job_id]["status"] = "completed"
    _jobs[job_id]["current_file"] = None
    _jobs[job_id]["progress"] = 100.0

@router.post("/batch", response_model=BatchResponse)
async def batch_analyze(request: BatchRequest):
    """
    Analyze multiple MP3 files in background

    Examples:
    - Analyze all unanalyzed files from May 2025: {"year": 2025, "month": 5}
    - Analyze all from 2025: {"year": 2025}
    - Analyze specific files: {"mp3_paths": ["path1", "path2"]}
    """
    if request.mp3_paths:
        # Explicit list
        mp3_files = [Path(p) for p in request.mp3_paths if Path(p).exists()]
    else:
        # Auto-discover unanalyzed
        mp3_files = get_unanalyzed_files(request.year, request.month)

    if not mp3_files:
        raise HTTPException(status_code=404, detail="No files found matching criteria")

    # Create job
    job_id = str(uuid.uuid4())

    # Start batch analysis in daemon thread (doesn't block server restart)
    thread = threading.Thread(
        target=run_batch_analysis,
        args=(job_id, mp3_files),
        daemon=True,
        name=f"BatchAnalysis-{job_id[:8]}"
    )
    thread.start()

    return BatchResponse(
        job_id=job_id,
        files_queued=len(mp3_files),
        files=[f.name for f in mp3_files],
        message=f"✓ Batch analysis started: {len(mp3_files)} files"
    )

@router.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    """
    Get status of batch analysis job
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _jobs[job_id]

@router.post("/batch/{job_id}/cancel")
async def cancel_batch_job(job_id: str):
    """
    Cancel a running batch analysis job
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Job is not running (status: {job['status']})")

    # Set cancelled flag - the background task will check this
    _jobs[job_id]["cancelled"] = True

    return {"message": "Job cancellation requested", "job_id": job_id}

@router.get("/batch")
async def list_all_jobs():
    """
    List all analysis jobs (batch and single file)
    """
    # Import single file jobs dict
    from app.api.v1.analyze import _single_jobs

    jobs_list = []

    # Add batch jobs
    for job_id, job_data in _jobs.items():
        jobs_list.append({
            "job_id": job_id,
            "status": job_data["status"],
            "total": job_data["total"],
            "completed": job_data["completed"],
            "failed": job_data["failed"],
            "progress": job_data.get("progress", 0),
            "type": "batch"
        })

    # Add single file jobs
    for job_id, job_data in _single_jobs.items():
        jobs_list.append({
            "job_id": job_id,
            "status": job_data["status"],
            "total": job_data.get("total_segments", 1),
            "completed": job_data.get("current_segment", 0),
            "failed": 0,
            "progress": job_data.get("progress", 0),
            "type": "single",
            "file": job_data.get("file", "")
        })

    # Sort: running first, then by most recent
    jobs_list.sort(key=lambda x: (x["status"] != "running", -x.get("progress", 0)))
    return jobs_list


@router.get("/outdated-csvs")
async def get_outdated_csvs():
    """
    Get list of CSVs analyzed with old models (excluding edited files)

    Returns CSVs that need re-analysis with current active model
    """
    from app.services.model_registry import get_active_model_id, is_csv_edited
    from app.api.v1.uncertainty import derive_mp3_path_from_csv
    import pandas as pd

    active_model_id = get_active_model_id()
    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"

    if not results_folder.exists():
        return {
            "outdated_csvs": [],
            "count": 0,
            "active_model": active_model_id
        }

    outdated = []

    for csv_file in results_folder.glob("predictions_*.csv"):
        try:
            # Skip manually edited files
            if is_csv_edited(str(csv_file)):
                continue

            # Read CSV header
            df = pd.read_csv(csv_file, encoding='utf-8', quoting=1, nrows=1)

            # Old CSV without model_version
            if 'model_version' not in df.columns:
                csv_model_version = "unknown"
            else:
                # Read full CSV to get model_version
                full_df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)
                csv_model_version = full_df['model_version'].iloc[0]

            # Skip if already current version
            if csv_model_version == active_model_id:
                continue

            # Derive MP3 path
            try:
                mp3_path = derive_mp3_path_from_csv(csv_file)
                if not mp3_path.exists():
                    continue
            except:
                continue

            outdated.append({
                "csv_path": str(csv_file),
                "mp3_path": str(mp3_path),
                "old_model_version": csv_model_version
            })

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    return {
        "outdated_csvs": outdated,
        "count": len(outdated),
        "active_model": active_model_id
    }

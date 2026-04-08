"""
Batch Analysis API - uses multiprocessing to avoid blocking the server
"""
import logging
import os
import tempfile
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import uuid
import json
from cachetools import TTLCache
import polars as pl
from app.config import settings
from app.services.job_registry import get_job_registry
from app.services.analysis_queue import get_analysis_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Directory for job status files (shared with analyze.py, cross-platform)
JOBS_DIR = Path(tempfile.gettempdir()) / "filharmonia_jobs"
JOBS_DIR.mkdir(exist_ok=True)

class BatchRequest(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
    mp3_paths: Optional[List[str]] = None

class BatchResponse(BaseModel):
    job_id: str
    files_queued: int
    files: List[str]
    message: str

def get_job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def read_job_status(job_id: str) -> dict:
    job_file = get_job_file(job_id)
    if job_file.exists():
        try:
            return json.loads(job_file.read_text())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse job status {job_id}: {e}")
    return None

def write_job_status(job_id: str, status: dict):
    """Atomic write using temp file + rename pattern (CRIT-13)"""
    job_file = get_job_file(job_id)

    # Create temp file in same directory (ensures same filesystem)
    fd, tmp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=f'{job_id}_',
        dir=JOBS_DIR
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(status, f)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is on disk

        # Atomic replace (works on both Unix and Windows)
        os.replace(tmp_path, job_file)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

# In-memory cache for job tracking
# TTL cache: 4 hour expiry, max 50 entries (batch jobs can be long-running)
_jobs: TTLCache = TTLCache(maxsize=50, ttl=14400)

def get_unanalyzed_files(year: Optional[int] = None, month: Optional[int] = None) -> List[Path]:
    """
    Find MP3 files that don't have corresponding CSV in ANALYSIS_RESULTS
    """
    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"
    analyzed_files = set()

    if results_folder.exists():
        for csv_file in results_folder.glob("predictions_*.csv"):
            parts = csv_file.stem.split('_')
            if len(parts) >= 4:
                mp3_stem = parts[1]
                date_str = parts[2]
                analyzed_files.add((mp3_stem, date_str))

    mp3_files = []

    if year and month:
        folder = settings.SORTED_FOLDER / str(year) / f"{month:02d}"
        if folder.exists():
            for day_folder in folder.iterdir():
                if day_folder.is_dir():
                    mp3_files.extend(day_folder.glob("*.MP3"))
    elif year:
        folder = settings.SORTED_FOLDER / str(year)
        if folder.exists():
            mp3_files.extend(folder.rglob("*.MP3"))
    else:
        mp3_files.extend(settings.SORTED_FOLDER.rglob("*.MP3"))

    unanalyzed = []
    for mp3_file in mp3_files:
        rel_path = mp3_file.relative_to(settings.SORTED_FOLDER)
        parts = rel_path.parts
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            date_str = f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
            if (mp3_file.stem, date_str) not in analyzed_files:
                unanalyzed.append(mp3_file)
        else:
            if mp3_file.stem not in {stem for stem, _ in analyzed_files}:
                unanalyzed.append(mp3_file)

    return unanalyzed

def run_batch_analysis_process(job_id: str, mp3_files_str: List[str]):
    """
    Run batch analysis in separate process.
    Writes status to JSON file for IPC.
    """
    # CRITICAL: Set CPU limits BEFORE importing torch/heavy libs
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"

    import torch
    torch.set_num_threads(2)

    mp3_files = [Path(p) for p in mp3_files_str]

    write_job_status(job_id, {
        "status": "running",
        "total": len(mp3_files),
        "completed": 0,
        "failed": 0,
        "current_file": None,
        "current_file_progress": 0.0,
        "progress": 0.0,
        "results": [],
        "errors": [],
        "cancelled": False,
        "type": "batch"
    })

    # Import here to avoid loading in main process
    from app.services.analyze import get_analyze_service
    service = get_analyze_service()

    results = []
    errors = []
    completed = 0
    failed = 0

    for i, mp3_file in enumerate(mp3_files):
        # Check cancellation
        current_status = read_job_status(job_id)
        if current_status and current_status.get("cancelled", False):
            write_job_status(job_id, {
                **current_status,
                "status": "cancelled",
                "current_file": None
            })
            return

        def progress_callback(segment, total_segments, file_progress):
            global_progress = (i + file_progress / 100) / len(mp3_files) * 100
            write_job_status(job_id, {
                "status": "running",
                "total": len(mp3_files),
                "completed": completed,
                "failed": failed,
                "current_file": mp3_file.name,
                "current_file_progress": file_progress,
                "progress": round(global_progress, 1),
                "results": results,
                "errors": errors,
                "cancelled": False,
                "type": "batch"
            })

        def is_cancelled():
            current_status = read_job_status(job_id)
            return current_status and current_status.get("cancelled", False)

        try:
            result = service.analyze_file(mp3_file, on_progress=progress_callback, check_cancelled=is_cancelled)
            results.append({
                "mp3": str(mp3_file),
                "csv": result["csv_path"],
                "segments": result["segments_analyzed"]
            })
            completed += 1
        except InterruptedError:
            errors.append({"mp3": str(mp3_file), "error": "Cancelled by user"})
            write_job_status(job_id, {
                "status": "cancelled",
                "total": len(mp3_files),
                "completed": completed,
                "failed": failed,
                "current_file": None,
                "current_file_progress": 0,
                "progress": (completed / len(mp3_files)) * 100,
                "results": results,
                "errors": errors,
                "cancelled": True,
                "type": "batch"
            })
            return
        except Exception as e:
            errors.append({"mp3": str(mp3_file), "error": str(e)})
            failed += 1

    write_job_status(job_id, {
        "status": "completed",
        "total": len(mp3_files),
        "completed": completed,
        "failed": failed,
        "current_file": None,
        "current_file_progress": 100,
        "progress": 100.0,
        "results": results,
        "errors": errors,
        "cancelled": False,
        "type": "batch"
    })

@router.post("/batch", response_model=BatchResponse)
async def batch_analyze(request: BatchRequest):
    """Analyze multiple MP3 files in background process"""
    if request.mp3_paths:
        mp3_files = [Path(p) for p in request.mp3_paths if Path(p).exists()]
    else:
        mp3_files = get_unanalyzed_files(request.year, request.month)

    if not mp3_files:
        raise HTTPException(status_code=404, detail="No files found matching criteria")

    job_id = str(uuid.uuid4())

    initial_status = {
        "status": "queued",
        "total": len(mp3_files),
        "completed": 0,
        "failed": 0,
        "current_file": None,
        "current_file_progress": 0.0,
        "progress": 0.0,
        "results": [],
        "errors": [],
        "cancelled": False,
        "type": "batch",
        "started_at": datetime.now().isoformat()
    }
    write_job_status(job_id, initial_status)
    _jobs[job_id] = initial_status

    # Persist job to SQLite for restart recovery
    try:
        registry = await get_job_registry()
        await registry.create_job(
            job_id=job_id,
            job_type="batch_analysis",
            initial_data=initial_status
        )
    except Exception as e:
        logger.warning(f"Could not persist batch job to SQLite: {e}")

    # Enqueue - the global queue starts it when no other analysis is running
    queue = get_analysis_queue()
    mp3_strs = [str(f) for f in mp3_files]
    await queue.enqueue_batch(job_id, mp3_strs, JOBS_DIR)

    pos = queue.queue_position(job_id)
    if pos is not None:
        message = f"Batch queued (position {pos + 1}): {len(mp3_files)} files"
    else:
        message = f"Batch analysis started: {len(mp3_files)} files"

    return BatchResponse(
        job_id=job_id,
        files_queued=len(mp3_files),
        files=[f.name for f in mp3_files],
        message=message
    )

@router.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Get status of batch analysis job"""
    # Try to read from temp file first (most up-to-date during active processing)
    status = read_job_status(job_id)
    if status:
        _jobs[job_id] = status

        # Also update SQLite for persistence (non-blocking)
        try:
            registry = await get_job_registry()
            await registry.update_job(job_id, status)
        except Exception as e:
            logger.debug(f"Could not update batch job in SQLite: {e}")

        return status

    # Fallback to in-memory cache
    if job_id in _jobs:
        return _jobs[job_id]

    # Final fallback: SQLite (for jobs that survived a server restart)
    try:
        registry = await get_job_registry()
        job = await registry.get_job(job_id)
        if job:
            return job
    except Exception as e:
        logger.debug(f"Could not fetch batch job from SQLite: {e}")

    raise HTTPException(status_code=404, detail="Job not found")

@router.post("/batch/{job_id}/cancel")
async def cancel_batch_job(job_id: str):
    """Cancel a running batch analysis job"""
    status = read_job_status(job_id)
    if not status:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        status = _jobs[job_id]

    if status["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Job is not running (status: {status['status']})")

    # Set cancelled flag - process will check this
    status["cancelled"] = True
    write_job_status(job_id, status)

    return {"message": "Job cancellation requested", "job_id": job_id}

@router.get("/batch")
async def list_all_jobs():
    """List all analysis jobs (batch and single file)"""
    from app.api.v1.analyze import _single_jobs, read_job_status as read_single_status

    jobs_list = []

    # Scan job files for most up-to-date status
    for job_file in JOBS_DIR.glob("*.json"):
        job_id = job_file.stem
        status = read_job_status(job_id)
        if status:
            job_type = status.get("type", "single" if "file" in status else "batch")
            jobs_list.append({
                "job_id": job_id,
                "status": status.get("status", "unknown"),
                "total": status.get("total", status.get("total_segments", 1)),
                "completed": status.get("completed", status.get("current_segment", 0)),
                "failed": status.get("failed", 0),
                "progress": status.get("progress", 0),
                "type": job_type,
                "file": status.get("file", status.get("current_file", ""))
            })

    # Sort: running first, then by progress
    jobs_list.sort(key=lambda x: (x["status"] != "running", -x.get("progress", 0)))
    return jobs_list


@router.get("/outdated-csvs")
async def get_outdated_csvs():
    """Get list of CSVs analyzed with old models"""
    from app.services.model_registry import get_active_model_id, is_csv_edited
    from app.api.v1.uncertainty import derive_mp3_path_from_csv

    active_model_id = get_active_model_id()
    results_folder = settings.SORTED_FOLDER / "ANALYSIS_RESULTS"

    if not results_folder.exists():
        return {"outdated_csvs": [], "count": 0, "active_model": active_model_id}

    outdated = []

    for csv_file in results_folder.glob("predictions_*.csv"):
        try:
            if is_csv_edited(str(csv_file)):
                continue

            # Single read (PERF-01 fix)
            df = pl.read_csv(csv_file, encoding='utf-8')

            if 'model_version' not in df.columns:
                csv_model_version = "unknown"
            else:
                csv_model_version = df[0, 'model_version']

            if csv_model_version == active_model_id:
                continue

            try:
                mp3_path = derive_mp3_path_from_csv(csv_file)
                if not mp3_path.exists():
                    continue
            except Exception as e:
                logger.debug(f"Could not derive MP3 path for {csv_file}: {e}")
                continue

            outdated.append({
                "csv_path": str(csv_file),
                "mp3_path": str(mp3_path),
                "old_model_version": csv_model_version
            })

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    return {"outdated_csvs": outdated, "count": len(outdated), "active_model": active_model_id}

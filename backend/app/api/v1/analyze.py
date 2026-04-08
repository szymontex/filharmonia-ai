"""
Analyze API - uses subprocess to avoid blocking the server
"""
import logging
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel
import uuid
import json
import tempfile
from cachetools import TTLCache

from app.services.job_registry import get_job_registry
from app.services.analysis_queue import get_analysis_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Directory for job status files (cross-platform)
JOBS_DIR = Path(tempfile.gettempdir()) / "filharmonia_jobs"
JOBS_DIR.mkdir(exist_ok=True)

class AnalyzeRequest(BaseModel):
    mp3_path: str

class AnalyzeResponse(BaseModel):
    job_id: str
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
    """Atomic write using temp file + rename pattern"""
    job_file = get_job_file(job_id)

    fd, tmp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=f'{job_id}_',
        dir=JOBS_DIR
    )
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(status, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, job_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

# Keep in-memory cache for quick access
_single_jobs: TTLCache = TTLCache(maxsize=100, ttl=3600)

@router.post("/", response_model=AnalyzeResponse)
async def analyze_file(request: AnalyzeRequest):
    """
    Analyze MP3 file in background process - returns immediately with job_id.
    Jobs are queued globally so only one analysis runs at a time.
    """
    mp3_path = Path(request.mp3_path)

    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {mp3_path}")

    if not mp3_path.suffix.lower() == '.mp3':
        raise HTTPException(status_code=400, detail="File must be .mp3")

    job_id = str(uuid.uuid4())

    initial_status = {
        "status": "queued",
        "file": str(mp3_path),
        "progress": 0,
        "current_segment": 0,
        "total_segments": 0,
        "started_at": datetime.now().isoformat()
    }
    write_job_status(job_id, initial_status)
    _single_jobs[job_id] = initial_status

    # Persist to SQLite
    try:
        registry = await get_job_registry()
        await registry.create_job(
            job_id=job_id,
            job_type="single_analysis",
            initial_data=initial_status
        )
    except Exception as e:
        logger.warning(f"Could not persist job to SQLite: {e}")

    # Enqueue - the queue handles starting it when ready
    queue = get_analysis_queue()
    await queue.enqueue_single(job_id, str(mp3_path), JOBS_DIR)

    pos = queue.queue_position(job_id)
    if pos is not None:
        message = f"Analysis queued (position {pos + 1} in queue)"
    else:
        message = "Analysis started"

    return AnalyzeResponse(job_id=job_id, message=message)

@router.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get status of single file analysis"""
    status = read_job_status(job_id)
    if status:
        # Enrich with queue position if still queued
        if status.get("status") == "queued":
            queue = get_analysis_queue()
            pos = queue.queue_position(job_id)
            if pos is not None:
                status["queue_position"] = pos + 1
                status["queue_total"] = queue.queue_length

        _single_jobs[job_id] = status

        try:
            registry = await get_job_registry()
            await registry.update_job(job_id, status)
        except Exception as e:
            logger.debug(f"Could not update job in SQLite: {e}")

        return status

    if job_id in _single_jobs:
        return _single_jobs[job_id]

    try:
        registry = await get_job_registry()
        job = await registry.get_job(job_id)
        if job:
            return job
    except Exception as e:
        logger.debug(f"Could not fetch job from SQLite: {e}")

    raise HTTPException(status_code=404, detail="Job not found")

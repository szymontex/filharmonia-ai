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
import subprocess
import sys
import tempfile
from cachetools import TTLCache

from app.services.job_registry import get_job_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

# Directory for job status files (cross-platform)
JOBS_DIR = Path(tempfile.gettempdir()) / "filharmonia_jobs"
JOBS_DIR.mkdir(exist_ok=True)

# Path to worker script
WORKER_SCRIPT = Path(__file__).parent.parent.parent / "workers" / "analyze_worker.py"

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

# Keep track of running processes (for cleanup)
_processes = {}

# Also maintain in-memory cache for quick access
# TTL cache: 1 hour expiry, max 100 entries (single jobs are short-lived)
_single_jobs: TTLCache = TTLCache(maxsize=100, ttl=3600)

@router.post("/", response_model=AnalyzeResponse)
async def analyze_file(request: AnalyzeRequest):
    """
    Analyze MP3 file in background process - returns immediately with job_id
    """
    mp3_path = Path(request.mp3_path)

    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {mp3_path}")

    if not mp3_path.suffix.lower() == '.mp3':
        raise HTTPException(status_code=400, detail="File must be .mp3")

    job_id = str(uuid.uuid4())

    # Initialize job status BEFORE starting process
    initial_status = {
        "status": "starting",
        "file": str(mp3_path),
        "progress": 0,
        "current_segment": 0,
        "total_segments": 0,
        "started_at": datetime.now().isoformat()
    }
    write_job_status(job_id, initial_status)
    _single_jobs[job_id] = initial_status

    # Persist job to SQLite for restart recovery
    try:
        registry = await get_job_registry()
        await registry.create_job(
            job_id=job_id,
            job_type="single_analysis",
            initial_data=initial_status
        )
    except Exception as e:
        logger.warning(f"Could not persist job to SQLite: {e}")

    # Get the Python executable from current virtualenv
    python_exe = sys.executable

    # Log file for worker output
    log_file = JOBS_DIR / f"{job_id}.log"

    # Start analysis in separate subprocess
    # This completely isolates CPU-heavy work from the main server
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            [python_exe, str(WORKER_SCRIPT), job_id, str(mp3_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent.parent),
            start_new_session=True  # Detach from parent
        )

    _processes[job_id] = process

    return AnalyzeResponse(
        job_id=job_id,
        message=f"Analysis started in background process (PID: {process.pid})"
    )

@router.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get status of single file analysis"""
    # Try to read from temp file first (most up-to-date during active processing)
    status = read_job_status(job_id)
    if status:
        # Update in-memory cache
        _single_jobs[job_id] = status

        # Also update SQLite for persistence (non-blocking)
        try:
            registry = await get_job_registry()
            await registry.update_job(job_id, status)
        except Exception as e:
            logger.debug(f"Could not update job in SQLite: {e}")

        return status

    # Fallback to in-memory cache
    if job_id in _single_jobs:
        return _single_jobs[job_id]

    # Final fallback: SQLite (for jobs that survived a server restart)
    try:
        registry = await get_job_registry()
        job = await registry.get_job(job_id)
        if job:
            return job
    except Exception as e:
        logger.debug(f"Could not fetch job from SQLite: {e}")

    raise HTTPException(status_code=404, detail="Job not found")

"""
Analyze API
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel
import uuid
import threading
from app.services.analyze import get_analyze_service

router = APIRouter(prefix="/analyze", tags=["analyze"])

class AnalyzeRequest(BaseModel):
    mp3_path: str

class AnalyzeResponse(BaseModel):
    job_id: str
    message: str

# Single file job tracking (reuse from batch)
_single_jobs = {}

def run_single_analysis(job_id: str, mp3_path: Path):
    """Background task for single file analysis"""
    _single_jobs[job_id] = {
        "status": "running",
        "file": str(mp3_path),
        "progress": 0,
        "current_segment": 0,
        "total_segments": 0
    }

    def update_progress(current: int, total: int, percent: float):
        """Callback to update progress"""
        _single_jobs[job_id].update({
            "progress": percent,
            "current_segment": current,
            "total_segments": total
        })

    try:
        service = get_analyze_service()
        result = service.analyze_file(
            mp3_path,
            on_progress=update_progress
        )

        _single_jobs[job_id] = {
            "status": "completed",
            "file": str(mp3_path),
            "csv_path": result['csv_path'],
            "segments_analyzed": result['segments_analyzed'],
            "duration_seconds": result['duration_seconds'],
            "progress": 100,
            "current_segment": result['segments_analyzed'],
            "total_segments": result['segments_analyzed']
        }
    except Exception as e:
        _single_jobs[job_id] = {
            "status": "failed",
            "file": str(mp3_path),
            "error": str(e),
            "progress": 0,
            "current_segment": 0,
            "total_segments": 0
        }

@router.post("/", response_model=AnalyzeResponse)
async def analyze_file(request: AnalyzeRequest):
    """
    Analyze MP3 file in background - returns immediately with job_id
    """
    mp3_path = Path(request.mp3_path)

    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {mp3_path}")

    if not mp3_path.suffix.lower() == '.mp3':
        raise HTTPException(status_code=400, detail="File must be .mp3")

    job_id = str(uuid.uuid4())

    # Initialize job status BEFORE starting thread (prevents 404 on first status check)
    _single_jobs[job_id] = {
        "status": "starting",
        "file": str(mp3_path),
        "progress": 0,
        "current_segment": 0,
        "total_segments": 0
    }

    # Start analysis in daemon thread (doesn't block server restart)
    thread = threading.Thread(
        target=run_single_analysis,
        args=(job_id, mp3_path),
        daemon=True,
        name=f"SingleAnalysis-{job_id[:8]}"
    )
    thread.start()

    return AnalyzeResponse(
        job_id=job_id,
        message=f"Analysis started in background"
    )

@router.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get status of single file analysis"""
    if job_id not in _single_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _single_jobs[job_id]

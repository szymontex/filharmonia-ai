#!/usr/bin/env python3
"""
Standalone analysis worker - runs as separate process via subprocess
This allows proper error logging and isolation from main server
"""
import sys
import os
import json
import tempfile
import traceback
from pathlib import Path

# CRITICAL: Set CPU limits BEFORE importing torch
# Detect platform to set appropriate thread count
import platform as _plat
import subprocess as _sp

def _get_worker_threads() -> int:
    """Get recommended thread count for worker process."""
    if _plat.system() == "Darwin":
        try:
            r = _sp.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                        capture_output=True, text=True, timeout=5)
            return min(int(r.stdout.strip()), 6)
        except Exception:
            pass
    total = os.cpu_count() or 4
    return min(max(total // 2, 2), 4)

_threads = str(_get_worker_threads())
os.environ["OMP_NUM_THREADS"] = _threads
os.environ["MKL_NUM_THREADS"] = _threads
os.environ["OPENBLAS_NUM_THREADS"] = _threads
os.environ["VECLIB_MAXIMUM_THREADS"] = _threads
os.environ["NUMEXPR_NUM_THREADS"] = _threads

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Cross-platform temp directory for job status files
JOBS_DIR = Path(tempfile.gettempdir()) / "filharmonia_jobs"
JOBS_DIR.mkdir(exist_ok=True)

def write_job_status(job_id: str, status: dict):
    job_file = JOBS_DIR / f"{job_id}.json"
    job_file.write_text(json.dumps(status))

def main():
    if len(sys.argv) != 3:
        print("Usage: analyze_worker.py <job_id> <mp3_path>", file=sys.stderr)
        sys.exit(1)

    job_id = sys.argv[1]
    mp3_path_str = sys.argv[2]
    log_file = JOBS_DIR / f"{job_id}.log"

    try:
        print(f"[Worker] Starting analysis for job {job_id}")
        print(f"[Worker] MP3 path: {mp3_path_str}")

        mp3_path = Path(mp3_path_str)

        # Update status to running
        write_job_status(job_id, {
            "status": "running",
            "file": str(mp3_path),
            "progress": 0,
            "current_segment": 0,
            "total_segments": 0
        })

        print("[Worker] Importing torch...")
        import torch
        torch.set_num_threads(2)
        print(f"[Worker] Torch loaded, device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        def update_progress(current: int, total: int, percent: float):
            """Callback to update progress"""
            write_job_status(job_id, {
                "status": "running",
                "file": str(mp3_path),
                "progress": percent,
                "current_segment": current,
                "total_segments": total
            })
            print(f"[Worker] Progress: {current}/{total} ({percent:.1f}%)")

        print("[Worker] Importing analyze service...")
        from app.services.analyze import get_analyze_service

        print("[Worker] Getting service instance...")
        service = get_analyze_service()

        print("[Worker] Starting file analysis...")
        result = service.analyze_file(
            mp3_path,
            on_progress=update_progress
        )

        print(f"[Worker] Analysis completed: {result['segments_analyzed']} segments")

        write_job_status(job_id, {
            "status": "completed",
            "file": str(mp3_path),
            "csv_path": result['csv_path'],
            "segments_analyzed": result['segments_analyzed'],
            "duration_seconds": result['duration_seconds'],
            "progress": 100,
            "current_segment": result['segments_analyzed'],
            "total_segments": result['segments_analyzed']
        })

        print("[Worker] Done!")

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker] ERROR: {error_msg}", file=sys.stderr)
        log_file.write_text(error_msg)

        write_job_status(job_id, {
            "status": "failed",
            "file": mp3_path_str,
            "error": str(e),
            "progress": 0,
            "current_segment": 0,
            "total_segments": 0
        })
        sys.exit(1)

if __name__ == "__main__":
    main()

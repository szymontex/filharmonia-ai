"""
Global analysis queue - controls how many analyses run concurrently.
All analysis requests (single file and batch) go through this queue.

Configure via ANALYSIS_MAX_CONCURRENT env var (default: 1).
"""
import asyncio
import logging
import multiprocessing
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).parent.parent / "workers" / "analyze_worker.py"
MAX_CONCURRENT = int(os.getenv("ANALYSIS_MAX_CONCURRENT", "1"))


class AnalysisQueue:
    """FIFO queue with configurable concurrency limit for analysis jobs."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self.max_concurrent = max(1, max_concurrent)
        self._queue: deque = deque()
        self._running: dict = {}  # job_id -> process
        self._lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None
        # Track all processes for shutdown cleanup
        self.processes: dict = {}

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def running_count(self) -> int:
        return len(self._running)

    def queue_position(self, job_id: str) -> Optional[int]:
        """Return 0-based position in queue, or None if not queued."""
        for i, entry in enumerate(self._queue):
            if entry["job_id"] == job_id:
                return i
        return None

    async def enqueue_single(self, job_id: str, mp3_path: str, jobs_dir: Path):
        """Enqueue a single-file analysis."""
        entry = {
            "job_id": job_id,
            "type": "single",
            "mp3_path": mp3_path,
            "jobs_dir": str(jobs_dir),
        }
        async with self._lock:
            self._queue.append(entry)
            logger.info("Queued single analysis %s (queue depth: %d, running: %d/%d)",
                        job_id[:8], len(self._queue), len(self._running), self.max_concurrent)
        await self._maybe_start_next()

    async def enqueue_batch(self, job_id: str, mp3_files: list[str], jobs_dir: Path):
        """Enqueue a batch analysis."""
        entry = {
            "job_id": job_id,
            "type": "batch",
            "mp3_files": mp3_files,
            "jobs_dir": str(jobs_dir),
        }
        async with self._lock:
            self._queue.append(entry)
            logger.info("Queued batch analysis %s (%d files, queue depth: %d, running: %d/%d)",
                        job_id[:8], len(mp3_files), len(self._queue),
                        len(self._running), self.max_concurrent)
        await self._maybe_start_next()

    async def _maybe_start_next(self):
        """Start queued jobs until we hit the concurrency limit."""
        while True:
            # Grab one entry if we have capacity
            async with self._lock:
                # Reap finished processes
                finished = [jid for jid, proc in self._running.items()
                            if not self._is_alive(proc)]
                for jid in finished:
                    logger.info("Analysis %s finished", jid[:8])
                    del self._running[jid]

                if not self._queue or len(self._running) >= self.max_concurrent:
                    break
                entry = self._queue.popleft()

            # Launch outside the lock
            job_id = entry["job_id"]
            if entry["type"] == "single":
                jobs_dir = Path(entry["jobs_dir"])
                process = self._start_single(job_id, entry["mp3_path"], jobs_dir)
            else:
                process = self._start_batch(job_id, entry["mp3_files"])

            async with self._lock:
                self._running[job_id] = process
                self.processes[job_id] = process

        # Ensure poller is running if we have active jobs
        async with self._lock:
            has_running = bool(self._running)
            has_queued = bool(self._queue)
        if has_running or has_queued:
            if self._poll_task is None or self._poll_task.done():
                self._poll_task = asyncio.ensure_future(self._poll_completion())

    @staticmethod
    def _is_alive(proc) -> bool:
        if isinstance(proc, subprocess.Popen):
            return proc.poll() is None
        elif hasattr(proc, "is_alive"):
            return proc.is_alive()
        return False

    def _start_single(self, job_id: str, mp3_path: str, jobs_dir: Path):
        """Launch single-file analysis subprocess."""
        python_exe = sys.executable
        log_file = jobs_dir / f"{job_id}.log"

        with open(log_file, "w") as log:
            process = subprocess.Popen(
                [python_exe, str(WORKER_SCRIPT), job_id, mp3_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent),
                start_new_session=True,
            )
        logger.info("Started single analysis %s (PID %d)", job_id[:8], process.pid)
        return process

    def _start_batch(self, job_id: str, mp3_files: list[str]):
        """Launch batch analysis in a separate process."""
        from app.api.v1.batch import run_batch_analysis_process

        process = multiprocessing.Process(
            target=run_batch_analysis_process,
            args=(job_id, mp3_files),
            daemon=True,
            name=f"BatchAnalysis-{job_id[:8]}",
        )
        process.start()
        logger.info("Started batch analysis %s (PID %d, %d files)",
                     job_id[:8], process.pid, len(mp3_files))
        return process

    async def _poll_completion(self):
        """Poll running processes; when one finishes, start next from queue."""
        while True:
            await asyncio.sleep(1)
            async with self._lock:
                any_running = any(self._is_alive(p) for p in self._running.values())
                if not any_running and not self._queue:
                    break
            await self._maybe_start_next()

    def terminate_all(self):
        """Terminate all running and queued processes (for shutdown)."""
        self._queue.clear()
        for job_id, proc in list(self.processes.items()):
            if isinstance(proc, subprocess.Popen):
                if proc.poll() is None:
                    logger.info("Terminating worker %s", job_id[:8])
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            elif hasattr(proc, "terminate"):
                if proc.is_alive():
                    logger.info("Terminating batch worker %s", job_id[:8])
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
        self.processes.clear()
        self._running.clear()


# Singleton
_queue: Optional[AnalysisQueue] = None


def get_analysis_queue() -> AnalysisQueue:
    global _queue
    if _queue is None:
        _queue = AnalysisQueue()
    return _queue

"""
SQLite-backed job registry with TTL cache for fast reads.

Purpose: Jobs currently stored in temp files and in-memory dicts are lost on restart.
This service provides persistent storage that survives server restarts while
maintaining fast reads via TTL cache.

Usage:
    from app.services.job_registry import get_job_registry

    registry = await get_job_registry()
    await registry.create_job("job-123", "batch", {"status": "starting"})
    job = await registry.get_job("job-123")
    await registry.update_job("job-123", {"status": "running", "progress": 50})
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from cachetools import TTLCache


class JobRegistry:
    """
    SQLite-backed job storage with TTL cache for hot reads.

    Features:
    - SQLite persistence survives server restart
    - TTLCache (1000 entries, 1 hour TTL) for fast reads
    - WAL mode for better concurrency
    - Async write lock to prevent "database is locked" errors
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the job registry.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._cache: TTLCache[str, Dict[str, Any]] = TTLCache(maxsize=1000, ttl=3600)
        self._write_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """
        Create tables if not exist, enable WAL mode.

        Must be called before using the registry.
        """
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")

            # Create jobs table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Create indexes for common queries
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)"
            )

            await db.commit()

        self._initialized = True

    async def create_job(
        self,
        job_id: str,
        job_type: str,
        initial_data: Dict[str, Any]
    ) -> None:
        """
        Create a new job entry.

        Args:
            job_id: Unique job identifier
            job_type: Type of job ('single' or 'batch')
            initial_data: Initial job data including status

        Raises:
            RuntimeError: If registry not initialized
        """
        if not self._initialized:
            raise RuntimeError("JobRegistry not initialized. Call initialize() first.")

        status = initial_data.get('status', 'starting')
        data_json = json.dumps(initial_data)

        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO jobs (job_id, job_type, status, data)
                       VALUES (?, ?, ?, ?)""",
                    (job_id, job_type, status, data_json)
                )
                await db.commit()

        # Populate cache
        self._cache[job_id] = initial_data

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job data, cache-first with SQLite fallback.

        Args:
            job_id: Job identifier to look up

        Returns:
            Job data dict if found, None otherwise

        Raises:
            RuntimeError: If registry not initialized
        """
        if not self._initialized:
            raise RuntimeError("JobRegistry not initialized. Call initialize() first.")

        # Check hot cache first
        if job_id in self._cache:
            return self._cache[job_id]

        # Fallback to SQLite
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT data FROM jobs WHERE job_id = ?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    data = json.loads(row['data'])
                    # Populate cache for next read
                    self._cache[job_id] = data
                    return data

        return None

    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """
        Update job with new data (merge with existing).

        Args:
            job_id: Job identifier to update
            updates: Dict of fields to update (merged with existing data)

        Raises:
            ValueError: If job not found
            RuntimeError: If registry not initialized
        """
        if not self._initialized:
            raise RuntimeError("JobRegistry not initialized. Call initialize() first.")

        current = await self.get_job(job_id)
        if current is None:
            raise ValueError(f"Job {job_id} not found")

        # Merge updates into current data
        current.update(updates)
        status = current.get('status', 'running')
        data_json = json.dumps(current)

        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """UPDATE jobs
                       SET status = ?, data = ?, updated_at = datetime('now')
                       WHERE job_id = ?""",
                    (status, data_json, job_id)
                )
                await db.commit()

        # Update cache
        self._cache[job_id] = current

    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List jobs, optionally filtered by status.

        Args:
            status: Filter by status (optional)
            limit: Maximum number of jobs to return (default 100)

        Returns:
            List of job data dicts with job_id and job_type included

        Raises:
            RuntimeError: If registry not initialized
        """
        if not self._initialized:
            raise RuntimeError("JobRegistry not initialized. Call initialize() first.")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if status:
                query = """SELECT job_id, job_type, status, data
                          FROM jobs WHERE status = ?
                          ORDER BY created_at DESC LIMIT ?"""
                params = (status, limit)
            else:
                query = """SELECT job_id, job_type, status, data
                          FROM jobs
                          ORDER BY created_at DESC LIMIT ?"""
                params = (limit,)

            async with db.execute(query, params) as cursor:
                jobs = []
                async for row in cursor:
                    data = json.loads(row['data'])
                    data['job_id'] = row['job_id']
                    data['job_type'] = row['job_type']
                    jobs.append(data)
                return jobs

    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Remove jobs older than N days.

        Args:
            days: Delete jobs older than this many days (default 7)

        Returns:
            Number of jobs deleted

        Raises:
            RuntimeError: If registry not initialized
        """
        if not self._initialized:
            raise RuntimeError("JobRegistry not initialized. Call initialize() first.")

        async with self._write_lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Get count before delete
                async with db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE created_at < datetime('now', ?)",
                    (f'-{days} days',)
                ) as cursor:
                    row = await cursor.fetchone()
                    count = row[0] if row else 0

                # Delete old jobs
                await db.execute(
                    "DELETE FROM jobs WHERE created_at < datetime('now', ?)",
                    (f'-{days} days',)
                )
                await db.commit()

        # Note: Cache entries will expire naturally via TTL
        return count


# Singleton instance
_registry: Optional[JobRegistry] = None


async def get_job_registry() -> JobRegistry:
    """
    Get or create the singleton JobRegistry instance.

    The registry is stored at `settings.FILHARMONIA_BASE / '.claude' / 'jobs.db'`.

    Returns:
        Initialized JobRegistry instance
    """
    global _registry

    if _registry is None:
        from app.config import settings
        db_path = settings.FILHARMONIA_BASE / '.claude' / 'jobs.db'
        _registry = JobRegistry(db_path)
        await _registry.initialize()

    return _registry


def reset_registry() -> None:
    """
    Reset the singleton registry (for testing purposes).
    """
    global _registry
    _registry = None

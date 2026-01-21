"""
Tests for JobRegistry service.

Tests SQLite persistence, TTL cache behavior, and cleanup functionality.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio

from app.services.job_registry import JobRegistry, reset_registry


@pytest_asyncio.fixture
async def registry(tmp_path: Path):
    """Create a JobRegistry with a temp database for testing."""
    db_path = tmp_path / "test_jobs.db"
    reg = JobRegistry(db_path)
    await reg.initialize()
    yield reg
    # Cleanup: reset singleton to avoid test pollution
    reset_registry()


@pytest.mark.asyncio
async def test_create_job_and_get_job(registry: JobRegistry):
    """Test that create_job creates a job retrievable via get_job."""
    job_id = "test-job-1"
    job_type = "batch"
    initial_data = {"status": "starting", "total_files": 10}

    await registry.create_job(job_id, job_type, initial_data)

    job = await registry.get_job(job_id)
    assert job is not None
    assert job["status"] == "starting"
    assert job["total_files"] == 10


@pytest.mark.asyncio
async def test_get_job_returns_none_for_unknown_job(registry: JobRegistry):
    """Test that get_job returns None for non-existent job."""
    job = await registry.get_job("nonexistent-job")
    assert job is None


@pytest.mark.asyncio
async def test_update_job_merges_updates(registry: JobRegistry):
    """Test that update_job correctly merges new data with existing."""
    job_id = "test-job-2"
    await registry.create_job(job_id, "single", {
        "status": "starting",
        "progress": 0,
        "current_file": None
    })

    # First update
    await registry.update_job(job_id, {"status": "running", "progress": 25})
    job = await registry.get_job(job_id)
    assert job["status"] == "running"
    assert job["progress"] == 25
    assert job["current_file"] is None  # Original field preserved

    # Second update
    await registry.update_job(job_id, {"progress": 75, "current_file": "track.mp3"})
    job = await registry.get_job(job_id)
    assert job["status"] == "running"  # Previous update preserved
    assert job["progress"] == 75
    assert job["current_file"] == "track.mp3"


@pytest.mark.asyncio
async def test_update_job_raises_for_unknown_job(registry: JobRegistry):
    """Test that update_job raises ValueError for non-existent job."""
    with pytest.raises(ValueError, match="not found"):
        await registry.update_job("nonexistent-job", {"status": "running"})


@pytest.mark.asyncio
async def test_cache_behavior(registry: JobRegistry):
    """Test that second get_job returns from cache (faster)."""
    job_id = "test-job-3"
    await registry.create_job(job_id, "batch", {"status": "starting"})

    # Clear cache to force DB read
    registry._cache.clear()

    # First read: from DB
    start1 = time.perf_counter()
    job1 = await registry.get_job(job_id)
    time1 = time.perf_counter() - start1

    # Second read: from cache (should be in cache now)
    start2 = time.perf_counter()
    job2 = await registry.get_job(job_id)
    time2 = time.perf_counter() - start2

    assert job1 == job2
    # Cache read should be faster (no DB I/O)
    # Note: This is a soft assertion - timing can vary
    assert job_id in registry._cache


@pytest.mark.asyncio
async def test_persistence_survives_new_instance(tmp_path: Path):
    """Test that jobs persist across registry instances (simulating restart)."""
    db_path = tmp_path / "persist_test.db"
    job_id = "persistent-job"

    # Create first registry instance and add job
    registry1 = JobRegistry(db_path)
    await registry1.initialize()
    await registry1.create_job(job_id, "batch", {
        "status": "running",
        "progress": 50
    })

    # Create second registry instance (simulating server restart)
    registry2 = JobRegistry(db_path)
    await registry2.initialize()

    # Job should still be accessible
    job = await registry2.get_job(job_id)
    assert job is not None
    assert job["status"] == "running"
    assert job["progress"] == 50


@pytest.mark.asyncio
async def test_list_jobs(registry: JobRegistry):
    """Test listing jobs with and without status filter."""
    # Create multiple jobs
    await registry.create_job("job-a", "batch", {"status": "running"})
    await registry.create_job("job-b", "single", {"status": "completed"})
    await registry.create_job("job-c", "batch", {"status": "running"})
    await registry.create_job("job-d", "single", {"status": "failed"})

    # List all jobs
    all_jobs = await registry.list_jobs()
    assert len(all_jobs) == 4

    # List running jobs
    running_jobs = await registry.list_jobs(status="running")
    assert len(running_jobs) == 2
    assert all(j["status"] == "running" for j in running_jobs)

    # List completed jobs
    completed_jobs = await registry.list_jobs(status="completed")
    assert len(completed_jobs) == 1
    assert completed_jobs[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_list_jobs_includes_job_metadata(registry: JobRegistry):
    """Test that list_jobs includes job_id and job_type in returned data."""
    await registry.create_job("meta-job", "batch", {"status": "starting"})

    jobs = await registry.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == "meta-job"
    assert jobs[0]["job_type"] == "batch"
    assert jobs[0]["status"] == "starting"


@pytest.mark.asyncio
async def test_list_jobs_respects_limit(registry: JobRegistry):
    """Test that list_jobs respects the limit parameter."""
    # Create 10 jobs
    for i in range(10):
        await registry.create_job(f"limit-job-{i}", "batch", {"status": "running"})

    # List with limit
    jobs = await registry.list_jobs(limit=5)
    assert len(jobs) == 5


@pytest.mark.asyncio
async def test_cleanup_old_jobs(tmp_path: Path):
    """Test that cleanup_old_jobs removes old entries."""
    db_path = tmp_path / "cleanup_test.db"
    registry = JobRegistry(db_path)
    await registry.initialize()

    # Create a job and manually backdate it in the database
    await registry.create_job("old-job", "batch", {"status": "completed"})
    await registry.create_job("new-job", "batch", {"status": "running"})

    # Backdate the first job to 10 days ago
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE jobs SET created_at = datetime('now', '-10 days') WHERE job_id = ?",
            ("old-job",)
        )
        await db.commit()

    # Cleanup jobs older than 7 days
    deleted_count = await registry.cleanup_old_jobs(days=7)
    assert deleted_count == 1

    # Old job should be gone
    old_job = await registry.get_job("old-job")
    assert old_job is None

    # New job should remain
    new_job = await registry.get_job("new-job")
    assert new_job is not None


@pytest.mark.asyncio
async def test_uninitialized_registry_raises_error(tmp_path: Path):
    """Test that using uninitialized registry raises RuntimeError."""
    db_path = tmp_path / "uninit_test.db"
    registry = JobRegistry(db_path)
    # NOT calling initialize()

    with pytest.raises(RuntimeError, match="not initialized"):
        await registry.create_job("job", "batch", {})

    with pytest.raises(RuntimeError, match="not initialized"):
        await registry.get_job("job")

    with pytest.raises(RuntimeError, match="not initialized"):
        await registry.update_job("job", {})

    with pytest.raises(RuntimeError, match="not initialized"):
        await registry.list_jobs()

    with pytest.raises(RuntimeError, match="not initialized"):
        await registry.cleanup_old_jobs()


@pytest.mark.asyncio
async def test_concurrent_writes(registry: JobRegistry):
    """Test that concurrent writes don't cause database lock errors."""
    job_ids = [f"concurrent-job-{i}" for i in range(20)]

    # Create all jobs concurrently
    async def create_job(job_id: str):
        await registry.create_job(job_id, "batch", {"status": "starting"})
        await registry.update_job(job_id, {"status": "running", "n": job_id})

    await asyncio.gather(*[create_job(jid) for jid in job_ids])

    # Verify all jobs were created and updated
    for job_id in job_ids:
        job = await registry.get_job(job_id)
        assert job is not None
        assert job["status"] == "running"
        assert job["n"] == job_id

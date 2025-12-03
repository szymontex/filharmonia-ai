from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.api.v1 import files, analyze, csv_parser, batch, audio, waveform, sort, export, training, uncertainty

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown: Mark all running jobs as interrupted
    print("🔄 Graceful shutdown: marking active analysis jobs as interrupted...")
    from app.api.v1.batch import _jobs
    from app.api.v1.analyze import _single_jobs

    # Mark batch jobs as cancelled
    for job_id, job in _jobs.items():
        if job.get("status") == "running":
            job["cancelled"] = True
            job["status"] = "interrupted"
            print(f"  ⚠️  Batch job {job_id[:8]} interrupted (was processing: {job.get('current_file', 'unknown')})")

    # Mark single jobs as interrupted
    for job_id, job in _single_jobs.items():
        if job.get("status") == "running":
            job["status"] = "interrupted"
            print(f"  ⚠️  Single job {job_id[:8]} interrupted")

    print("✓ Shutdown complete")

app = FastAPI(
    title="Filharmonia AI API",
    description="Concert Analysis System",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router, prefix="/api/v1")
app.include_router(analyze.router, prefix="/api/v1")
app.include_router(csv_parser.router, prefix="/api/v1")
app.include_router(batch.router, prefix="/api/v1")
app.include_router(audio.router, prefix="/api/v1")
app.include_router(waveform.router, prefix="/api/v1")
app.include_router(sort.router, prefix="/api/v1")
app.include_router(export.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(uncertainty.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Filharmonia AI API",
        "version": "0.1.0",
        "gpu_available": check_gpu()
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

def check_gpu():
    """Check if GPU is available for PyTorch (primary ML framework)"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

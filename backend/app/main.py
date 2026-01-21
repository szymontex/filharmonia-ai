from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging
import uuid
from app.config import settings

logger = logging.getLogger(__name__)
from app.api.v1 import files, analyze, csv_parser, batch, audio, waveform, sort, export, training, uncertainty

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    print("=" * 50)
    print("Filharmonia AI - Starting up...")
    print("=" * 50)

    # Check PyTorch and GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device = "CUDA" if gpu_available else "CPU"
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  PyTorch device: {device} ({gpu_name})")
        else:
            print(f"  PyTorch device: {device}")
        print(f"  PyTorch version: {torch.__version__}")
    except ImportError:
        print("  WARNING: PyTorch not installed")

    # Check torchaudio
    torchaudio_available = False
    try:
        import torchaudio
        print(f"  torchaudio version: {torchaudio.__version__}")
        print(f"  Audio backend: torchaudio (native)")
        torchaudio_available = True
    except ImportError:
        print("  WARNING: torchaudio not installed, falling back to librosa")

    # Check librosa and its backends (if torchaudio not available)
    if not torchaudio_available:
        try:
            import librosa
            print(f"  librosa version: {librosa.__version__}")

            # Check soundfile (preferred backend)
            try:
                import soundfile
                print(f"  Audio backend: soundfile {soundfile.__version__} (fast)")
            except ImportError:
                print("  WARNING: soundfile not installed")

                # Check audioread fallback
                try:
                    import audioread
                    print(f"  Audio backend: audioread (slow, ffmpeg-based)")
                    print("  RECOMMENDATION: Install soundfile for faster audio loading")
                except ImportError:
                    print("  ERROR: No audio backend available!")
                    print("  Install soundfile: pip install soundfile")
        except ImportError:
            print("  WARNING: librosa not installed")

    # Check ffmpeg (needed for some audio formats)
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ffmpeg: {version_line}")
        else:
            print("  WARNING: ffmpeg not working correctly")
    except FileNotFoundError:
        print("  WARNING: ffmpeg not found in PATH (needed for some audio formats)")

    print("=" * 50)
    print("Startup complete!")
    print("=" * 50)

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


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions from FastAPI and Starlette internals."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "type": "http_error"
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with useful details."""
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "details": exc.errors(),
            "type": "validation_error"
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected errors.
    Logs full traceback but returns generic message with error ID to client.
    """
    error_id = str(uuid.uuid4())[:8]
    logger.error(
        f"[{error_id}] Unhandled exception on {request.method} {request.url}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "error_id": error_id,
            "type": "server_error"
        }
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

@app.get("/api/v1/info")
async def info():
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
    except ImportError:
        return False

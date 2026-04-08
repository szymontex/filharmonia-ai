from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging
import signal
import subprocess
import uuid
from app.config import settings

logger = logging.getLogger(__name__)
from app.api.v1 import files, analyze, csv_parser, batch, audio, waveform, sort, export, training, uncertainty, filharmonia

def terminate_all_workers():
    """Terminate all worker processes via global analysis queue"""
    from app.services.analysis_queue import get_analysis_queue
    get_analysis_queue().terminate_all()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    print("=" * 50)
    print("Filharmonia AI - Starting up...")
    print("=" * 50)

    # Unified device detection
    try:
        import torch
        from app.core.device_manager import get_device_manager
        dm = get_device_manager()
        logger.info("Device: %s (%s)", dm.device_type, dm.device_name)
        print(f"  PyTorch device: {dm.device_type} ({dm.device_name})")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Threads: {dm.recommended_threads}, Batch size: {dm.recommended_batch_size}")
        if dm.is_mps:
            print(f"  Apple Silicon: {dm.device_name}")
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

    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        terminate_all_workers()

    import sys
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    yield

    # Shutdown
    logger.info("Shutting down, terminating workers...")
    terminate_all_workers()

    # Mark running jobs as interrupted in cache
    from app.api.v1.batch import _jobs
    from app.api.v1.analyze import _single_jobs

    for job_id, job in _jobs.items():
        if job.get("status") == "running":
            job["cancelled"] = True
            job["status"] = "interrupted"

    for job_id, job in _single_jobs.items():
        if job.get("status") == "running":
            job["status"] = "interrupted"

    # Cleanup old jobs from SQLite
    try:
        from app.services.job_registry import get_job_registry
        import asyncio
        registry = asyncio.get_event_loop().run_until_complete(get_job_registry())
        asyncio.get_event_loop().run_until_complete(registry.cleanup_old_jobs(days=7))
        logger.info("Old jobs cleaned up from SQLite")
    except Exception as e:
        logger.warning(f"Could not cleanup old jobs: {e}")

    logger.info("Shutdown complete")

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
            "code": f"HTTP_{exc.status_code}",
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
            "code": "VALIDATION_ERROR",
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
            "code": "INTERNAL_ERROR",
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

# Request timeout (excludes /analyze endpoints)
from app.core.middleware import TimeoutMiddleware
app.add_middleware(TimeoutMiddleware, timeout=60.0)

# Silence noisy polling/autosave endpoints from uvicorn access log
class _QuietPollFilter(logging.Filter):
    NOISY = ("/csv/autosave", "/analyze/batch")
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self.NOISY)

logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())

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
app.include_router(filharmonia.router, prefix="/api/v1")

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
    """Health check with system information"""
    import torch

    # Determine audio backend
    audio_backend = "unknown"
    try:
        import torchaudio
        audio_backend = f"torchaudio {torchaudio.__version__}"
    except ImportError:
        try:
            import soundfile
            audio_backend = f"soundfile {soundfile.__version__}"
        except ImportError:
            try:
                import audioread
                audio_backend = "audioread (ffmpeg)"
            except ImportError:
                audio_backend = "none"

    from app.core.device_manager import get_device_manager
    dm = get_device_manager()

    return {
        "status": "healthy",
        "device": dm.device_type,
        "device_name": dm.device_name,
        "audio_backend": audio_backend,
        "torch_version": torch.__version__
    }

def check_gpu():
    """Check if GPU is available for PyTorch (CUDA, ROCm, or MPS)"""
    try:
        from app.core.device_manager import get_device_manager
        return get_device_manager().is_gpu
    except Exception:
        return False

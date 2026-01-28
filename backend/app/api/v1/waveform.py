"""
Waveform Data Generation API
"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import hashlib
import asyncio
import librosa
import numpy as np

from app.config import settings
from app.core.security import validate_path_or_raise_http

router = APIRouter(prefix="/waveform", tags=["waveform"])


def get_cache_path(mp3_path: Path, samples_per_pixel: int) -> Path:
    """
    Generate cache file path based on file path and params.
    Includes file mtime in cache key for automatic invalidation when MP3 changes.
    """
    mtime = mp3_path.stat().st_mtime
    # Include mp3_path, samples_per_pixel, and mtime in cache key
    cache_key = f"{mp3_path}:{samples_per_pixel}:{mtime}"
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    return settings.WAVEFORM_CACHE_DIR / f"{mp3_path.stem}_{hash_key}.json"

@router.get("/data")
async def get_waveform_data(
    path: str = Query(..., description="Path to MP3 file"),
    samples_per_pixel: int = Query(512, description="Samples per pixel (lower = more detail)")
):
    """
    Generate waveform data for Peaks.js

    Returns min/max amplitude values for each pixel.
    Results are cached to filesystem for instant repeat access.
    """
    # Validate path is within SORTED_FOLDER
    mp3_path = validate_path_or_raise_http(path, settings.SORTED_FOLDER)
    cache_path = get_cache_path(mp3_path, samples_per_pixel)

    # Check cache first
    if cache_path.exists():
        try:
            cached_data = await asyncio.to_thread(cache_path.read_text)
            return JSONResponse(json.loads(cached_data))
        except (json.JSONDecodeError, IOError):
            # Cache corrupted, regenerate
            pass

    try:
        # Load audio (mono, lower sample rate for speed)
        def _load_and_process():
            y, sr = librosa.load(str(mp3_path), sr=8000, mono=True)

            # Calculate how many data points we need
            num_pixels = len(y) // samples_per_pixel

            # Generate min/max for each pixel
            data = []
            for i in range(num_pixels):
                start = i * samples_per_pixel
                end = start + samples_per_pixel
                segment = y[start:end]

                if len(segment) > 0:
                    data.append({
                        'min': float(np.min(segment)),
                        'max': float(np.max(segment))
                    })

            return {
                'sample_rate': sr,
                'samples_per_pixel': samples_per_pixel,
                'length': len(y),
                'duration': len(y) / sr,
                'data': data
            }

        result = await asyncio.to_thread(_load_and_process)

        # Cache result (create dir if needed)
        def _write_cache():
            settings.WAVEFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result))

        await asyncio.to_thread(_write_cache)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating waveform: {str(e)}")

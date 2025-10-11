"""
Waveform Data Generation API
"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import librosa
import numpy as np

router = APIRouter(prefix="/waveform", tags=["waveform"])

@router.get("/data")
async def get_waveform_data(
    path: str = Query(..., description="Path to MP3 file"),
    samples_per_pixel: int = Query(512, description="Samples per pixel (lower = more detail)")
):
    """
    Generate waveform data for Peaks.js

    Returns min/max amplitude values for each pixel
    Much faster than full waveform rendering
    """
    mp3_path = Path(path)

    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        # Load audio (mono, lower sample rate for speed)
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

        return JSONResponse({
            'sample_rate': sr,
            'samples_per_pixel': samples_per_pixel,
            'length': len(y),
            'duration': len(y) / sr,
            'data': data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating waveform: {str(e)}")

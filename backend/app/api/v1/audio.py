"""
Audio Streaming API with Range Request Support
"""
from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
import os

router = APIRouter(prefix="/audio", tags=["audio"])

@router.get("/stream")
async def stream_audio(
    path: str = Query(..., description="Path to MP3 file"),
    request: Request = None
):
    """
    Stream MP3 file with range request support for seeking

    Example: /api/v1/audio/stream?path=Y:\\!_FILHARMONIA\\SORTED\\2025\\05\\13\\SONG059.MP3
    """
    mp3_path = Path(path)

    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not mp3_path.suffix.lower() == '.mp3':
        raise HTTPException(status_code=400, detail="File must be .mp3")

    file_size = os.path.getsize(mp3_path)

    # Parse Range header
    range_header = request.headers.get("range")

    if range_header:
        # Parse range like "bytes=0-1023"
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if byte_range[1] else file_size - 1

        # Read chunk
        with open(mp3_path, "rb") as f:
            f.seek(start)
            chunk_size = end - start + 1
            data = f.read(chunk_size)

        # Return 206 Partial Content
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(data)),
            "Content-Type": "audio/mpeg",
        }

        return Response(content=data, status_code=206, headers=headers)

    else:
        # Return full file with Accept-Ranges header
        def iter_file():
            with open(mp3_path, "rb") as f:
                yield from f

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }

        return StreamingResponse(
            iter_file(),
            media_type="audio/mpeg",
            headers=headers
        )

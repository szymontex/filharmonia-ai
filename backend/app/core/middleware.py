"""
Custom middleware for request handling
"""
import asyncio
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to timeout long-running requests.

    Note: This only works for async endpoints. Sync endpoints that block
    the event loop will not be interrupted by this timeout.

    Excludes /analyze endpoints which are meant to be long-running
    (they return immediately with job_id, actual work is in subprocess).
    """

    def __init__(self, app, timeout: float = 60.0):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        # Skip timeout for analysis endpoints (they're meant to be long-running)
        # These endpoints actually return quickly with a job_id; the heavy work
        # happens in separate processes. But we keep this check for safety.
        if "/analyze" in request.url.path:
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout: {request.method} {request.url.path}")
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "message": "Request timeout",
                    "type": "timeout_error"
                }
            )

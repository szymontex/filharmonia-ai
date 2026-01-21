"""
Security utilities for Filharmonia AI
"""
from pathlib import Path
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class PathTraversalError(ValueError):
    """Raised when path traversal is detected"""
    pass


def validate_path_within_root(user_path: str, allowed_root: Path, must_exist: bool = True) -> Path:
    """
    Validate that a user-provided path stays within allowed directory.

    Handles:
    - Relative paths with ..
    - Absolute paths
    - Symlinks (resolved to real path)

    Args:
        user_path: User-provided path (can be relative or absolute)
        allowed_root: The directory that user_path must be within
        must_exist: If True, raises FileNotFoundError if path doesn't exist

    Returns:
        Resolved, validated Path object

    Raises:
        PathTraversalError: If path is outside allowed directory
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    root = allowed_root.resolve()

    # Handle both absolute and relative paths
    user_path_obj = Path(user_path)
    if user_path_obj.is_absolute():
        requested = user_path_obj.resolve()
    else:
        requested = (root / user_path).resolve()

    # Check ancestry using Path.parents (handles symlinks correctly)
    if root not in requested.parents and requested != root:
        logger.warning(f"Path traversal attempt blocked: {user_path} -> {requested}")
        raise PathTraversalError(f"Access denied: path outside allowed directory")

    if must_exist and not requested.exists():
        raise FileNotFoundError(f"File not found: {requested}")

    return requested


def validate_path_or_raise_http(user_path: str, allowed_root: Path, must_exist: bool = True) -> Path:
    """
    Validate path and raise HTTPException on failure.

    Convenience wrapper for use in FastAPI endpoints.
    """
    try:
        return validate_path_within_root(user_path, allowed_root, must_exist)
    except PathTraversalError:
        raise HTTPException(status_code=403, detail="Access denied: path outside allowed directory")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {user_path}")

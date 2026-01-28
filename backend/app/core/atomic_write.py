"""
Atomic file write utility
"""
import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, content: str, encoding: str = 'utf-8') -> None:
    """Write content atomically using temp file + os.replace.

    Creates a temp file in the same directory as target, writes content,
    then atomically replaces the target. If anything fails, the temp file
    is cleaned up and the original file remains untouched.

    Args:
        path: Target file path
        content: Content to write
        encoding: Text encoding (default: utf-8)

    Raises:
        OSError: If file operations fail
    """
    parent = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

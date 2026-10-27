"""Small filesystem helpers: atomic writes that survive a forced exit.

A crash, kill, or power loss mid-write can leave a half-written file on disk.
Writing to a temporary file in the *same directory* and then ``os.replace``-ing
it into place is atomic on POSIX and Windows, so a reader (or a resumed run)
only ever sees the complete old file or the complete new file — never a
truncated one. This matters for the markdown output, the ``_chunks.jsonl``, and
the JSON manifest / run-state / page-cache files that drive resume.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path | str, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write ``text`` to ``path``.

    The temp file is created in the destination directory (so ``os.replace`` is
    a same-filesystem rename) and ``fsync``-ed before the rename. On any error
    the temp file is cleaned up and the original ``path`` is left untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def write_json_atomic(path: Path | str, data: Any, *, indent: int | None = 2) -> Path:
    """Atomically serialize ``data`` to ``path`` as UTF-8 JSON."""
    return write_text_atomic(path, json.dumps(data, indent=indent, ensure_ascii=False))

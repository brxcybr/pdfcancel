"""Signal-aware stop flag and durable run state for long jobs.

``stop_on_signals`` turns SIGINT/SIGTERM into a cooperative stop: instead of
tearing the process down mid-write, the handler sets a ``threading.Event`` that
the conversion loop checks so in-flight pages can finish and checkpoint first.

``RunState`` persists overall progress for a directory/backlog run to a JSON
file (written atomically after every document) so ``pdfcancel status`` can
report progress without grepping a log, and so a resumed run has a record.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pdfcancel.fsio import write_json_atomic

RUN_STATE_NAME = ".pdfcancel_run.json"


@contextmanager
def stop_on_signals() -> Iterator[threading.Event]:
    """Install SIGINT/SIGTERM handlers that *set* (not raise) a stop Event.

    Restores the previous handlers on exit. Registration is skipped gracefully
    when not running on the main thread (e.g. under some test runners), in which
    case the returned Event simply never fires from a signal.
    """
    stop = threading.Event()
    previous: dict[int, object] = {}

    def _handler(signum, frame):  # noqa: ANN001 - signal handler signature
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            previous[sig] = None
    try:
        yield stop
    finally:
        for sig, handler in previous.items():
            if handler is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunState:
    """Durable per-run progress, flushed atomically after each document."""

    def __init__(self, path: Path | str, *, total: int) -> None:
        self.path = Path(path)
        now = _now()
        self.data: dict = {
            "started_at": now,
            "updated_at": now,
            "total": total,
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "incomplete": 0,
            "current": None,
            "failures": [],
            "finished": False,
        }

    def flush(self) -> None:
        self.data["updated_at"] = _now()
        write_json_atomic(self.path, self.data)

    def set_current(self, name: str | None) -> None:
        self.data["current"] = name
        self.flush()

    def record(self, name: str, status: str) -> None:
        """Count a finished document. ``status`` is done/skipped/incomplete/failed."""
        if status in ("done", "skipped", "incomplete"):
            self.data[status] += 1
        else:
            self.data["failed"] += 1
            self.data["failures"].append(name)
        self.flush()

    def finish(self) -> None:
        self.data["current"] = None
        self.data["finished"] = True
        self.flush()

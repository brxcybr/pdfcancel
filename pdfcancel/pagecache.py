"""Per-page transcription checkpoint for the vlm OCR engine.

Each page's markdown is written to its own small file as soon as it is
transcribed, so a crash, kill, or OS sleep costs at most the pages still in
flight — a resumed run reloads the finished pages and only transcribes the
missing ones. A sidecar ``meta.json`` records the inputs the cache is valid for
(pdf content hash, dpi, model, prompt); if any of those change, the cache is
reset so stale pages are never reused.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pdfcancel.fsio import write_json_atomic, write_text_atomic

CACHE_DIRNAME = ".pdfcancel_cache"


class PageCache:
    """Durable per-page markdown cache rooted at ``<root>/.pdfcancel_cache/<stem>``."""

    def __init__(self, root: Path | str, *, stem: str, meta: dict) -> None:
        self.dir = Path(root) / CACHE_DIRNAME / stem
        self.meta = meta
        self._meta_path = self.dir / "meta.json"

    def _page_path(self, index: int) -> Path:
        """Path for a 0-indexed page (stored 1-indexed, zero-padded)."""
        return self.dir / f"page-{index + 1:05d}.md"

    def validate_or_reset(self) -> None:
        """Ensure the cache dir exists and matches ``meta``; wipe it if stale."""
        existing = None
        if self._meta_path.exists():
            try:
                existing = json.loads(self._meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                existing = None
        if existing != self.meta:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._meta_path, self.meta)

    def load_page(self, index: int) -> str | None:
        """Return the cached markdown for a page, or ``None`` if not cached."""
        page = self._page_path(index)
        if page.exists():
            try:
                return page.read_text()
            except OSError:
                return None
        return None

    def save_page(self, index: int, markdown: str) -> None:
        """Atomically checkpoint a completed page."""
        write_text_atomic(self._page_path(index), markdown)

    def cleanup(self) -> None:
        """Remove the whole cache (call once the document is fully assembled)."""
        shutil.rmtree(self.dir, ignore_errors=True)

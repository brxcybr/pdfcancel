"""Page-boundary markers for pdfcancel --preserve-pages.

When --preserve-pages is enabled, an HTML comment marker is injected before
each page's markdown at join time:

    <!-- pdfcancel-page: 3 -->

Markers survive post-OCR cleanup (clean.py treats them as protected lines)
and are consumed by the chunker (chunks.py), which records page_start /
page_end metadata on each chunk and strips the marker comments from the
chunk text itself.
"""

from __future__ import annotations

import bisect
import re

PAGE_MARKER_TEMPLATE = "<!-- pdfcancel-page: {n} -->"

# A marker line: only the comment (plus surrounding spaces) on its own line.
PAGE_MARKER_RE = re.compile(
    r"^[ \t]*<!--\s*pdfcancel-page:\s*(\d+)\s*-->[ \t]*$",
    re.MULTILINE,
)


def page_marker(page_number: int) -> str:
    """Return the marker comment for a 1-indexed page number."""
    return PAGE_MARKER_TEMPLATE.format(n=page_number)


def is_page_marker_line(line: str) -> bool:
    """True if the (single) line is a pdfcancel page marker."""
    return bool(PAGE_MARKER_RE.fullmatch(line.strip()))


def join_pages(page_markdowns: list[str], *, preserve_pages: bool = False) -> str:
    """Join per-page markdown, optionally injecting page markers (1-indexed)."""
    if not preserve_pages:
        return "\n\n".join(page_markdowns)
    parts = []
    for n, page_md in enumerate(page_markdowns, 1):
        parts.append(f"{page_marker(n)}\n\n{page_md}")
    return "\n\n".join(parts)


def build_page_index(text: str) -> list[tuple[int, int]]:
    """Return a sorted list of (char_offset, page_number) for each marker."""
    return [(m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(text)]


def page_range_for_span(
    page_index: list[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    """Return (page_start, page_end) for a character span [start, end).

    Returns (None, None) when the text carries no page markers.
    Content before the first marker is attributed to that marker's page.
    """
    if not page_index:
        return None, None
    offsets = [off for off, _ in page_index]

    # Page active at `start`: last marker at or before start, else the first.
    i = bisect.bisect_right(offsets, start) - 1
    page_start = page_index[max(i, 0)][1]

    # Page active at `end`: last marker strictly before end, else the first.
    j = bisect.bisect_left(offsets, max(end, start)) - 1
    page_end = page_index[max(j, 0)][1]

    return page_start, max(page_end, page_start)


def strip_page_markers(text: str) -> str:
    """Remove page marker lines from text, collapsing leftover blank runs."""
    if "pdfcancel-page" not in text:
        return text
    text = PAGE_MARKER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")

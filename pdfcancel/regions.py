"""Split a page into reading-order regions for separate transcription.

A dense page (a two-column reference list) can send a vision model into a
repetition loop on every attempt. Transcribing it in smaller pieces avoids
that: :func:`page_regions` cuts the page at blank gutters, a two-column run
into its left then right column, and a single-column page into halves. It
works from the rendered pixels, so scans split the same way born-digital
pages do.
"""

from __future__ import annotations

ANALYSIS_DPI = 50
INK = 200            # gray level below which a pixel counts as ink (light fills too)
GAP_ROWS = 4         # blank rows (at ANALYSIS_DPI) that separate bands
GUTTER_COLS = 7      # blank columns that make a column gutter (~10 pt; wider than a word space)
GUTTER_MIN_ROWS = 20  # a band shorter than this (~3 lines) has no columns
MIN_REGION_FRAC = 0.04  # a band shorter than this share of the page has no columns


def _ink_rows(page) -> tuple[list[bytes], int, int]:
    """Rows of 0/1 bytes (1 = ink) for ``page`` rendered in gray."""
    import pymupdf

    pix = page.get_pixmap(dpi=ANALYSIS_DPI, colorspace=pymupdf.csGRAY, alpha=False)
    table = bytes(1 if v < INK else 0 for v in range(256))
    w, h, stride = pix.width, pix.height, pix.stride
    data = pix.samples
    return [data[y * stride:y * stride + w].translate(table) for y in range(h)], w, h


def _runs(flags: list[bool], min_len: int) -> list[tuple[int, int]]:
    """``[(start, end)]`` runs of True at least ``min_len`` long."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def _gutter(rows: list[bytes], w: int, y0: int, y1: int) -> set[int]:
    """The column gutter of rows ``y0:y1``: a blank run in the middle third
    with ink on both sides (a centered figure has blank margins, not columns)."""
    if y1 - y0 < GUTTER_MIN_ROWS:
        return set()
    band = rows[y0:y1]
    lo, hi = int(w * 0.3), int(w * 0.7)
    blank = [lo <= x < hi and not any(r[x] for r in band) for x in range(w)]
    col_ink = [sum(r[x] for r in band) for x in range(w)]
    total = sum(col_ink) or 1
    for s, e in sorted(_runs(blank, GUTTER_COLS), key=lambda r: abs((r[0] + r[1]) / 2 - w / 2)):
        if min(sum(col_ink[:s]), sum(col_ink[e:])) >= 0.15 * total:
            return set(range(s, e))
    return set()


def _wide(cols: set[int], w: int) -> set[int]:
    """The columns of ``cols`` in runs at least GUTTER_COLS wide."""
    return {x for s, e in _runs([x in cols for x in range(w)], GUTTER_COLS) for x in range(s, e)}


def page_regions(page) -> list[tuple[float, float, float, float]]:
    """Reading-order ``(x0, y0, x1, y1)`` regions in page points, or ``[]``.

    Bands separated by blank rows are grouped while they share a column
    gutter; a grouped run yields its left column, then its right. A page with
    no gutter is cut into halves at the blank gap nearest its middle. Returns
    ``[]`` when the page can't be split (one unbroken block).
    """
    rows, w, h = _ink_rows(page)
    inked = [any(r) for r in rows]
    bands = _runs(inked, 1)
    if not bands:
        return []
    # Merge bands closer than GAP_ROWS (lines of one paragraph).
    merged = [list(bands[0])]
    for s, e in bands[1:]:
        if s - merged[-1][1] < GAP_ROWS:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Group consecutive bands that share a gutter.
    groups: list[tuple[int, int, set[int]]] = []
    for s, e in merged:
        g = _gutter(rows, w, s, e)
        shared = _wide(g & groups[-1][2], w) if groups else set()
        if shared:
            groups[-1] = (groups[-1][0], e, shared)
        elif groups and not groups[-1][2] and not g:
            groups[-1] = (groups[-1][0], e, set())
        else:
            groups.append((s, e, g))

    sx, sy = page.rect.width / w, page.rect.height / h
    regions: list[tuple[float, float, float, float]] = []
    for s, e, g in groups:
        if g and e - s > h * MIN_REGION_FRAC:
            mid = (min(g) + max(g)) / 2
            regions += [(0, s * sy, mid * sx, e * sy), (mid * sx, s * sy, page.rect.width, e * sy)]
        else:
            regions.append((0, s * sy, page.rect.width, e * sy))

    # Fold a sliver (a caption or footer line) into the full-width region above it.
    out: list[list[float]] = []
    for r in regions:
        if out and (r[3] - r[1] < h * sy * MIN_REGION_FRAC) and out[-1][0] == r[0] \
                and out[-1][2] == r[2]:
            out[-1][3] = r[3]
        else:
            out.append(list(r))
    if len(out) == 1:
        return _halves(bands, page, sy)
    return [tuple(r) for r in out]


def _halves(bands, page, sy) -> list[tuple[float, float, float, float]]:
    """Cut a single-column page at the blank gap nearest its middle."""
    if len(bands) < 2:
        return []
    middle = (bands[0][0] + bands[-1][1]) / 2
    cut = min(((bands[i][1] + bands[i + 1][0]) / 2 for i in range(len(bands) - 1)),
              key=lambda y: abs(y - middle))
    return [(0, 0, page.rect.width, cut * sy), (0, cut * sy, page.rect.width, page.rect.height)]


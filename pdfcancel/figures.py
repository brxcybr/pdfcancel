"""Local figure extraction for the vlm OCR engine.

Mistral OCR returns server-cropped figures; an OpenAI-compatible vision model
only returns page text, leaving ``![figure](figure)`` placeholders. This module
recovers the figures locally with PyMuPDF so ``--images`` / ``--embed-images``
/ ``--full`` work on the vlm engine without sending anything off-network:

1. Detect figures per page: embedded raster images plus vector drawing
   clusters (most charts), merged into whole figures and grown to take in
   nearby axis/tick/legend labels.
2. Drop non-figures: running headers/footers, logos repeated across pages,
   ruled tables, shaded text boxes, and monospace code/API callouts (the OCR
   already transcribed them).
3. Crop each figure at ``dpi`` and link it into the page's markdown in place of
   the OCR placeholders, as ``![pNNN-fKK.png](pNNN-fKK.png)``. That matches
   the Mistral engine's ``{image_id: bytes}`` + ``![id](id)`` contract, so
   image extraction, embedding, and descriptions work unchanged.

A scanned page (one image covering the page) with a placeholder is
rendered whole as ``pNNN-page.png`` so it still gets a
description.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple

from pdfcancel.pages import PAGE_MARKER_RE

PLACEHOLDER_RE = re.compile(r"!\[figure\]\(figure\)")
_CAPTION = r"(?:fig(?:ure)?|chart|exhibit|plate)\.?\s*(\d+[a-z]?)\b"
CAPTION_RE = re.compile(r"\s*" + _CAPTION, re.IGNORECASE)               # text-layer block
CAPTION_LINE_RE = re.compile(r"^[ \t>*_#]*" + _CAPTION, re.IGNORECASE | re.MULTILINE)
CAPTION_REACH = 40         # pt between a figure and its printed caption

MIN_AREA = 0.02            # of page area
MIN_W, MIN_H = 0.12, 0.05  # of page width / height
FULL_PAGE = 0.85           # an image this large is a scan/background
HEADER_ZONE = 0.07         # top/bottom fraction holding running headers
FRAGMENT_GAP = 10          # pt between graphic fragments of one figure
LABEL_REACH = 18           # pt around a figure in which short lines join it
LABEL_MAX_CHARS = 60
PROSE_LINE_CHARS = 25    # mean chars/line of a paragraph block
TEXTBOX_MIN_CHARS = 100    # text-box test: this much prose ...
TEXTBOX_MAX_PATHS = 6      # ... inside at most this many drawn marks
CODE_BOX_MONO = 0.5        # vector region whose text is mostly monospace = code box
MONO_FONT_RE = re.compile(r"mono|courier|consol|menlo|code|typewriter", re.IGNORECASE)
REPEAT_PAGES = 3           # identical crops on this many pages are logos


def _pymupdf():
    try:
        import pymupdf
    except ImportError as e:  # pragma: no cover - exercised via the CLI
        raise SystemExit(
            "Error: local figure extraction needs PyMuPDF. "
            "Install it with: pip install 'pdfcancel[vlm]'"
        ) from e
    return pymupdf


def _merge(rects: Iterable, gap: float = 6) -> list:
    """Union rectangles that touch or sit within ``gap`` points of each other."""
    pymupdf = _pymupdf()
    rects = [pymupdf.Rect(r) for r in rects]
    merged = True
    while merged:
        merged = False
        out: list = []
        for r in rects:
            for o in out:
                if o.intersects(pymupdf.Rect(r) + (-gap, -gap, gap, gap)):
                    o.include_rect(r)
                    merged = True
                    break
            else:
                out.append(pymupdf.Rect(r))
        rects = out
    return rects


def _has_listing(lines: list, *, margin: float = 2.5) -> bool:
    """True when two or more lines stack at one left margin, line after line."""
    lines = sorted(lines, key=lambda r: (r.y0, r.x0))
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            if b.y0 - a.y1 > 1.5 * a.height:
                break
            if abs(b.x0 - a.x0) <= margin and b.y0 >= a.y1 - 0.5 * a.height:
                return True
    return False


def detect_figures(page, *, include_tables: bool = False) -> tuple[list, bool]:
    """Return ``(figure rects in reading order, page_is_scanned)`` for a page."""
    pymupdf = _pymupdf()
    pr = page.rect
    top, bottom = HEADER_ZONE * pr.height, (1 - HEADER_ZONE) * pr.height
    page_area = pr.get_area()
    scanned = False

    def big_enough(r) -> bool:
        return (r.get_area() >= MIN_AREA * page_area
                and r.width >= MIN_W * pr.width and r.height >= MIN_H * pr.height)

    # Collect every graphic fragment, merge neighbours, then size-filter: one
    # chart is often several small raster tiles plus vector-outlined labels,
    # none of which qualifies on its own.
    images = [pymupdf.Rect(i["bbox"]) & pr for i in page.get_image_info()]
    parts = []
    for r in images:
        if r.get_area() >= FULL_PAGE * page_area:
            scanned = True
        elif not r.is_empty:
            parts.append(r)

    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    tables = []
    if not include_tables:
        try:
            tables = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
        except Exception:
            tables = []
    for r in clusters:
        r = r & pr
        if r.is_empty or r.get_area() >= FULL_PAGE * page_area:
            continue
        if r.y1 <= top or r.y0 >= bottom:  # header/footer rules and logos
            continue
        # Ruled tables are already transcribed as markdown tables by the OCR.
        if any((r & t).get_area() >= 0.6 * r.get_area() for t in tables):
            continue
        parts.append(r)

    rects = [r for r in _merge(parts, gap=FRAGMENT_GAP) if big_enough(r)]

    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    lines = [
        (pymupdf.Rect(ln["bbox"]), "".join(s["text"] for s in ln["spans"]).strip(), b)
        for b in blocks for ln in b["lines"]
    ]

    # Shaded code/spec/callout boxes are vector rectangles full of prose. Charts
    # with long category labels look text-heavy too, but they are drawn from
    # many marks (bars, points); row bands, frames, and rules don't count.
    paths = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]

    def is_text_box(r) -> bool:
        if any((r & i).get_area() >= 0.2 * r.get_area() for i in images):
            return False
        texts = [t for lr, t, _ in lines if t and lr in r]
        chars = sum(map(len, texts))
        long_chars = sum(len(t) for t in texts if len(t) >= 40)
        if chars < TEXTBOX_MIN_CHARS or long_chars < 0.4 * chars:
            return False
        marks = [p for p in paths if p in r and p.width < 0.6 * r.width
                 and min(p.width, p.height) > 1]
        return len(marks) <= TEXTBOX_MAX_PATHS

    # Spec/API documents draw request, header, and JSON callouts as vector
    # boxes with decorative marks (method badges, rules) that defeat the mark
    # count above. Their tell is monospace text laid out as a listing: lines
    # stacked at one left margin. A diagram may label its nodes in monospace,
    # but those labels sit apart, one per node.
    def is_mono(sp: dict) -> bool:
        return bool(sp["flags"] & 8 or MONO_FONT_RE.search(sp["font"]))

    def is_code_box(r) -> bool:
        if any((r & i).get_area() >= 0.2 * r.get_area() for i in images):
            return False
        chars = mono = 0
        mono_lines = []
        for b in blocks:
            for ln in b["lines"]:
                lr = pymupdf.Rect(ln["bbox"])
                if lr not in r:
                    continue
                spans = [sp for sp in ln["spans"] if sp["text"].strip()]
                n = sum(len(sp["text"]) for sp in spans)
                m = sum(len(sp["text"]) for sp in spans if is_mono(sp))
                chars, mono = chars + n, mono + m
                if n and m >= CODE_BOX_MONO * n:
                    mono_lines.append(lr)
        if chars < 20 or mono < CODE_BOX_MONO * chars:
            return False
        return _has_listing(mono_lines)

    rects = [r for r in rects if not is_text_box(r) and not is_code_box(r)]

    # Vector charts keep axis labels, ticks, and legends as text outside the
    # drawing cluster: pull in short nearby lines. Paragraph lines (short in
    # two-column layouts) and running headers/footers are never labels.
    def is_prose(b: dict) -> bool:
        # By line length, not line count: a row of tick labels is one block
        # of many very short lines.
        bw = b["bbox"][2] - b["bbox"][0]
        chars = sum(len(s["text"]) for ln in b["lines"] for s in ln["spans"])
        return bw >= 0.25 * pr.width and chars / len(b["lines"]) >= PROSE_LINE_CHARS

    labels = [
        lr for lr, text, b in lines
        if text and len(text) <= LABEL_MAX_CHARS and lr.y0 >= top and lr.y1 <= bottom
        and not is_prose(b)
    ]
    for r in rects:
        for _ in range(3):  # labels chain: tick -> axis title -> legend
            reach = pymupdf.Rect(r) + (-LABEL_REACH, -LABEL_REACH, LABEL_REACH, LABEL_REACH)
            grow = [lr for lr in labels
                    if lr.intersects(reach) and not r.contains(lr) and lr.width <= 1.1 * r.width]
            if not grow:
                break
            for lr in grow:
                r.include_rect(lr)
    rects = _merge([(r + (-4, -4, 4, 4)) & pr for r in _merge(rects)])

    # Reading order: left column, then right; full-width figures sort as left.
    return sorted(rects, key=lambda r: (1 if r.x0 >= 0.45 * pr.width else 0, r.y0)), scanned


def _split_pages(markdown: str) -> list[tuple[int | None, str]]:
    """``[(page_no, text)]``; each page's text starts with its own marker.

    Text before the first marker is a ``None`` preamble. Concatenating the
    texts reproduces ``markdown`` exactly.
    """
    marks = list(PAGE_MARKER_RE.finditer(markdown))
    starts = [m.start() for m in marks] + [len(markdown)]
    out: list[tuple[int | None, str]] = [(None, markdown[:starts[0]])]
    for m, end in zip(marks, starts[1:]):
        out.append((int(m.group(1)), markdown[m.start():end]))
    return out


def extract_figures(
    pdf_path,
    markdown: str,
    *,
    dpi: int = 200,
    include_tables: bool = False,
) -> tuple[str, dict[str, bytes]]:
    """Crop figures from ``pdf_path`` and link them into page-marked markdown.

    ``markdown`` must carry ``<!-- pdfcancel-page: N -->`` markers. Returns the
    markdown with ``![id](id)`` references and an ``{id: png_bytes}`` mapping.
    """
    pymupdf = _pymupdf()
    if not PAGE_MARKER_RE.search(markdown):
        raise ValueError("extract_figures needs page-marked markdown")

    doc = pymupdf.open(str(pdf_path))
    sections = _split_pages(markdown)
    placeholders = {n: len(PLACEHOLDER_RE.findall(t)) for n, t in sections if n}

    # Render everything first so crops repeated across pages (logos) can be dropped.
    crops: dict[int, list[_Crop]] = {}
    seen: Counter[str] = Counter()
    try:
        for page in doc:
            n = page.number + 1
            rects, scanned = detect_figures(page, include_tables=include_tables)
            crops[n] = [
                _Crop(f"p{n:03d}-f{k:02d}.png", page.get_pixmap(dpi=dpi, clip=r).tobytes("png"),
                      *_crop_info(page, r))
                for k, r in enumerate(rects, 1)
            ]
            if not rects and scanned and placeholders.get(n):
                # A scan has no separable figures: describe the whole page. On a
                # born-digital page, a figure every filter rejected stays a
                # placeholder rather than costing a whole-page render.
                crops[n] = [_Crop(f"p{n:03d}-page.png",
                                  page.get_pixmap(dpi=dpi).tobytes("png"), True, None)]
            for h in {hashlib.sha256(c.png).hexdigest() for c in crops[n]}:
                seen[h] += 1
    finally:
        doc.close()
    repeated = {h for h, c in seen.items() if c >= REPEAT_PAGES}

    images: dict[str, bytes] = {}
    out = []
    for n, text in sections:
        figs = [c for c in crops.get(n, []) if n
                and hashlib.sha256(c.png).hexdigest() not in repeated]
        images.update((c.name, c.png) for c in figs)
        out.append(_place(text, figs))
    return "".join(out), images


class _Crop(NamedTuple):
    name: str
    png: bytes
    pictorial: bool        # mostly raster with no text layer: an OCR placeholder's kind
    number: str | None     # figure number from the caption printed beside it

    @property
    def ref(self) -> str:
        return f"![{self.name}]({self.name})"


def _crop_info(page, r) -> tuple[bool, str | None]:
    """``(pictorial, caption number)`` for a figure rect, from the text layer."""
    pymupdf = _pymupdf()
    raster = sum((pymupdf.Rect(i["bbox"]) & r).get_area() for i in page.get_image_info())
    words = page.get_text("words", clip=r)
    pictorial = raster >= 0.5 * r.get_area() and sum(len(w[4]) for w in words) < 30
    best = None
    for b in page.get_text("blocks"):
        br, text = pymupdf.Rect(b[:4]), b[4]
        m = CAPTION_RE.match(text)
        if not m or br.x1 < r.x0 or br.x0 > r.x1:
            continue
        gap = br.y0 - r.y1 if br.y0 >= r.y1 - 4 else r.y0 - br.y1
        if -4 <= gap <= CAPTION_REACH and (best is None or gap < best[0]):
            best = (gap, m.group(1).lower())
    return pictorial, best and best[1]


def _place(text: str, figs: list[_Crop]) -> str:
    """Link a page's crops into its markdown.

    The OCR prompt asks for placeholders only for purely pictorial images, so
    crops and placeholders need not correspond one-to-one. A crop whose
    printed caption ("Figure 3") appears in the markdown goes at that caption,
    taking the placeholder just above it if there is one. The rest fill the
    remaining placeholders in order when the counts agree; otherwise only
    pictorial crops do. Anything still unplaced is appended at the page's end.
    """
    holders = [m.span() for m in PLACEHOLDER_RE.finditer(text)]
    captions: dict[str, int] = {}
    for m in CAPTION_LINE_RE.finditer(text):
        captions.setdefault(m.group(1).lower(), m.start())
    taken: dict[int, str] = {}          # placeholder index -> ref
    inserts: list[tuple[int, str]] = []  # (offset, ref)
    rest = []
    for c in figs:
        at = captions.get(c.number) if c.number else None
        if at is None:
            rest.append(c)
            continue
        prev = [i for i, (_, end) in enumerate(holders) if end <= at and i not in taken
                and not any(end <= o < at for o in captions.values())]
        if prev:
            taken[prev[-1]] = c.ref
        else:
            inserts.append((at, c.ref + "\n\n"))
    free = [i for i in range(len(holders)) if i not in taken]
    fill = rest if len(free) == len(rest) else [c for c in rest if c.pictorial]
    for i, c in zip(free, fill):
        taken[i] = c.ref
        rest.remove(c)

    edits = [(*holders[i], ref) for i, ref in taken.items()] + [(o, o, s) for o, s in inserts]
    for start, end, s in sorted(edits, reverse=True):
        text = text[:start] + s + text[end:]
    if rest:
        text = text.rstrip("\n") + "\n\n" + "\n\n".join(c.ref for c in rest) + "\n\n"
    return text


"""Consistent heading levels from section numbering.

Pages are transcribed independently, so a model may mark ``3. Methods`` as
``###`` on one page and ``1. Introduction`` as ``#`` on another, or leave a
section heading as a plain line (``2.1. Related surveys``, ``II. METHODOLOGY``).
:func:`normalize_headings` reads the document's own numbering and:

- gives every numbered heading the level its depth implies (``1.`` → ``##``,
  ``1.1`` → ``###``; IEEE ``II.`` → ``##`` and ``B.`` → ``###``);
- promotes a plain standalone line to a heading only when its number is a
  plausible next section (so a numbered list in the body is left alone);
- puts unnumbered headings one level below the section they sit in.

A document with no section numbering is left unchanged by
:func:`normalize_headings`; :func:`font_heading_levels` levels its headings
by the type sizes in the PDF's text layer instead.
"""

from __future__ import annotations

import re
from collections import Counter

from pdfcancel.pages import PAGE_MARKER_RE, is_page_marker_line

_DECIMAL = re.compile(r"(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.*)")
_ROMAN = re.compile(r"([IVX]{1,5})\.\s+(\S.*)")
_LETTER = re.compile(r"([A-H])\.\s+(\S.*)")
_HEADING = re.compile(r"(#{1,6})\s+(.*?)\s*#*\s*")
_WRAP = re.compile(r"(\*\*|__|\*|_)(.+)\1")
_ROMANS = {r: i for i, r in enumerate(
    "I II III IV V VI VII VIII IX X XI XII XIII XIV XV".split(), 1)}
# Unnumbered back-matter and front-matter sections, at the top section level.
_NAMED = re.compile(
    r"(?:abstract|references|bibliography|acknowledge?ments?|appendix(?:\s+\w+)?"
    r"|conclusions?|introduction|related work|keywords|index terms)",
    re.IGNORECASE)
MIN_NUMBERED = 3   # fewer numbered headings than this: no scheme to normalize
MAX_TITLE_WORDS = 15


def _plain(line: str) -> tuple[int, str]:
    """``(current heading level or 0, text without # and emphasis)``."""
    level, text = 0, line.strip()
    m = _HEADING.fullmatch(text)
    if m:
        level, text = len(m.group(1)), m.group(2)
    m = _WRAP.fullmatch(text)
    if m:
        text = m.group(2).strip()
    return level, text


def _title_ok(title: str, *, caps: bool = False) -> bool:
    words = title.split()
    if not words or len(words) > MAX_TITLE_WORDS or len(title) > 120:
        return False
    if title.rstrip()[-1] in ".,;" or not (title[0].isupper() or title[0].isdigit()):
        return False
    if caps:  # an IEEE section title is set in capitals
        letters = [c for c in title if c.isalpha()]
        return sum(c.isupper() for c in letters) >= 0.8 * len(letters)
    return True


def _parse(text: str) -> tuple[str, tuple[int, ...]] | None:
    """``(scheme, number)`` for a numbered heading text, else ``None``."""
    m = _ROMAN.fullmatch(text)
    if m and m.group(1) in _ROMANS:
        return "roman", (_ROMANS[m.group(1)],)
    m = _LETTER.fullmatch(text)
    if m:
        return "letter", (ord(m.group(1)) - 64,)
    m = _DECIMAL.fullmatch(text)
    if m:
        num = tuple(int(p) for p in m.group(1).split("."))
        if all(0 < p <= 40 for p in num):
            return "decimal", num
    return None


def _next_ok(prev: tuple[int, ...] | None, num: tuple[int, ...]) -> bool:
    """Is ``num`` a plausible section after ``prev`` (allowing one skipped)?"""
    if prev is None:
        return num[-1] == 1 and all(p <= 2 for p in num)
    k = len(num)
    if k <= len(prev):
        return num[:k - 1] == prev[:k - 1] and prev[k - 1] < num[-1] <= prev[k - 1] + 2
    return k == len(prev) + 1 and num[:-1] == prev and num[-1] in (1, 2)


def _blocks(lines: list[str]) -> list[int]:
    """Indexes of standalone lines: blank (or a page marker) on both sides, outside code."""
    blank = [not ln.strip() or is_page_marker_line(ln.strip()) for ln in lines]
    out, fenced = [], False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or blank[i] or ln.lstrip()[:1] in ("|", ">", "-", "!", "<"):
            continue
        before = i == 0 or blank[i - 1]
        after = i == len(lines) - 1 or blank[i + 1]
        if before and after or _HEADING.fullmatch(ln.strip()):
            out.append(i)
    return out


def _find(lines: list[str]) -> list[tuple[int, int, str, tuple[int, ...] | None]]:
    """``[(line, level, scheme, number)]`` for headings and numbered section lines."""
    found: list[tuple[int, int, str, tuple[int, ...] | None]] = []
    last: dict[str, tuple[int, ...] | None] = {"decimal": None, "roman": None, "letter": None}
    for i in _blocks(lines):
        level, text = _plain(lines[i])
        parsed = _parse(text)
        if parsed is None:
            if _NAMED.fullmatch(text) and (level or text.isupper() or text.istitle()):
                found.append((i, level, "named", None))
            elif level:
                found.append((i, level, "", None))
            continue
        scheme, num = parsed
        title = re.sub(r"^\S+\s+", "", text)
        if scheme == "letter" and last["roman"] is None:
            scheme, num = "decimal", None  # "A. Smith" outside an IEEE paper
        if num is None:
            if level:
                found.append((i, level, "", None))
            continue
        if not level and not (_title_ok(title, caps=scheme == "roman")
                              and _next_ok(last[scheme], num)):
            continue
        if scheme == "roman":
            last["letter"] = None
        last[scheme] = num
        found.append((i, level, scheme, num))
    return found


def is_numbered(markdown: str) -> bool:
    """True when the document numbers its sections (normalize_headings applies)."""
    return sum(1 for f in _find(markdown.split("\n")) if f[3] is not None) >= MIN_NUMBERED


def normalize_headings(markdown: str) -> str:
    """Heading levels from section numbering (see the module docstring)."""
    lines = markdown.split("\n")
    found = _find(lines)
    if sum(1 for f in found if f[3] is not None) < MIN_NUMBERED:
        return markdown

    section = 1  # level of the enclosing section
    started = False
    for i, level, scheme, num in found:
        text = _plain(lines[i])[1]
        if scheme == "decimal":
            new = min(len(num) + 1, 6)
        elif scheme == "roman":
            new = 2
        elif scheme == "letter":
            new = 3
        elif scheme == "named":
            new = 2
        elif not started:
            new = level  # front matter: the title and anything before section 1
        else:
            new = min(section + 1, 6)
        if scheme:
            started, section = True, new
        lines[i] = "#" * new + " " + text
    return "\n".join(lines)


# ── Unnumbered documents: levels from font sizes ─────────────────────────────

FONT_MIN_HEADINGS = 5   # fewer headings than this: leave the model's levels
SIZE_TIER = 0.75        # pt: sizes this close are one heading level
_KEY = re.compile(r"[^a-z0-9]+")


def _key(text: str) -> str:
    """Letters and digits only, for matching a heading to the text layer."""
    import unicodedata

    return _KEY.sub("", unicodedata.normalize("NFKC", text).lower())


def _bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _page_lines(page) -> list[tuple[str, float, bool]]:
    """``[(key, size, bold)]`` for each text line of ``page``; none for a scan.

    A scan's OCR text layer is drawn invisible (render mode 3) under the page
    image, and its type sizes are estimates, so it gives no evidence.
    """
    trace = page.get_texttrace()
    chars = sum(len(sp["chars"]) for sp in trace)
    if chars and sum(len(sp["chars"]) for sp in trace if sp["type"] == 3) >= chars / 2:
        return []
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [sp for sp in line["spans"] if sp["text"].strip()]
            if spans:
                big = max(spans, key=lambda sp: sp["size"])
                out.append((_key("".join(sp["text"] for sp in spans)),
                            round(big["size"] * 2) / 2, any(_bold(sp) for sp in spans)))
    return out


def _style(key: str, lines: list[tuple[str, float, bool]], body: float) -> float | None:
    """Font size of the heading ``key`` on its page, or ``None`` without evidence.

    The largest matching line wins: a heading also appears in running headers
    and the table of contents in smaller type. A heading wrapped over lines
    matches when consecutive lines join to exactly its text; a bold run-in
    heading matches the line it starts.
    """
    best = None
    for j, (k, size, bold) in enumerate(lines):
        if not k:
            continue
        if not (k == key or (len(key) >= 8 and k.startswith(key))):
            if not key.startswith(k):
                continue
            for k2, s2, b2 in lines[j + 1:j + 4]:  # a wrapped heading
                k, size, bold = k + k2, max(size, s2), bold or b2
                if len(k) >= len(key):
                    break
            if k != key:
                continue
        if (size > body or (bold and size >= body)) and (best is None or size > best):
            best = size
    return best


def font_heading_levels(markdown: str, pdf_path) -> str:
    """Heading levels from the PDF's font sizes, for unnumbered documents.

    Without numbering, each page's levels are the model's guess from that page
    alone (a book's every section came out as ``#``). On a born-digital PDF,
    the type size of each heading in the text layer (sizes within
    ``SIZE_TIER`` of each other counted as one) ranks them document-wide:
    the largest size used by two or more headings is ``#``, the next ``##``,
    and so on; a one-off size takes the level of the next larger size, and a
    one-off larger than all of them (a title) is ``#`` above the rest. A
    heading missing from its page's text layer (text in a figure) is kept
    out of the top level. Needs page markers; numbered documents, scans
    (including scans with an OCR text layer), and documents with few
    headings are returned unchanged.
    """
    if is_numbered(markdown) or not PAGE_MARKER_RE.search(markdown):
        return markdown
    lines = markdown.split("\n")
    heads: list[tuple[int, int, int, str]] = []  # (line, page, level, key)
    page, fenced = 0, False
    for i, ln in enumerate(lines):
        m = PAGE_MARKER_RE.fullmatch(ln.strip())
        if m:
            page, fenced = int(m.group(1)), False
            continue
        if ln.lstrip().startswith("```"):
            fenced = not fenced
        if fenced:
            continue  # "# comment" in a code sample is not a heading
        level, text = _plain(ln)
        if level and page and _key(text):
            heads.append((i, page, level, _key(text)))
    if len(heads) < FONT_MIN_HEADINGS:
        return markdown

    import pymupdf

    sizes: dict[int, float | None] = {}
    with pymupdf.open(str(pdf_path)) as doc:
        chars: Counter[float] = Counter()
        cache: dict[int, list] = {}
        for pg in sorted({h[1] for h in heads if h[1] <= doc.page_count}):
            cache[pg] = _page_lines(doc[pg - 1])
            for k, size, _ in cache[pg]:
                chars[size] += len(k)
        if not chars:
            return markdown  # a scan: no text layer to read sizes from
        body = chars.most_common(1)[0][0]
        for i, pg, _, key in heads:
            if cache.get(pg):  # pages without a text layer give no evidence
                sizes[i] = _style(key, cache[pg], body)

    tier: dict[float, float] = {}
    for size in sorted({s for s in sizes.values() if s is not None}, reverse=True):
        top = next(iter(reversed(tier.values())), None)
        tier[size] = top if top is not None and top - size <= SIZE_TIER else size
    sizes = {i: s if s is None else tier[s] for i, s in sizes.items()}
    counts = Counter(s for s in sizes.values() if s is not None)
    ranked = sorted((s for s, c in counts.items() if c >= 2), reverse=True)
    if len(ranked) < 2:
        return markdown
    offset = 1 if any(s > ranked[0] for s in counts) else 0
    rank = {s: k + 1 + offset for k, s in enumerate(ranked)}

    def level_for(size: float) -> int:
        if size in rank:
            return rank[size]
        larger = [s for s in ranked if s > size]
        return rank[larger[-1]] if larger else 1

    for i, _, level, _ in heads:
        if i not in sizes:
            continue
        size = sizes[i]
        new = level_for(size) if size is not None else max(level, offset + 2)
        lines[i] = "#" * min(new, 6) + " " + _plain(lines[i])[1]
    return "\n".join(lines)

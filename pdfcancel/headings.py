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

A document with no section numbering is left unchanged.
"""

from __future__ import annotations

import re

from pdfcancel.pages import is_page_marker_line

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


def normalize_headings(markdown: str) -> str:
    """Heading levels from section numbering (see the module docstring)."""
    lines = markdown.split("\n")
    found: list[tuple[int, int, str, tuple[int, ...] | None]] = []  # (line, level, scheme, num)
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

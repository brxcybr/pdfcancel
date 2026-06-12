"""Post-OCR cleanup for pdfcancel.

Strips page headers, footers, page numbers, download watermarks, and other
artifacts that Mistral OCR faithfully reproduces from PDF pages. Also rejoins
sentences that were broken across page boundaries.

This runs automatically after OCR unless --no-clean is passed.
"""

from __future__ import annotations

import re
from collections import Counter

from pdfcancel.pages import is_page_marker_line


def clean_markdown(text: str) -> str:
    """Apply all cleanup passes to OCR markdown output.

    Order matters: classification banners first (before repeating-header
    detection removes them without preserving the marking), then the rest.
    """
    text = _handle_classification_markings(text)
    text = _strip_download_watermarks(text)
    text = _strip_repeating_headers_footers(text)
    text = _strip_bare_page_numbers(text)
    text = _rejoin_broken_sentences(text)
    text = _collapse_blank_lines(text)
    return text


# ---------------------------------------------------------------------------
# Pass 0: Classification markings
# ---------------------------------------------------------------------------

# Classification banner patterns (standalone lines at top/bottom of pages).
# These are the marking levels from DoD/IC marking standards.
_CLASSIFICATION_BANNERS = re.compile(
    r"^\s*"
    r"(?:"
    r"UNCLASSIFIED(?:\s*//\s*(?:FOUO|FOR OFFICIAL USE ONLY|CUI|NOFORN|REL TO|LIMDIS)[\w\s/,]*)?"
    r"|CUI"
    r"|CONTROLLED UNCLASSIFIED INFORMATION"
    r"|FOUO"
    r"|FOR OFFICIAL USE ONLY"
    r"|CONFIDENTIAL(?:\s*//\s*[\w\s/,]*)?"
    r"|SECRET(?:\s*//\s*(?:NOFORN|REL TO|LIMDIS|ORCON)[\w\s/,]*)?"
    r"|TOP SECRET(?:\s*//\s*(?:SCI|NOFORN|SI|TK|HCS|ORCON)[\w\s/,]*)?"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# "Page N of M" patterns common in government documents.
_PAGE_N_OF_M_RE = re.compile(
    r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _handle_classification_markings(text: str) -> str:
    """Detect classification banner markings, strip per-page repeats, and
    consolidate into a single notice at the top of the document.

    Government/military documents typically have classification banners at the
    top and bottom of every page (e.g., "CUI", "UNCLASSIFIED//FOUO",
    "SECRET//NOFORN"). Portion markings like "(U)" or "(CUI)" at the start
    of paragraphs are LEFT IN PLACE — they are content-level markers.

    This pass:
    1. Finds all classification banner lines
    2. Determines the highest classification level present
    3. Strips all banner lines from the body
    4. Strips "Page N of M" lines
    5. Inserts a single classification notice at the top
    """
    # Find all banner matches
    banners = _CLASSIFICATION_BANNERS.findall(text)
    if not banners:
        # Still strip "Page N of M" even if no classification banners
        text = _PAGE_N_OF_M_RE.sub("", text)
        return text

    # Determine the highest classification from the banners found
    classification = _determine_classification(banners)

    # Strip all banner lines from the body
    text = _CLASSIFICATION_BANNERS.sub("", text)

    # Strip "Page N of M" lines
    text = _PAGE_N_OF_M_RE.sub("", text)

    # Insert classification notice at the top of the document
    notice = f"> **Classification:** {classification}\n"
    text = notice + "\n" + text.lstrip("\n")

    return text


def _determine_classification(banners: list[str]) -> str:
    """Return the highest classification level found in the banner lines.

    Hierarchy: TOP SECRET > SECRET > CONFIDENTIAL > CUI/FOUO > UNCLASSIFIED
    Returns the full marking string of the highest level found.
    """
    # Normalize and deduplicate
    normalized = set()
    for b in banners:
        normalized.add(b.strip().upper())

    # Check from highest to lowest
    for marking in sorted(normalized, key=_classification_rank, reverse=True):
        return marking  # Return the highest-ranked one

    return "UNCLASSIFIED"


def _classification_rank(marking: str) -> int:
    """Return a numeric rank for sorting classification levels."""
    m = marking.upper()
    if m.startswith("TOP SECRET"):
        return 5
    if m.startswith("SECRET"):
        return 4
    if m.startswith("CONFIDENTIAL"):
        return 3
    if any(m.startswith(x) for x in ("CUI", "CONTROLLED", "FOUO", "FOR OFFICIAL")):
        return 2
    if m.startswith("UNCLASSIFIED"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Pass 1: Download watermarks
# ---------------------------------------------------------------------------

# Matches lines like:
#   "Downloaded by University of South Florida At 14:39 20 October 2015 (PT)"
#   "Downloaded by [Institutional Name] At HH:MM DD Month YYYY (TZ)"
_WATERMARK_RE = re.compile(
    r"^Downloaded by .+$",
    re.MULTILINE,
)


def _strip_download_watermarks(text: str) -> str:
    """Remove institutional download watermark lines."""
    return _WATERMARK_RE.sub("", text)


# ---------------------------------------------------------------------------
# Pass 2: Repeating headers and footers
# ---------------------------------------------------------------------------

def _strip_repeating_headers_footers(text: str) -> str:
    """Detect and remove lines that repeat 3+ times and look like page headers/footers.

    Targets patterns like:
      - "ICS 23,3" / "ICSC23,3" (journal abbreviation + volume)
      - "Strategic cyber intelligence" (running header matching paper title)
      - "A. Zibak et al." (author abbreviation)
      - "Digital Threats: Research and Practice, Vol. 3, No. 4, Article 44. ..."
      - "AIBDF 2024, December 27-29, 2024, Ganzhou, China"
      - "Jiyuan Fang et al."
      - Conference/venue lines that repeat on every page

    Heuristic: any line that appears 3+ times, is ≤120 chars, is NOT a markdown
    heading, list item, or table row, and is NOT inside a code block, is likely
    a page header/footer artifact.
    """
    lines = text.split("\n")
    # Count exact occurrences of each line (stripped)
    line_counts: Counter[str] = Counter()
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] += 1

    # Also count normalized forms (collapse whitespace, strip trailing periods)
    # so that "AIBDF 2024, Ganzhou, China" and
    # "AIBDF 2024, December 27-29, 2024, Ganzhou, China" can be grouped
    # when they share a common prefix.
    normalized_counts: Counter[str] = Counter()
    norm_to_originals: dict[str, set[str]] = {}
    for line_text, count in line_counts.items():
        # Normalize: take first 20 chars as a prefix key for short-prefix grouping
        norm = line_text[:20].strip()
        normalized_counts[norm] += count
        norm_to_originals.setdefault(norm, set()).add(line_text)

    # Build set of lines to remove
    artifacts: set[str] = set()
    for line_text, count in line_counts.items():
        # Short lines (≤20 chars) are likely journal abbrevs / page markers;
        # use a lower threshold of 2. Longer lines need 3+ repeats.
        min_count = 2 if len(line_text) <= 20 else 3
        # Check if normalized prefix group collectively hits threshold
        norm = line_text[:20].strip()
        norm_count = normalized_counts.get(norm, 0)
        if count < min_count and norm_count < 3:
            continue
        # Skip markdown structural elements and HTML comments (incl. the
        # pdfcancel page markers, which share a common 20-char prefix and
        # would otherwise be grouped as a "repeating header")
        if line_text.startswith(("#", "-", "*", "|", ">", "```", "![", "[", "<!--")):
            continue
        # Skip lines that are actual content (long prose paragraphs)
        if len(line_text) > 150:
            continue
        # Skip lines that look like reference entries (Author (Year))
        if re.match(r"^[A-Z][a-z]+.*\(\d{4}\)", line_text):
            continue
        # This is likely a repeating header/footer
        artifacts.add(line_text)

    if not artifacts:
        return text

    cleaned = []
    for line in lines:
        if line.strip() in artifacts:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Pass 3: Bare page numbers
# ---------------------------------------------------------------------------

# Lines that are just a number (1-4 digits), possibly with whitespace.
# These are page numbers inserted by OCR between page content.
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Lines that are Roman numerals (common in front matter): i, ii, iii, iv, ...
_ROMAN_NUM_RE = re.compile(r"^\s*[ivxlcdm]+\s*$", re.IGNORECASE)


def _strip_bare_page_numbers(text: str) -> str:
    """Remove lines that are just bare page numbers.

    Careful not to remove numbered list items — those have trailing content
    or are preceded by other list items. We only remove lines where the
    ENTIRE line (after stripping) is just a number.
    """
    lines = text.split("\n")
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _PAGE_NUM_RE.match(stripped) or _ROMAN_NUM_RE.match(stripped):
            # Check context: a page number is typically near a blank line
            # (before or after). If both neighbors are non-blank content,
            # it's more likely part of a list or table — keep it.
            prev_blank = (i == 0) or (not lines[i - 1].strip())
            next_blank = (i == len(lines) - 1) or (not lines[i + 1].strip())
            # Also check if neighbors are headings (page numbers often
            # appear right before/after section breaks)
            prev_heading = (i > 0) and lines[i - 1].strip().startswith("#")
            next_heading = (i < len(lines) - 1) and lines[i + 1].strip().startswith("#")
            if prev_blank or next_blank or prev_heading or next_heading:
                continue  # Skip this line (it's a page number)
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Pass 4: Rejoin broken sentences
# ---------------------------------------------------------------------------

def _rejoin_broken_sentences(text: str) -> str:
    """Rejoin sentences that were split across page boundaries.

    After stripping page artifacts, we may have patterns like:

        ...the analyst defines the parameters of the target event (or kind of

        preceded those events, especially those that were "necessary" conditions...

    The heuristic: if a non-blank line ends WITHOUT sentence-ending punctuation
    (.!?:) and the next non-blank line starts with a lowercase letter, join them.
    This is conservative — it won't merge headings, list items, or new paragraphs.
    """
    lines = text.split("\n")
    result: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # If this line is blank, a heading, list item, table, image, or a
        # page marker comment — just keep it (markers must stay on their
        # own line, never merged into a sentence)
        if (
            not stripped
            or stripped.startswith(("#", "-", "*", "|", ">", "```", "!["))
            or stripped.startswith(("(", "["))
            or is_page_marker_line(stripped)
        ):
            result.append(line)
            i += 1
            continue

        # Check if this line looks like it was cut mid-sentence:
        # - Does NOT end with sentence-ending punctuation
        # - Next non-blank line starts with lowercase
        if not re.search(r"[.!?:;]\s*$", stripped):
            # Find next non-blank line
            j = i + 1
            blanks_between = 0
            while j < len(lines) and not lines[j].strip():
                blanks_between += 1
                j += 1

            if (
                j < len(lines)
                and blanks_between <= 2  # At most 2 blank lines (page gap)
                and lines[j].strip()
                and lines[j].strip()[0].islower()
                # Don't merge into list items or structural elements
                and not lines[j].strip().startswith(("-", "*", "|", ">", "```"))
            ):
                # Join: current line + space + next line (trimmed)
                joined = stripped + " " + lines[j].strip()
                result.append(joined)
                i = j + 1
                continue

        result.append(line)
        i += 1

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Pass 5: Collapse excessive blank lines
# ---------------------------------------------------------------------------

def _collapse_blank_lines(text: str) -> str:
    """Reduce runs of 3+ blank lines to 2 (one visual paragraph break)."""
    return re.sub(r"\n{4,}", "\n\n\n", text)

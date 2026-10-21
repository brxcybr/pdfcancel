"""Chunking module for pdfcancel --chunks mode.

Splits cleaned markdown into structured chunks for RAG/LLM pipelines.
Uses Chonkie's chunkers with markdown-aware splitting rules.

Figure-aware: images, their descriptions, and captions are detected as
atomic "figure blocks" that are never split across chunk boundaries.
Outputs JSONL with rich metadata (source, section breadcrumb, chunk index,
content_type).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console

from pdfcancel.charts import extract_chart_metadata
from pdfcancel.pages import (
    build_page_index,
    page_range_for_span,
    strip_page_markers,
)

console = Console()


# ---------------------------------------------------------------------------
# Figure block detection
# ---------------------------------------------------------------------------

# A figure block is an image reference, optionally followed by a description
# blockquote and/or a caption line. We detect these and protect them from
# being split by the chunker.
#
# Pattern matches:
#   ![alt](path)                          ← image ref
#   > **Figure description:** ...         ← optional description (our injected block)
#   Figure N: caption text...             ← optional caption line
#
# The entire block is replaced with a single-line sentinel before chunking,
# then expanded back after chunking.

_FIGURE_BLOCK_RE = re.compile(
    r"("
    r"!\[[^\]]*\]\([^)]+\)"                  # image reference
    r"(?:\n>.*)*"                             # optional description/data blockquote
    r"(?:\n(?:Figure|Fig\.?)\s*\d*[^\n]*)?"    # optional caption
    r")",
    re.MULTILINE,
)

_SENTINEL_PREFIX = "\x00FIGBLOCK:"
_SENTINEL_RE = re.compile(r"\x00FIGBLOCK:\d+\x00")


def _protect_figure_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Replace figure blocks with single-line sentinels.

    Returns (modified_text, {sentinel_id: original_block}).
    """
    blocks: dict[str, str] = {}
    counter = 0

    def _replace(match: re.Match) -> str:
        nonlocal counter
        block = match.group(0)
        # Only protect multi-line blocks (image + description/caption)
        if "\n" not in block:
            return block
        sentinel_id = f"{_SENTINEL_PREFIX}{counter}\x00"
        blocks[sentinel_id] = block
        counter += 1
        return sentinel_id

    protected = _FIGURE_BLOCK_RE.sub(_replace, text)
    return protected, blocks


def _restore_figure_blocks(text: str, blocks: dict[str, str]) -> str:
    """Replace sentinels back with original figure blocks."""
    for sentinel_id, original in blocks.items():
        text = text.replace(sentinel_id, original)
    return text


def _clean_metadata_text(text: str) -> str:
    """Remove protected-block sentinels and control chars from metadata strings."""
    text = _SENTINEL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"FIGBLOCK:\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _classify_content(text: str) -> str:
    """Classify a chunk's content type based on heuristics."""
    stripped = text.strip()
    if "**Figure description:**" in text or _FIGURE_BLOCK_RE.search(text):
        return "figure"
    if stripped.startswith("---\n"):
        return "frontmatter"
    if stripped.startswith("> **Classification:"):
        return "frontmatter"
    # Check for references section (lines starting with [N] or Author (Year))
    ref_lines = re.findall(r"^(?:- \[\d+\]|[A-Z][a-z]+.*\(\d{4}\))", stripped, re.MULTILINE)
    if ref_lines and len(ref_lines) >= 2:
        return "references"
    # Check for abstract
    if re.search(r"^#+ Abstract", stripped, re.MULTILINE | re.IGNORECASE):
        return "abstract"
    # Check if mostly table rows
    lines = [line for line in stripped.split("\n") if line.strip()]
    if (
        lines
        and sum(1 for line in lines if line.strip().startswith("|")) > len(lines) * 0.5
    ):
        return "table"
    return "prose"


# ---------------------------------------------------------------------------
# Document-level metadata extraction
# ---------------------------------------------------------------------------

def _extract_doc_metadata(text: str, source_file: str) -> dict[str, str]:
    """Extract document-level metadata from the markdown content.

    Checks for YAML frontmatter first (injected by Zotero plugin),
    then falls back to heuristic extraction from the first ~2000 chars.

    Returns a dict with keys: doc_title, doc_author, doc_year, doc_doi.
    All values are strings; empty string if not found.
    """
    meta: dict[str, str] = {
        "doc_title": "",
        "doc_author": "",
        "doc_year": "",
        "doc_doi": "",
    }

    # Try YAML frontmatter first (from Zotero sync)
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            fm = text[4:end]
            for line in fm.split("\n"):
                if line.startswith("title:"):
                    meta["doc_title"] = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("authors:"):
                    raw = line.split(":", 1)[1].strip().strip("[]")
                    # Take first author's last name
                    first = raw.split(",")[0].strip()
                    meta["doc_author"] = first
                elif line.startswith("year:"):
                    meta["doc_year"] = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("doi:"):
                    meta["doc_doi"] = line.split(":", 1)[1].strip().strip('"')
            if meta["doc_title"]:
                return meta

    # Heuristic: extract from first ~2000 chars of markdown
    head = text[:2000]

    # Title: first H1 heading
    title_match = re.search(r"^# (.+)$", head, re.MULTILINE)
    if title_match:
        meta["doc_title"] = title_match.group(1).strip()

    # Author: lines between title and abstract that look like names
    # (short lines with capitalized words, no markdown syntax)
    if title_match:
        after_title = head[title_match.end():]
        for line in after_title.split("\n")[:10]:
            line = line.strip()
            if not line or line.startswith(("#", "!", "|", "-", "*", ">")):
                continue
            # Looks like a name: short, capitalized, no special chars
            if len(line) < 60 and re.match(r"^[A-Z][a-z]+ ", line):
                meta["doc_author"] = line.split(",")[0].split(" and ")[0].strip()
                break

    # Year: look for 4-digit year in common patterns
    year_match = re.search(
        r"(?:(?:19|20)\d{2})",
        head[:500],
    )
    if year_match:
        meta["doc_year"] = year_match.group(0)

    # DOI: look for DOI pattern
    doi_match = re.search(r"10\.\d{4,}/[^\s]+", head)
    if doi_match:
        meta["doc_doi"] = doi_match.group(0).rstrip(".),")

    # Fallback title from source filename (Author - Year - Title.pdf)
    if not meta["doc_title"]:
        stem = Path(source_file).stem
        parts = stem.split(" - ")
        if len(parts) >= 3:
            meta["doc_title"] = parts[-1].strip()
            meta["doc_author"] = meta["doc_author"] or parts[0].strip()
            meta["doc_year"] = meta["doc_year"] or parts[1].strip()

    return meta


# ---------------------------------------------------------------------------
# Main chunking entry point
# ---------------------------------------------------------------------------

def chunk_markdown(
    markdown_content: str,
    *,
    source_file: str,
    chunk_size: int = 1024,
    chunker_type: str = "recursive",
) -> list[dict[str, Any]]:
    """Split markdown into chunks with metadata.

    Figure blocks (image + description + caption) are kept atomic — they
    will never be split across chunk boundaries. Each chunk receives
    document-level metadata (title, author, year, DOI) for filtered search.

    Args:
        markdown_content: The cleaned markdown text.
        source_file: Original PDF filename for metadata.
        chunk_size: Maximum tokens per chunk.
        chunker_type: One of "recursive", "semantic", "sentence", "hierarchical".

    Returns:
        List of chunk dicts with "text", "metadata" keys.
    """
    if chunker_type == "hierarchical":
        return _chunk_markdown_hierarchical(
            markdown_content,
            source_file=source_file,
            chunk_size=chunk_size,
        )

    # Extract document-level metadata once
    doc_meta = _extract_doc_metadata(markdown_content, source_file)

    # Phase 1: Protect figure blocks from splitting
    protected_text, figure_blocks = _protect_figure_blocks(markdown_content)

    # Phase 2: Chunk the protected text
    try:
        chunker = _create_chunker(chunker_type, chunk_size)
    except ImportError as e:
        raise SystemExit(f"Error: {e}") from e
    raw_chunks = chunker(protected_text)

    # Phase 3: Restore figure blocks and build metadata
    heading_index = _build_heading_index(markdown_content)
    page_index = build_page_index(protected_text)

    results = []
    for idx, chunk in enumerate(raw_chunks):
        # Restore any sentinels in this chunk
        text = _restore_figure_blocks(chunk.text, figure_blocks)

        # Map back to original position for section lookup
        section = _section_at_offset(heading_index, chunk.start_index)

        # Page tracking (--preserve-pages): record the page range covered
        # by this chunk, then strip the marker comments from the text
        page_start, page_end = page_range_for_span(
            page_index, chunk.start_index, chunk.end_index,
        )
        text = strip_page_markers(text)

        # Classify content type
        content_type = _classify_content(text)
        chart_meta = extract_chart_metadata(text)

        page_meta = (
            {"page_start": page_start, "page_end": page_end}
            if page_start is not None
            else {}
        )
        results.append({
            "text": text,
            "metadata": {
                "source": source_file,
                "chunk_index": idx,
                "section": section,
                "content_type": content_type,
                "has_figure": content_type == "figure" or "![" in text,
                "token_count": chunk.token_count,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                **page_meta,
                **doc_meta,
                **chart_meta,
            },
        })

    return results


def write_chunks_jsonl(chunks: list[dict], output_path: Path) -> Path:
    """Write chunks to a JSONL file (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return output_path


def _create_chunker(chunker_type: str, chunk_size: int) -> Any:
    """Instantiate the appropriate Chonkie chunker."""
    if chunker_type == "recursive":
        from chonkie import RecursiveChunker
        try:
            return RecursiveChunker.from_recipe("markdown", chunk_size=chunk_size)
        except (AttributeError, Exception):
            # Fallback if from_recipe not available in this version
            return RecursiveChunker(chunk_size=chunk_size)
    elif chunker_type == "semantic":
        try:
            from chonkie import SemanticChunker
        except ImportError as e:
            raise ImportError(
                "Semantic chunking requires the optional chonkie[semantic] extra.\n"
                '  pip install "pdfcancel[chunking]"'
            ) from e
        return SemanticChunker(chunk_size=chunk_size)
    elif chunker_type == "sentence":
        from chonkie import SentenceChunker
        return SentenceChunker(chunk_size=chunk_size)
    else:
        raise SystemExit(
            f"Error: Unknown chunker type '{chunker_type}'.\n"
            "  Supported: recursive, semantic, sentence, hierarchical"
        )


# Matches markdown headings: # Title, ## Section, ### Subsection, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _build_heading_index(text: str) -> list[tuple[int, int, str]]:
    """Build a sorted list of (char_offset, level, heading_text).

    Used to look up the section breadcrumb for any position in the document.
    """
    headings = []
    for match in _HEADING_RE.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append((match.start(), level, title))
    return headings


def _section_at_offset(
    heading_index: list[tuple[int, int, str]],
    offset: int,
) -> str:
    """Return the section breadcrumb string for a given character offset.

    Example: "Introduction > Risk Management > Strategic Metrics"
    """
    if not heading_index:
        return ""

    # Find all headings that precede this offset
    active: dict[int, str] = {}  # level → heading text
    for pos, level, title in heading_index:
        if pos > offset:
            break
        # Set this level's heading and clear any deeper levels
        active[level] = title
        for deeper in list(active.keys()):
            if deeper > level:
                del active[deeper]

    if not active:
        return ""

    # Build breadcrumb from shallowest to deepest
    return " > ".join(active[k] for k in sorted(active.keys()))


def _stable_id(source_file: str, *parts: object) -> str:
    raw = "::".join([source_file, *(str(p) for p in parts)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _heading_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _section_blocks(text: str) -> list[dict[str, Any]]:
    headings = _build_heading_index(text)
    if not headings:
        return [
            {
                "start": 0,
                "end": len(text),
                "level": 0,
                "title": "",
                "path": "",
                "path_ids": [],
            }
        ]

    blocks: list[dict[str, Any]] = []
    active_titles: dict[int, str] = {}
    active_ids: dict[int, str] = {}

    for idx, (start, level, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        clean_title = _clean_metadata_text(title)
        active_titles[level] = clean_title
        active_ids[level] = _heading_slug(clean_title)
        for deeper in list(active_titles):
            if deeper > level:
                del active_titles[deeper]
                del active_ids[deeper]
        ordered_levels = sorted(k for k in active_titles if k <= level)
        blocks.append(
            {
                "start": start,
                "end": end,
                "level": level,
                "title": clean_title,
                "path": " > ".join(active_titles[k] for k in ordered_levels),
                "path_ids": [active_ids[k] for k in ordered_levels],
            }
        )
    return blocks


def _chunk_markdown_hierarchical(
    markdown_content: str,
    *,
    source_file: str,
    chunk_size: int,
) -> list[dict[str, Any]]:
    doc_meta = _extract_doc_metadata(markdown_content, source_file)
    protected_text, figure_blocks = _protect_figure_blocks(markdown_content)
    try:
        chunker = _create_chunker("semantic", chunk_size)
    except ImportError as e:
        console.print(
            f"  [yellow]Warning: {e}[/yellow]\n"
            "  [yellow]Falling back to recursive chunking for hierarchical "
            "mode.[/yellow]"
        )
        chunker = _create_chunker("recursive", chunk_size)
    page_index = build_page_index(protected_text)

    results: list[dict[str, Any]] = []
    global_idx = 0
    for section_idx, section in enumerate(_section_blocks(protected_text)):
        section_text = protected_text[section["start"] : section["end"]].strip()
        if not section_text:
            continue
        parent_id = _stable_id(source_file, "section", section_idx, section["path"])

        raw_chunks = chunker(section_text)
        section_child_ids: list[str] = []
        start_result_idx = len(results)
        for local_idx, chunk in enumerate(raw_chunks):
            text = _restore_figure_blocks(chunk.text, figure_blocks)
            abs_start = section["start"] + chunk.start_index
            abs_end = section["start"] + chunk.end_index
            page_start, page_end = page_range_for_span(
                page_index, abs_start, abs_end,
            )
            text = strip_page_markers(text)
            content_type = _classify_content(text)
            chart_meta = extract_chart_metadata(text)
            page_meta = (
                {"page_start": page_start, "page_end": page_end}
                if page_start is not None
                else {}
            )
            chunk_id = _stable_id(source_file, parent_id, local_idx, chunk.start_index)
            section_child_ids.append(chunk_id)
            results.append(
                {
                    "text": text,
                    "metadata": {
                        "source": source_file,
                        "chunk_index": global_idx,
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "section_id": parent_id,
                        "section": section["path"],
                        "section_path": section["path"],
                        "section_level": section["level"],
                        "section_title": section["title"],
                        "section_index": section_idx,
                        "section_chunk_index": local_idx,
                        "section_chunk_count": len(raw_chunks),
                        "section_path_ids": section["path_ids"],
                        "content_type": content_type,
                        "has_figure": content_type == "figure" or "![" in text,
                        "token_count": chunk.token_count,
                        "start_index": abs_start,
                        "end_index": abs_end,
                        "chunker_type": "hierarchical",
                        **page_meta,
                        **doc_meta,
                        **chart_meta,
                    },
                }
            )
            global_idx += 1

        for offset, child_id in enumerate(section_child_ids):
            metadata = results[start_result_idx + offset]["metadata"]
            metadata["previous_chunk_id"] = section_child_ids[offset - 1] if offset else ""
            metadata["next_chunk_id"] = (
                section_child_ids[offset + 1] if offset + 1 < len(section_child_ids) else ""
            )

    return results

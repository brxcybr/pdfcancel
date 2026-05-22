"""Chunking module for pdfcancel --chunks mode.

Splits cleaned markdown into structured chunks for RAG/LLM pipelines.
Uses Chonkie's chunkers with markdown-aware splitting rules.

Figure-aware: images, their descriptions, and captions are detected as
atomic "figure blocks" that are never split across chunk boundaries.
Outputs JSONL with rich metadata (source, section breadcrumb, chunk index,
content_type).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


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
    r"(?:\n> \*\*Figure description:\*\* [^\n]+)?"  # optional description
    r"(?:\n(?:Figure|Fig\.?)\s*\d*[^\n]*)?"    # optional caption
    r")",
    re.MULTILINE,
)

_SENTINEL_PREFIX = "\x00FIGBLOCK:"


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
    lines = [l for l in stripped.split("\n") if l.strip()]
    if lines and sum(1 for l in lines if l.strip().startswith("|")) > len(lines) * 0.5:
        return "table"
    return "prose"


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
    will never be split across chunk boundaries.

    Args:
        markdown_content: The cleaned markdown text.
        source_file: Original PDF filename for metadata.
        chunk_size: Maximum tokens per chunk.
        chunker_type: One of "recursive", "semantic", "sentence".

    Returns:
        List of chunk dicts with "text", "metadata" keys.
    """
    # Phase 1: Protect figure blocks from splitting
    protected_text, figure_blocks = _protect_figure_blocks(markdown_content)

    # Phase 2: Chunk the protected text
    chunker = _create_chunker(chunker_type, chunk_size)
    raw_chunks = chunker(protected_text)

    # Phase 3: Restore figure blocks and build metadata
    heading_index = _build_heading_index(markdown_content)

    results = []
    for idx, chunk in enumerate(raw_chunks):
        # Restore any sentinels in this chunk
        text = _restore_figure_blocks(chunk.text, figure_blocks)

        # Map back to original position for section lookup
        # Use the chunk's start position in the protected text to find
        # the approximate section in the original
        section = _section_at_offset(heading_index, chunk.start_index)

        # Classify content type
        content_type = _classify_content(text)

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
        except ImportError:
            raise SystemExit(
                "Error: Semantic chunking requires chonkie[semantic].\n"
                "  pip install 'chonkie[semantic]'"
            )
        return SemanticChunker(chunk_size=chunk_size)
    elif chunker_type == "sentence":
        from chonkie import SentenceChunker
        return SentenceChunker(chunk_size=chunk_size)
    else:
        raise SystemExit(
            f"Error: Unknown chunker type '{chunker_type}'.\n"
            "  Supported: recursive, semantic, sentence"
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

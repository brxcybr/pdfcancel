"""Chunking module for pdfcancel --chunks mode.

Splits cleaned markdown into structured chunks for RAG/LLM pipelines.
Uses Chonkie's chunkers with markdown-aware splitting rules.
Outputs JSONL with rich metadata (source, section breadcrumb, chunk index).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def chunk_markdown(
    markdown_content: str,
    *,
    source_file: str,
    chunk_size: int = 1024,
    chunker_type: str = "recursive",
) -> list[dict[str, Any]]:
    """Split markdown into chunks with metadata.

    Args:
        markdown_content: The cleaned markdown text.
        source_file: Original PDF filename for metadata.
        chunk_size: Maximum tokens per chunk.
        chunker_type: One of "recursive", "semantic", "sentence".

    Returns:
        List of chunk dicts with "text", "metadata" keys.
    """
    chunker = _create_chunker(chunker_type, chunk_size)
    raw_chunks = chunker(markdown_content)

    # Build section heading index for breadcrumbs
    heading_index = _build_heading_index(markdown_content)

    results = []
    for idx, chunk in enumerate(raw_chunks):
        text = chunk.text
        start = chunk.start_index

        # Find the section breadcrumb for this chunk's position
        section = _section_at_offset(heading_index, start)

        results.append({
            "text": text,
            "metadata": {
                "source": source_file,
                "chunk_index": idx,
                "section": section,
                "token_count": chunk.token_count,
                "start_index": start,
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

"""Tests for page metadata propagation through chunking (chunks.py)."""

from __future__ import annotations

from types import SimpleNamespace

from pdfcancel import chunks as chunks_mod
from pdfcancel.chunks import chunk_markdown
from pdfcancel.pages import join_pages


class FakeChunker:
    """Splits on blank lines, mimicking Chonkie chunk objects."""

    def __call__(self, text: str):
        parts = [part for part in text.split("\n\n") if part.strip()]
        out = []
        cursor = 0
        for part in parts:
            start = text.find(part, cursor)
            end = start + len(part)
            cursor = end
            out.append(
                SimpleNamespace(
                    text=part,
                    start_index=start,
                    end_index=end,
                    token_count=len(part.split()),
                )
            )
        return out


def _patch_chunker(monkeypatch):
    monkeypatch.setattr(
        chunks_mod, "_create_chunker", lambda *_args, **_kwargs: FakeChunker()
    )


def test_chunks_carry_page_start_and_page_end(monkeypatch):
    _patch_chunker(monkeypatch)
    markdown = join_pages(
        [
            "# Title\n\nFirst page paragraph.",
            "Second page paragraph.",
            "Third page paragraph.",
        ],
        preserve_pages=True,
    )

    chunks = chunk_markdown(markdown, source_file="paper.pdf")
    text_chunks = [
        c for c in chunks if "paragraph" in c["text"] or "Title" in c["text"]
    ]
    assert text_chunks

    first = next(c for c in chunks if "First page paragraph" in c["text"])
    second = next(c for c in chunks if "Second page paragraph" in c["text"])
    third = next(c for c in chunks if "Third page paragraph" in c["text"])
    assert first["metadata"]["page_start"] == 1
    assert first["metadata"]["page_end"] == 1
    assert second["metadata"]["page_start"] == 2
    assert third["metadata"]["page_start"] == 3

    # Marker comments are stripped from the chunk text itself
    for c in chunks:
        assert "pdfcancel-page" not in c["text"]


def test_chunk_spanning_pages_reports_range(monkeypatch):
    # A single chunk that straddles a page boundary: feed a chunker that
    # returns the whole document as one chunk.
    class WholeDocChunker:
        def __call__(self, text):
            return [
                SimpleNamespace(
                    text=text, start_index=0, end_index=len(text),
                    token_count=len(text.split()),
                )
            ]

    monkeypatch.setattr(
        chunks_mod, "_create_chunker", lambda *_a, **_k: WholeDocChunker()
    )
    markdown = join_pages(["page one text", "page two text"], preserve_pages=True)
    chunks = chunk_markdown(markdown, source_file="paper.pdf")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_start"] == 1
    assert chunks[0]["metadata"]["page_end"] == 2
    assert "pdfcancel-page" not in chunks[0]["text"]


def test_chunks_without_markers_have_no_page_metadata(monkeypatch):
    _patch_chunker(monkeypatch)
    markdown = "# Title\n\nA paragraph without any page markers.\n"
    chunks = chunk_markdown(markdown, source_file="paper.pdf")
    assert chunks
    for c in chunks:
        assert "page_start" not in c["metadata"]
        assert "page_end" not in c["metadata"]


def test_hierarchical_chunks_carry_page_metadata(monkeypatch):
    _patch_chunker(monkeypatch)
    markdown = join_pages(
        [
            "# Intro\n\nIntro text on page one.",
            "## Methods\n\nMethods text on page two.",
        ],
        preserve_pages=True,
    )
    chunks = chunk_markdown(
        markdown, source_file="paper.pdf", chunker_type="hierarchical"
    )
    assert chunks
    intro = next(c for c in chunks if "Intro text" in c["text"])
    methods = next(c for c in chunks if "Methods text" in c["text"])
    assert intro["metadata"]["page_start"] == 1
    assert methods["metadata"]["page_start"] == 2
    for c in chunks:
        assert "pdfcancel-page" not in c["text"]


def test_hierarchical_falls_back_to_recursive_without_semantic(monkeypatch):
    calls = []

    def fake_create(chunker_type, chunk_size):
        calls.append(chunker_type)
        if chunker_type == "semantic":
            raise ImportError("Semantic chunking requires chonkie[semantic]")
        return FakeChunker()

    monkeypatch.setattr(chunks_mod, "_create_chunker", fake_create)
    markdown = "# Title\n\nSome content for fallback testing.\n"
    chunks = chunk_markdown(
        markdown, source_file="paper.pdf", chunker_type="hierarchical"
    )
    assert chunks
    assert calls == ["semantic", "recursive"]


def test_explicit_semantic_importerror_becomes_systemexit(monkeypatch):
    import pytest

    def fake_create(chunker_type, chunk_size):
        raise ImportError('pip install "pdfcancel[chunking]"')

    monkeypatch.setattr(chunks_mod, "_create_chunker", fake_create)
    with pytest.raises(SystemExit):
        chunk_markdown("text", source_file="p.pdf", chunker_type="semantic")

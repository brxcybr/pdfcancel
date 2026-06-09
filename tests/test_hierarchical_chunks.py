from __future__ import annotations

from types import SimpleNamespace

from pdfcancel import chunks as chunks_mod
from pdfcancel.chunks import chunk_markdown


class FakeChunker:
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


def test_hierarchical_chunks_include_section_and_sibling_metadata(monkeypatch):
    monkeypatch.setattr(chunks_mod, "_create_chunker", lambda *_args, **_kwargs: FakeChunker())
    markdown = """# Course

Intro text.

## Metrics

Precision explains positive predictions.

Recall explains positive cases.

## Figures

![img-1.jpeg](img-1.jpeg)
> **Figure description:** A chart comparing precision and recall.
"""

    chunks = chunk_markdown(
        markdown,
        source_file="book.pdf",
        chunker_type="hierarchical",
        chunk_size=64,
    )

    metrics = [c for c in chunks if c["metadata"]["section_path"] == "Course > Metrics"]
    assert len(metrics) == 3
    assert {c["metadata"]["parent_id"] for c in metrics} == {
        metrics[0]["metadata"]["parent_id"]
    }
    assert metrics[1]["metadata"]["previous_chunk_id"]
    assert metrics[1]["metadata"]["next_chunk_id"]
    assert metrics[1]["metadata"]["section_chunk_count"] == 3

    figure = next(c for c in chunks if "Figure description" in c["text"])
    assert figure["metadata"]["content_type"] == "figure"
    assert figure["metadata"]["has_figure"] is True
    assert figure["metadata"]["section_path"] == "Course > Figures"


def test_hierarchical_chunks_keep_blockquoted_figure_data_atomic(monkeypatch):
    monkeypatch.setattr(chunks_mod, "_create_chunker", lambda *_args, **_kwargs: FakeChunker())
    markdown = """# Metrics

![img-1.jpeg](img-1.jpeg)
> **Figure description:** A ROC curve comparing TPR and FPR.
>
> | Metric | Value |
> | --- | --- |
> | AUC | 0.94 |

The receiver operating characteristic is discussed next.
"""

    chunks = chunk_markdown(
        markdown,
        source_file="book.pdf",
        chunker_type="hierarchical",
        chunk_size=64,
    )

    figure_chunks = [c for c in chunks if c["metadata"]["content_type"] == "figure"]
    assert len(figure_chunks) == 1
    assert "Figure description" in figure_chunks[0]["text"]
    assert "| AUC | 0.94 |" in figure_chunks[0]["text"]


def test_hierarchical_section_metadata_strips_figure_sentinels(monkeypatch):
    monkeypatch.setattr(chunks_mod, "_create_chunker", lambda *_args, **_kwargs: FakeChunker())
    markdown = """# Conclusion

## ![img-1.jpeg](img-1.jpeg)
> **Figure description:** A chart in a malformed OCR heading.

### FIGBLOCK:9

The section text explains the chart.
"""

    chunks = chunk_markdown(
        markdown,
        source_file="book.pdf",
        chunker_type="hierarchical",
        chunk_size=64,
    )

    assert chunks
    for chunk in chunks:
        section_path = chunk["metadata"]["section_path"]
        section_title = chunk["metadata"]["section_title"]
        assert "FIGBLOCK" not in section_path
        assert "\x00" not in section_path
        assert "FIGBLOCK" not in section_title
        assert "\x00" not in section_title

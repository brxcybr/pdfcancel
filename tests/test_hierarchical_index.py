from __future__ import annotations

import numpy as np

from pdfcancel import index as index_mod


class FakeEmbeddingModel:
    def encode(self, texts):
        return np.array([[float(len(text) or 1), 1.0] for text in texts], dtype=np.float32)


def test_hierarchical_context_expands_to_same_section_figures(monkeypatch, tmp_path):
    monkeypatch.setattr(index_mod, "_get_embedding_model", lambda: FakeEmbeddingModel())
    index_mod.set_index_path(tmp_path / "idx.db")
    parent_id = "section-1"
    chunks = [
        {
            "text": "Precision is the ratio of true positives to predicted positives.",
            "metadata": {
                "chunk_index": 0,
                "chunk_id": "c0",
                "parent_id": parent_id,
                "section": "Metrics",
                "section_path": "Metrics",
                "content_type": "prose",
                "token_count": 10,
            },
        },
        {
            "text": "![img-1.jpeg](img-1.jpeg)\n> **Figure description:** Precision and recall trade off on a classifier chart.",
            "metadata": {
                "chunk_index": 1,
                "chunk_id": "c1",
                "parent_id": parent_id,
                "section": "Metrics",
                "section_path": "Metrics",
                "content_type": "figure",
                "token_count": 12,
            },
        },
    ]

    index_mod.ingest_chunks(chunks, "test", "book.pdf")
    results = index_mod.search("precision", "test", mode="text", context=True, top_k=1)

    assert results
    assert results[0]["parent_id"] == parent_id
    assert "Figure description" in results[0]["figure_context"]

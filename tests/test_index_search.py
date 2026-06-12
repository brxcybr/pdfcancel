"""Tests for RRF merging, FTS5 query escaping, and page columns (index.py)."""

from __future__ import annotations

import sqlite3

from pdfcancel.index import _escape_fts_query, _init_db, _rrf_merge, _search_fts


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------

def _result(rid, score, method):
    return {"id": rid, "score": score, "method": method, "content_type": "prose"}


def test_rrf_merge_uses_ranks_not_raw_scores():
    # FTS BM25-style scores (unbounded) vs cosine similarity (0-1): a huge
    # raw score must not dominate — only rank position matters.
    fts = [_result(1, 55.0, "text"), _result(2, 54.0, "text")]
    sem = [_result(3, 0.99, "semantic"), _result(2, 0.95, "semantic")]

    merged = _rrf_merge([fts, sem], k=60)
    by_id = {r["id"]: r for r in merged}

    # id=2 appears in both lists (rank 2 each): 1/62 + 1/62
    assert by_id[2]["score"] == (1 / 62) + (1 / 62)
    # id=1 and id=3 each rank 1 in one list: 1/61
    assert by_id[1]["score"] == 1 / 61
    assert by_id[3]["score"] == 1 / 61
    # The doubly-found chunk wins despite lower raw scores in both lists
    assert merged[0]["id"] == 2


def test_rrf_merge_combines_methods():
    fts = [_result(1, 10.0, "text")]
    sem = [_result(1, 0.5, "semantic")]
    merged = _rrf_merge([fts, sem])
    assert len(merged) == 1
    assert merged[0]["method"] == "text+semantic"


def test_rrf_merge_single_list_preserves_order():
    fts = [_result(1, 9.0, "text"), _result(2, 8.0, "text"), _result(3, 7.0, "text")]
    merged = _rrf_merge([fts])
    assert [r["id"] for r in merged] == [1, 2, 3]


def test_rrf_merge_empty_lists():
    assert _rrf_merge([[], []]) == []


# ---------------------------------------------------------------------------
# FTS5 query escaping
# ---------------------------------------------------------------------------

def test_escape_fts_query_wraps_terms_in_quotes():
    assert _escape_fts_query("threat intelligence") == '"threat" "intelligence"'


def test_escape_fts_query_handles_special_characters():
    assert _escape_fts_query("ATT&CK") == '"ATT&CK"'
    assert _escape_fts_query("risk-management") == '"risk-management"'
    assert _escape_fts_query('say "hello"') == '"say" """hello"""'


def test_escape_fts_query_empty():
    assert _escape_fts_query("") == '""'
    assert _escape_fts_query("   ") == '""'


def _make_db():
    conn = sqlite3.connect(":memory:")
    _init_db(conn)
    return conn


def _insert_chunk(conn, text, page_start=None, page_end=None):
    conn.execute(
        """INSERT INTO chunks
           (source, chunk_index, chunk_id, parent_id, section, section_path,
            content_type, text, token_count, embedding, metadata,
            page_start, page_end)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "paper.pdf", 0, "", "", "", "", "prose", text, 5,
            b"\x00\x00\x00\x00", "{}", page_start, page_end,
        ),
    )
    conn.commit()


def test_fts_match_with_special_characters_does_not_raise():
    conn = _make_db()
    _insert_chunk(conn, "The ATT&CK framework maps adversary techniques.")
    # Unescaped, these queries raise sqlite3.OperationalError (syntax error)
    for query in ['ATT&CK', 'tech-niques OR', '"unbalanced', "NEAR("]:
        results = _search_fts(conn, query, 10)
        assert isinstance(results, list)
    # And a sane query still finds the row
    results = _search_fts(conn, "ATT&CK", 10)
    assert len(results) == 1
    conn.close()


def test_fts_results_include_page_columns():
    conn = _make_db()
    _insert_chunk(conn, "Findings discussed in detail here.", page_start=4, page_end=5)
    results = _search_fts(conn, "findings", 10)
    assert len(results) == 1
    assert results[0]["page_start"] == 4
    assert results[0]["page_end"] == 5
    conn.close()


def test_old_db_without_page_columns_upgrades_gracefully():
    conn = sqlite3.connect(":memory:")
    # Simulate a legacy schema (no page columns)
    conn.executescript("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_id TEXT, parent_id TEXT, section TEXT, section_path TEXT,
            content_type TEXT, text TEXT NOT NULL, token_count INTEGER,
            embedding BLOB, metadata TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, source, section, content='chunks', content_rowid='id'
        );
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, source, section)
            VALUES (new.id, new.text, new.source, new.section);
        END;
        CREATE TABLE sources (
            source TEXT PRIMARY KEY, indexed_at TEXT, chunk_count INTEGER
        );
    """)
    conn.execute(
        """INSERT INTO chunks
           (source, chunk_index, text, token_count, embedding, metadata)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("old.pdf", 0, "legacy chunk text", 3, b"\x00", "{}"),
    )
    conn.commit()

    # _init_db must add the new columns without breaking existing rows
    _init_db(conn)
    results = _search_fts(conn, "legacy", 10)
    assert len(results) == 1
    assert results[0]["page_start"] is None
    assert results[0]["page_end"] is None
    conn.close()

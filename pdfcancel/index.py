"""Per-project search index for pdfcancel.

Stores chunks with embeddings in a SQLite database for fast retrieval.
Supports both full-text search (FTS5) and semantic similarity search
using local embeddings (no API cost).

Index location: ~/.pdfcancel/indexes/<name>.db
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

console = Console()

INDEX_DIR = Path.home() / ".pdfcancel" / "indexes"


# Custom path override — set by CLI --index-path flag
_custom_index_path: Path | None = None


def set_index_path(path: Path | None) -> None:
    """Override the default index location."""
    global _custom_index_path
    _custom_index_path = path


def _index_path(name: str) -> Path:
    """Return the path to a named index database."""
    if _custom_index_path:
        # Custom path: use it directly (name is ignored if path ends in .db)
        p = Path(_custom_index_path)
        if p.suffix == ".db":
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{name}.db"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR / f"{name}.db"


def _get_embedding_model():
    """Load the local embedding model (cached after first call)."""
    try:
        from model2vec import StaticModel
        return StaticModel.from_pretrained("minishlab/potion-base-32M")
    except ImportError:
        raise SystemExit(
            "Error: Semantic indexing requires model2vec.\n"
            "  pip install model2vec"
        )


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_id TEXT,
            parent_id TEXT,
            section TEXT,
            section_path TEXT,
            content_type TEXT,
            text TEXT NOT NULL,
            token_count INTEGER,
            embedding BLOB,
            metadata TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, source, section,
            content='chunks',
            content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, source, section)
            VALUES (new.id, new.text, new.source, new.section);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, source, section)
            VALUES ('delete', old.id, old.text, old.source, old.section);
        END;

        CREATE TABLE IF NOT EXISTS sources (
            source TEXT PRIMARY KEY,
            indexed_at TEXT,
            chunk_count INTEGER
        );
    """)
    _ensure_column(conn, "chunks", "chunk_id", "TEXT")
    _ensure_column(conn, "chunks", "parent_id", "TEXT")
    _ensure_column(conn, "chunks", "section_path", "TEXT")
    _ensure_column(conn, "chunks", "content_type", "TEXT")
    # Page citation columns (--preserve-pages); added via ALTER so old
    # databases keep working (values stay NULL for legacy rows)
    _ensure_column(conn, "chunks", "page_start", "INTEGER")
    _ensure_column(conn, "chunks", "page_end", "INTEGER")


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    col_type: str,
) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def ingest_chunks(
    chunks: list[dict[str, Any]],
    index_name: str,
    source_file: str,
) -> int:
    """Add chunks to a named index with embeddings.

    Args:
        chunks: List of chunk dicts from chunk_markdown().
        index_name: Name of the project index.
        source_file: Original PDF filename.

    Returns:
        Number of chunks ingested.
    """
    db_path = _index_path(index_name)
    conn = sqlite3.connect(str(db_path))
    _init_db(conn)

    # Remove existing chunks for this source (re-index)
    conn.execute("DELETE FROM chunks WHERE source = ?", (source_file,))

    # Generate embeddings for all chunk texts
    model = _get_embedding_model()
    texts = [c["text"] for c in chunks]
    console.print(f"    Generating embeddings for {len(texts)} chunks ...")
    embeddings = model.encode(texts)

    # Insert chunks with embeddings
    for chunk, emb in zip(chunks, embeddings):
        meta = chunk.get("metadata", {})
        conn.execute(
            """INSERT INTO chunks
               (source, chunk_index, chunk_id, parent_id, section, section_path,
                content_type, text, token_count, embedding, metadata,
                page_start, page_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source_file,
                meta.get("chunk_index", 0),
                meta.get("chunk_id", ""),
                meta.get("parent_id", ""),
                meta.get("section", ""),
                meta.get("section_path", meta.get("section", "")),
                meta.get("content_type", "prose"),
                chunk["text"],
                meta.get("token_count", 0),
                emb.astype(np.float32).tobytes(),
                json.dumps(meta, ensure_ascii=False),
                meta.get("page_start"),
                meta.get("page_end"),
            ),
        )

    # Update sources table
    from datetime import datetime, timezone
    conn.execute(
        """INSERT OR REPLACE INTO sources (source, indexed_at, chunk_count)
           VALUES (?, ?, ?)""",
        (source_file, datetime.now(timezone.utc).isoformat(), len(chunks)),
    )

    conn.commit()
    conn.close()
    return len(chunks)


# Content-type weights for hybrid scoring.
# Higher weight = boosted in results; lower = demoted.
_CONTENT_TYPE_WEIGHTS: dict[str, float] = {
    "abstract": 1.5,
    "figure": 1.3,
    "prose": 1.0,
    "table": 1.0,
    "frontmatter": 0.7,
    "references": 0.5,
}

# Reciprocal rank fusion constant (standard value from the RRF paper).
_RRF_K = 60


def _rrf_merge(
    result_lists: list[list[dict[str, Any]]],
    k: int = _RRF_K,
) -> list[dict[str, Any]]:
    """Merge ranked result lists with reciprocal rank fusion.

    Each result contributes 1 / (k + rank) per list it appears in (rank is
    1-based within its list). BM25 and cosine scores are never compared
    directly — only ranks matter. Results found by multiple methods get
    their methods joined (e.g. "text+semantic").

    Input lists must already be sorted best-first. Returns merged results
    with "score" set to the RRF score, sorted descending.
    """
    merged: dict[Any, dict[str, Any]] = {}
    scores: dict[Any, float] = {}

    for results in result_lists:
        for rank, r in enumerate(results, 1):
            rid = r["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            if rid in merged:
                existing = merged[rid]
                methods = existing["method"].split("+")
                if r["method"] not in methods:
                    existing["method"] = "+".join(methods + [r["method"]])
            else:
                merged[rid] = dict(r)

    out = []
    for rid, r in merged.items():
        r["score"] = scores[rid]
        out.append(r)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def search(
    query: str,
    index_name: str,
    *,
    top_k: int = 10,
    mode: str = "hybrid",
    context: bool = False,
) -> list[dict[str, Any]]:
    """Search an index for chunks matching a query.

    Args:
        query: Search query string.
        index_name: Name of the project index.
        top_k: Maximum number of results.
        mode: "semantic", "text", or "hybrid" (default).
        context: If True, include neighboring chunks as context_before/context_after.

    Returns:
        List of result dicts with text, source, section, score,
        and optionally context_before / context_after.
    """
    db_path = _index_path(index_name)
    if not db_path.exists():
        raise SystemExit(f"Error: Index '{index_name}' not found.\n  Run: pdfcancel <pdf> --index {index_name}")

    conn = sqlite3.connect(str(db_path))
    _init_db(conn)

    if mode == "hybrid":
        # Reciprocal rank fusion: BM25 ranks and cosine-similarity ranks
        # are fused positionally (raw scores live on different scales)
        fts_results = _search_fts(conn, query, top_k * 2)
        sem_results = _search_semantic(conn, query, top_k * 2)
        candidates = _rrf_merge([fts_results, sem_results])
    elif mode == "text":
        candidates = _search_fts(conn, query, top_k * 2)
    else:
        candidates = _search_semantic(conn, query, top_k * 2)

    # Apply content-type weighting
    for r in candidates:
        content_type = r.get("content_type", "prose")
        weight = _CONTENT_TYPE_WEIGHTS.get(content_type, 1.0)
        r["weighted_score"] = r["score"] * weight

    # Sort by weighted score descending, take top_k
    ranked = sorted(candidates, key=lambda x: x["weighted_score"], reverse=True)
    ranked = ranked[:top_k]

    # Context expansion: fetch neighboring chunks for each result
    if context and ranked:
        _expand_context(conn, ranked)

    conn.close()
    return ranked


def _expand_context(
    conn: sqlite3.Connection,
    results: list[dict[str, Any]],
) -> None:
    """Fetch neighboring and same-section context for each result.

    Adds linear neighbors for all indexes. For hierarchical indexes, also adds
    same-section sibling snippets and figure chunks attached to the same parent.
    """
    for r in results:
        source = r["source"]
        chunk_idx = r.get("chunk_index", -1)
        parent_id = r.get("parent_id") or ""

        # If we don't have chunk_index, try to get it from the DB
        if chunk_idx < 0:
            row = conn.execute(
                "SELECT chunk_index FROM chunks WHERE id = ?", (r["id"],)
            ).fetchone()
            if row:
                chunk_idx = row[0]

        # Fetch previous chunk
        prev = conn.execute(
            "SELECT text FROM chunks WHERE source = ? AND chunk_index = ?",
            (source, chunk_idx - 1),
        ).fetchone()
        r["context_before"] = prev[0] if prev else ""

        # Fetch next chunk
        nxt = conn.execute(
            "SELECT text FROM chunks WHERE source = ? AND chunk_index = ?",
            (source, chunk_idx + 1),
        ).fetchone()
        r["context_after"] = nxt[0] if nxt else ""

        if not parent_id:
            r["section_context"] = ""
            r["figure_context"] = ""
            continue

        section_rows = conn.execute(
            """SELECT text FROM chunks
               WHERE source = ? AND parent_id = ? AND id != ?
               ORDER BY ABS(chunk_index - ?), chunk_index
               LIMIT 4""",
            (source, parent_id, r["id"], chunk_idx),
        ).fetchall()
        r["section_context"] = "\n\n".join(row[0] for row in section_rows)

        figure_rows = conn.execute(
            """SELECT text FROM chunks
               WHERE source = ? AND parent_id = ? AND content_type = 'figure'
               ORDER BY chunk_index
               LIMIT 3""",
            (source, parent_id),
        ).fetchall()
        r["figure_context"] = "\n\n".join(row[0] for row in figure_rows)


def list_indexes() -> list[dict[str, Any]]:
    """List all available indexes with their stats."""
    if not INDEX_DIR.exists():
        return []

    indexes = []
    for db_file in sorted(INDEX_DIR.glob("*.db")):
        name = db_file.stem
        conn = sqlite3.connect(str(db_file))
        try:
            _init_db(conn)
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = row[0] if row else 0
            sources = conn.execute("SELECT source, chunk_count FROM sources").fetchall()
            indexes.append({
                "name": name,
                "path": str(db_file),
                "total_chunks": chunk_count,
                "sources": [{"source": s, "chunks": c} for s, c in sources],
            })
        finally:
            conn.close()

    return indexes


def _escape_fts_query(query: str) -> str:
    """Escape a user query for FTS5 MATCH.

    Each whitespace-separated term is wrapped in double quotes (with embedded
    quotes doubled) so FTS5 operators and special characters like -, :, *,
    or & cannot break the MATCH expression.
    """
    terms = query.split()
    if not terms:
        return '""'
    return " ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _search_fts(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Full-text search using SQLite FTS5 with BM25 ranking."""
    rows = conn.execute(
        """SELECT c.id, c.text, c.source, c.section, c.token_count,
                  c.chunk_index, c.metadata, c.chunk_id, c.parent_id,
                  c.section_path, c.content_type, rank * -1 as score,
                  c.page_start, c.page_end
           FROM chunks_fts fts
           JOIN chunks c ON c.id = fts.rowid
           WHERE chunks_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (_escape_fts_query(query), top_k),
    ).fetchall()

    results = []
    for r in rows:
        meta = json.loads(r[6]) if r[6] else {}
        results.append({
            "id": r[0],
            "text": r[1],
            "source": r[2],
            "section": r[3],
            "token_count": r[4],
            "chunk_index": r[5],
            "doc_title": meta.get("doc_title", ""),
            "doc_author": meta.get("doc_author", ""),
            "doc_year": meta.get("doc_year", ""),
            "chunk_id": r[7] or meta.get("chunk_id", ""),
            "parent_id": r[8] or meta.get("parent_id", ""),
            "section_path": r[9] or meta.get("section_path", r[3] or ""),
            "content_type": r[10] or meta.get("content_type", "prose"),
            "has_structured_chart_data": meta.get("has_structured_chart_data", False),
            "chart_data": meta.get("chart_data"),
            "vega_lite_spec": meta.get("vega_lite_spec"),
            "page_start": r[12] if r[12] is not None else meta.get("page_start"),
            "page_end": r[13] if r[13] is not None else meta.get("page_end"),
            "score": r[11],
            "method": "text",
        })
    return results


def _search_semantic(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Semantic search using cosine similarity on embeddings."""
    model = _get_embedding_model()
    query_emb = model.encode([query])[0].astype(np.float32)

    # Load all embeddings (fine for <100k chunks; for larger, use a vector DB)
    rows = conn.execute(
        """SELECT id, text, source, section, token_count, chunk_index, metadata,
                  embedding, chunk_id, parent_id, section_path, content_type,
                  page_start, page_end
           FROM chunks"""
    ).fetchall()

    if not rows:
        return []

    results = []
    for row in rows:
        chunk_emb = np.frombuffer(row[7], dtype=np.float32)
        similarity = float(np.dot(query_emb, chunk_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8
        ))
        meta = json.loads(row[6]) if row[6] else {}
        results.append({
            "id": row[0],
            "text": row[1],
            "source": row[2],
            "section": row[3],
            "token_count": row[4],
            "chunk_index": row[5],
            "doc_title": meta.get("doc_title", ""),
            "doc_author": meta.get("doc_author", ""),
            "doc_year": meta.get("doc_year", ""),
            "chunk_id": row[8] or meta.get("chunk_id", ""),
            "parent_id": row[9] or meta.get("parent_id", ""),
            "section_path": row[10] or meta.get("section_path", row[3] or ""),
            "content_type": row[11] or meta.get("content_type", "prose"),
            "has_structured_chart_data": meta.get("has_structured_chart_data", False),
            "chart_data": meta.get("chart_data"),
            "vega_lite_spec": meta.get("vega_lite_spec"),
            "page_start": row[12] if row[12] is not None else meta.get("page_start"),
            "page_end": row[13] if row[13] is not None else meta.get("page_end"),
            "score": similarity,
            "method": "semantic",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

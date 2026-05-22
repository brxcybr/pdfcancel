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


def _index_path(name: str) -> Path:
    """Return the path to a named index database."""
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
            section TEXT,
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
            """INSERT INTO chunks (source, chunk_index, section, text, token_count, embedding, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                source_file,
                meta.get("chunk_index", 0),
                meta.get("section", ""),
                chunk["text"],
                meta.get("token_count", 0),
                emb.astype(np.float32).tobytes(),
                json.dumps(meta, ensure_ascii=False),
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


def search(
    query: str,
    index_name: str,
    *,
    top_k: int = 10,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Search an index for chunks matching a query.

    Args:
        query: Search query string.
        index_name: Name of the project index.
        top_k: Maximum number of results.
        mode: "semantic", "text", or "hybrid" (default).

    Returns:
        List of result dicts with text, source, section, score.
    """
    db_path = _index_path(index_name)
    if not db_path.exists():
        raise SystemExit(f"Error: Index '{index_name}' not found.\n  Run: pdfcancel <pdf> --index {index_name}")

    conn = sqlite3.connect(str(db_path))
    _init_db(conn)

    results = []

    if mode in ("text", "hybrid"):
        results.extend(_search_fts(conn, query, top_k))

    if mode in ("semantic", "hybrid"):
        results.extend(_search_semantic(conn, query, top_k))

    conn.close()

    # Deduplicate by chunk id, keep highest score
    seen: dict[int, dict] = {}
    for r in results:
        rid = r["id"]
        if rid not in seen or r["score"] > seen[rid]["score"]:
            seen[rid] = r

    # Sort by score descending, return top_k
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


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


def _search_fts(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Full-text search using SQLite FTS5 with BM25 ranking."""
    rows = conn.execute(
        """SELECT c.id, c.text, c.source, c.section, c.token_count,
                  rank * -1 as score
           FROM chunks_fts fts
           JOIN chunks c ON c.id = fts.rowid
           WHERE chunks_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (query, top_k),
    ).fetchall()

    return [
        {
            "id": r[0],
            "text": r[1],
            "source": r[2],
            "section": r[3],
            "token_count": r[4],
            "score": r[5],
            "method": "text",
        }
        for r in rows
    ]


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
        "SELECT id, text, source, section, token_count, embedding FROM chunks"
    ).fetchall()

    if not rows:
        return []

    results = []
    for row in rows:
        chunk_emb = np.frombuffer(row[5], dtype=np.float32)
        # Cosine similarity
        similarity = float(np.dot(query_emb, chunk_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8
        ))
        results.append({
            "id": row[0],
            "text": row[1],
            "source": row[2],
            "section": row[3],
            "token_count": row[4],
            "score": similarity,
            "method": "semantic",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

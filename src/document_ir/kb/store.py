"""SQLite registry for knowledge bases and documents (files under ``data/kbs/``)."""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from document_ir.paths import DATA_DIR, KB_DIR, SQLITE_PATH

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    """Derive a URL-safe slug from a display name.

    Args:
        name: Human-readable knowledge base name.

    Returns:
        Lowercase slug with non-alphanumeric runs replaced by ``_``, max 80 chars.
        Falls back to ``kb`` if the result would be empty.
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_") or "kb"
    if len(s) > 80:
        s = s[:80].rstrip("_")
    return s


def ensure_data_dirs() -> None:
    """Create ``data/`` and ``data/kbs/`` if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and row factory set.

    Returns:
        Connection to ``SQLITE_PATH``; rows are ``sqlite3.Row`` mappings.
    """
    ensure_data_dirs()
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create knowledge base and document tables if missing."""
    ensure_data_dirs()
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_id);
            """
        )
        conn.commit()


@dataclass
class KnowledgeBase:
    """A named knowledge base with a unique slug and storage directory."""

    id: str
    name: str
    slug: str
    created_at: str


@dataclass
class Document:
    """A document row pointing to Markdown under ``data/kbs/{slug}/{id}.md``."""

    id: str
    kb_id: str
    original_filename: str
    rel_path: str
    content_hash: str
    updated_at: str


def _unique_slug(conn: sqlite3.Connection, base_slug: str) -> str:
    """Return ``base_slug`` or ``base_slug_2``, ``base_slug_3``, … if taken."""
    slug = base_slug
    n = 2
    while True:
        row = conn.execute("SELECT 1 FROM knowledge_bases WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return slug
        slug = f"{base_slug}_{n}"
        n += 1


def create_knowledge_base(name: str) -> KnowledgeBase:
    """Insert a new knowledge base and create its directory under ``data/kbs/``.

    Args:
        name: Non-empty display name (trimmed).

    Returns:
        The created ``KnowledgeBase`` instance.

    Raises:
        ValueError: If ``name`` is empty after stripping.
    """
    name = name.strip()
    if not name:
        raise ValueError("Knowledge base name cannot be empty")
    init_db()
    kb_id = str(uuid.uuid4())
    base = slugify(name)
    with get_connection() as conn:
        slug = _unique_slug(conn, base)
        created = _utc_now()
        conn.execute(
            "INSERT INTO knowledge_bases (id, name, slug, created_at) VALUES (?, ?, ?, ?)",
            (kb_id, name, slug, created),
        )
        conn.commit()
    (KB_DIR / slug).mkdir(parents=True, exist_ok=True)
    logger.info("SQLite: knowledge base created | id=%s slug=%s name=%r", kb_id, slug, name)
    return KnowledgeBase(id=kb_id, name=name, slug=slug, created_at=created)


def list_knowledge_bases() -> list[KnowledgeBase]:
    """List all knowledge bases ordered by name."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, slug, created_at FROM knowledge_bases ORDER BY name"
        ).fetchall()
    return [KnowledgeBase(**dict(r)) for r in rows]


def get_knowledge_base_by_slug(slug: str) -> KnowledgeBase | None:
    """Look up a knowledge base by slug."""
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, slug, created_at FROM knowledge_bases WHERE slug = ?",
            (slug,),
        ).fetchone()
    return KnowledgeBase(**dict(row)) if row else None


def get_knowledge_base_by_id(kb_id: str) -> KnowledgeBase | None:
    """Look up a knowledge base by primary key."""
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, slug, created_at FROM knowledge_bases WHERE id = ?",
            (kb_id,),
        ).fetchone()
    return KnowledgeBase(**dict(row)) if row else None


def delete_knowledge_base(kb_id: str) -> KnowledgeBase | None:
    """Delete a knowledge base, its document rows, and on-disk Markdown files.

    Args:
        kb_id: Knowledge base UUID.

    Returns:
        The deleted ``KnowledgeBase`` if it existed, else ``None``.
    """
    kb = get_knowledge_base_by_id(kb_id)
    if not kb:
        return None
    for doc in list_documents(kb_id):
        path = absolute_md_path(doc)
        if path.is_file():
            path.unlink()
    slug_dir = KB_DIR / kb.slug
    if slug_dir.is_dir():
        shutil.rmtree(slug_dir, ignore_errors=True)
    with get_connection() as conn:
        conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        conn.commit()
    logger.info("SQLite: knowledge base deleted | id=%s slug=%s", kb.id, kb.slug)
    return kb


def list_documents(kb_id: str) -> list[Document]:
    """List documents belonging to a knowledge base, ordered by filename."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, kb_id, original_filename, rel_path, content_hash, updated_at "
            "FROM documents WHERE kb_id = ? ORDER BY original_filename",
            (kb_id,),
        ).fetchall()
    return [Document(**dict(r)) for r in rows]


def get_document(doc_id: str) -> Document | None:
    """Fetch a document by id."""
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, kb_id, original_filename, rel_path, content_hash, updated_at "
            "FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    return Document(**dict(row)) if row else None


def absolute_md_path(doc: Document) -> Path:
    """Absolute path to the Markdown file for ``doc``."""
    return DATA_DIR / "kbs" / doc.rel_path


def add_document(
    kb_id: str,
    original_filename: str,
    markdown_body: str,
    content_hash: str,
) -> Document:
    """Persist Markdown to disk and insert a document row.

    Args:
        kb_id: Parent knowledge base id.
        original_filename: Original upload filename (for display).
        markdown_body: Full Markdown content.
        content_hash: SHA-256 hex of source bytes (dedup key).

    Returns:
        The new ``Document`` instance.

    Raises:
        ValueError: If ``kb_id`` does not exist.
    """
    kb = get_knowledge_base_by_id(kb_id)
    if not kb:
        raise ValueError("Unknown knowledge base")
    doc_id = str(uuid.uuid4())
    rel = f"{kb.slug}/{doc_id}.md"
    path = DATA_DIR / "kbs" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_body, encoding="utf-8")
    updated = _utc_now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO documents (id, kb_id, original_filename, rel_path, content_hash, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, kb_id, original_filename, rel, content_hash, updated),
        )
        conn.commit()
    doc = Document(
        id=doc_id,
        kb_id=kb_id,
        original_filename=original_filename,
        rel_path=rel,
        content_hash=content_hash,
        updated_at=updated,
    )
    logger.info(
        "SQLite: document added | doc_id=%s kb_slug=%s file=%r hash=%s…",
        doc_id,
        kb.slug,
        original_filename,
        content_hash[:12],
    )
    return doc


def update_document_content(
    doc_id: str,
    markdown_body: str,
    content_hash: str,
    original_filename: str | None = None,
) -> Document | None:
    """Overwrite Markdown on disk and update document metadata.

    Args:
        doc_id: Document UUID.
        markdown_body: New Markdown content.
        content_hash: New content hash.
        original_filename: If set, updates the stored filename.

    Returns:
        Updated ``Document``, or ``None`` if ``doc_id`` was not found.
    """
    doc = get_document(doc_id)
    if not doc:
        return None
    path = absolute_md_path(doc)
    path.write_text(markdown_body, encoding="utf-8")
    updated = _utc_now()
    fname = original_filename if original_filename is not None else doc.original_filename
    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET original_filename = ?, content_hash = ?, updated_at = ? WHERE id = ?",
            (fname, content_hash, updated, doc_id),
        )
        conn.commit()
    logger.info("SQLite: document updated | doc_id=%s file=%r", doc_id, fname)
    return get_document(doc_id)


def delete_document_row(doc_id: str) -> Document | None:
    """Remove document row and delete its Markdown file if present.

    Returns:
        The deleted ``Document`` if it existed, else ``None``.
    """
    doc = get_document(doc_id)
    if not doc:
        return None
    path = absolute_md_path(doc)
    if path.is_file():
        path.unlink()
    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    logger.info("SQLite: document deleted | doc_id=%s file=%r", doc_id, doc.original_filename)
    return doc


def chroma_collection_for_slug(slug: str) -> str:
    """Chroma collection name used for a knowledge base slug."""
    return f"kb_{slug}"

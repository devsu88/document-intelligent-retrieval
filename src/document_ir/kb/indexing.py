"""Chunk Markdown, embed with OpenAI, and persist vectors in Chroma (one collection per KB slug)."""

from __future__ import annotations

import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from document_ir.kb.chroma_client import chroma_operation_lock, get_chroma_client
from document_ir.kb.store import (
    Document,
    absolute_md_path,
    chroma_collection_for_slug,
)
from document_ir.paths import CHROMA_DIR

EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# OpenAI embedding request limit (~8192 tokens); headroom for multi-input batches
_MAX_EMBED_BATCH_TOKENS = 7000
_MAX_SINGLE_INPUT_TOKENS = 8000

logger = logging.getLogger(__name__)

_openai: OpenAI | None = None


def _use_llm_document_chunking() -> bool:
    """True when ``USE_LLM_DOCUMENT_CHUNKING`` is unset or truthy (1/true/yes)."""
    return os.environ.get("USE_LLM_DOCUMENT_CHUNKING", "true").lower() in (
        "1",
        "true",
        "yes",
    )


def _split_markdown_for_index(markdown_text: str, source_label: str) -> list[str]:
    """Split document text into chunks for embedding.

    Tries LLM-based chunking when enabled; on failure or a single oversized chunk
    from the LLM, falls back to ``RecursiveCharacterTextSplitter``.

    Args:
        markdown_text: Full Markdown body.
        source_label: Filename or label stored in chunk metadata (for logs).

    Returns:
        List of chunk strings, possibly empty if input is empty after processing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    if _use_llm_document_chunking():
        try:
            from document_ir.ingestion.llm_chunking import llm_split_document

            parts = llm_split_document(markdown_text, source_label, "markdown")
            if parts:
                if len(parts) == 1 and len(markdown_text) > CHUNK_SIZE:
                    logger.warning(
                        "Indexing: LLM returned a single chunk for long text (%s chars > CHUNK_SIZE=%s); "
                        "using RecursiveCharacterTextSplitter for retrieval.",
                        len(markdown_text),
                        CHUNK_SIZE,
                    )
                    parts = splitter.split_text(markdown_text)
                    logger.info(
                        "Indexing: recursive split after LLM fallback | source=%r n_chunk=%s lengths=%s",
                        source_label,
                        len(parts),
                        [len(p) for p in parts],
                    )
                for i, p in enumerate(parts):
                    prev = p if len(p) <= 400 else p[:400] + "…"
                    logger.debug(
                        "Indexing: final chunk[%s] | source=%r len=%s | %r",
                        i,
                        source_label,
                        len(p),
                        prev,
                    )
                return parts
        except Exception as e:
            logger.warning("LLM document chunking failed, using character-based split: %s", e)
    return splitter.split_text(markdown_text)


def _get_openai() -> OpenAI:
    """Lazy singleton ``OpenAI`` client for embeddings."""
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def _embedding_token_count(text: str) -> int:
    """Approximate token count for embedding batch sizing (tiktoken or char/4 fallback)."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _shrink_oversized_chunks(chunks: list[str]) -> list[str]:
    """Re-split any chunk whose token estimate exceeds the embedding model input cap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    out: list[str] = []
    for c in chunks:
        if _embedding_token_count(c) > _MAX_SINGLE_INPUT_TOKENS:
            out.extend(splitter.split_text(c))
        else:
            out.append(c)
    return out


def _embed_chunks_batched(oa: OpenAI, model: str, chunks: list[str]) -> list[list[float]]:
    """Embed each chunk, splitting into multiple API calls when batch token sum is too large.

    Args:
        oa: OpenAI client.
        model: Embedding model id.
        chunks: Non-empty list of texts (caller ensures non-empty batches are sent).

    Returns:
        List of embedding vectors, same length as ``chunks``.

    Raises:
        RuntimeError: If the API returns a different number of embeddings than chunks.
    """
    if not chunks:
        return []
    embeddings: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    for ch in chunks:
        t = _embedding_token_count(ch)
        if batch and batch_tokens + t > _MAX_EMBED_BATCH_TOKENS:
            resp = oa.embeddings.create(model=model, input=batch).data
            embeddings.extend(e.embedding for e in resp)
            batch = []
            batch_tokens = 0
        batch.append(ch)
        batch_tokens += t
    if batch:
        resp = oa.embeddings.create(model=model, input=batch).data
        embeddings.extend(e.embedding for e in resp)
    if len(embeddings) != len(chunks):
        logger.error(
            "Indexing: embedding/chunk count mismatch | n_emb=%s n_chunk=%s",
            len(embeddings),
            len(chunks),
        )
        raise RuntimeError("Embedding count does not match chunk count")
    return embeddings


def remove_document_vectors(kb_slug: str, doc_id: str) -> None:
    """Delete all Chroma points for ``document_id`` in the KB collection.

    Args:
        kb_slug: Knowledge base slug (collection name derived via ``chroma_collection_for_slug``).
        doc_id: Document UUID matching metadata ``document_id``.
    """
    logger.info("Chroma: removing vectors | kb_slug=%s document_id=%s", kb_slug, doc_id)
    with chroma_operation_lock():
        name = chroma_collection_for_slug(kb_slug)
        coll = get_chroma_client().get_or_create_collection(name)
        coll.delete(where={"document_id": doc_id})
    logger.debug("Chroma: delete complete | collection=%s", name)


def index_markdown(
    kb_slug: str,
    doc_id: str,
    source_label: str,
    markdown_text: str,
) -> int:
    """Chunk, embed, and upsert one document into Chroma.

    Args:
        kb_slug: Target knowledge base slug.
        doc_id: Document UUID (ids become ``{doc_id}_chunk_{i}``).
        source_label: Stored in metadata as ``source`` (e.g. original filename).
        markdown_text: Full Markdown to index.

    Returns:
        Number of chunks written; 0 if text is empty or produces no chunks.
    """
    text = markdown_text.strip()
    if not text:
        logger.info("Indexing skipped: empty text | kb_slug=%s doc_id=%s", kb_slug, doc_id)
        return 0
    chunks = _split_markdown_for_index(text, source_label)
    if not chunks:
        logger.info("Indexing: no chunks | kb_slug=%s doc_id=%s source=%s", kb_slug, doc_id, source_label)
        return 0
    chunks = _shrink_oversized_chunks(chunks)
    if not chunks:
        logger.info("Indexing: no chunks after shrink | kb_slug=%s doc_id=%s", kb_slug, doc_id)
        return 0
    logger.info(
        "Indexing: embedding | kb_slug=%s doc_id=%s source=%s n_chunks=%s",
        kb_slug,
        doc_id,
        source_label,
        len(chunks),
    )
    oa = _get_openai()
    vectors = _embed_chunks_batched(oa, EMBEDDING_MODEL, chunks)
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metas = [{"source": source_label, "document_id": doc_id} for _ in chunks]
    with chroma_operation_lock():
        name = chroma_collection_for_slug(kb_slug)
        coll = get_chroma_client().get_or_create_collection(name)
        coll.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metas)
    logger.info(
        "Indexing: Chroma add OK | kb_slug=%s doc_id=%s chunks_written=%s",
        kb_slug,
        doc_id,
        len(chunks),
    )
    return len(chunks)


def index_document_file(kb_slug: str, doc: Document) -> int:
    """Load Markdown from disk for ``doc`` and call ``index_markdown``.

    Returns:
        Chunk count indexed, or 0 if the Markdown file is missing.
    """
    path = absolute_md_path(doc)
    if not path.is_file():
        logger.warning("Markdown file missing | path=%s doc_id=%s", path, doc.id)
        return 0
    body = path.read_text(encoding="utf-8")
    logger.debug("Indexing from file | kb_slug=%s file=%s", kb_slug, doc.original_filename)
    return index_markdown(kb_slug, doc.id, doc.original_filename, body)


def reindex_document(kb_slug: str, doc: Document) -> int:
    """Remove vectors for ``doc`` then index again from its Markdown file."""
    logger.info("Re-indexing | kb_slug=%s doc_id=%s", kb_slug, doc.id)
    remove_document_vectors(kb_slug, doc.id)
    return index_document_file(kb_slug, doc)


def drop_kb_collection(kb_slug: str) -> None:
    """Delete the Chroma collection for ``kb_slug`` if it exists (logs warning on failure)."""
    name = chroma_collection_for_slug(kb_slug)
    logger.info("Chroma: deleting collection | collection=%s", name)
    with chroma_operation_lock():
        client = get_chroma_client()
        try:
            client.delete_collection(name)
            logger.info("Chroma: collection deleted | collection=%s", name)
        except Exception as e:
            logger.warning("Chroma: delete_collection failed or collection missing | collection=%s err=%s", name, e)

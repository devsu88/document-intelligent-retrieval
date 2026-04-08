"""Singleton Chroma ``PersistentClient`` and a re-entrant lock for thread-safe access.

Gradio runs uploads, deletes, and chat on different threads; Chroma's SQLite backend
must not be hit concurrently from multiple threads without coordination.
"""

from __future__ import annotations

import logging
import threading

from chromadb import PersistentClient

from document_ir.paths import CHROMA_DIR

logger = logging.getLogger(__name__)

_chroma_rlock = threading.RLock()
_client: PersistentClient | None = None


def get_chroma_client() -> PersistentClient:
    """Return the process-wide persistent Chroma client, creating it on first use.

    Returns:
        A ``PersistentClient`` rooted at ``CHROMA_DIR`` (created if missing).

    Note:
        Caller should serialize mutating operations with ``chroma_operation_lock()``.
    """
    global _client
    with _chroma_rlock:
        if _client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _client = PersistentClient(path=str(CHROMA_DIR))
            logger.info("Chroma PersistentClient initialized | path=%s", CHROMA_DIR)
        return _client


def chroma_operation_lock():
    """Context manager that serializes Chroma I/O on the shared persistent directory.

    Uses an ``RLock`` so ``get_chroma_client()`` can be called while already holding
    the lock (nested acquire is allowed).

    Returns:
        The module-level ``threading.RLock`` (use as ``with chroma_operation_lock():``).
    """
    return _chroma_rlock

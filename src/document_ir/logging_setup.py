"""Central logging setup: uniform format, levels, and optional file output from env."""

from __future__ import annotations

import logging
import os
import sys

# Log record names look like document_ir.kb.indexing (module __name__); message text uses UI:/RAG:/… prefixes.
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "chromadb",
    "litellm",
    "openai",
    "urllib3",
    "multipart",
    "fsspec",
    "gradio",
    "uvicorn",
    "watchfiles",
    "datasets",
)


def configure_logging() -> None:
    """Configure the root logger once (idempotent if handlers already exist).

    Reads:

        LOG_LEVEL:
            DEBUG, INFO, WARNING, or ERROR for the root logger. Default: INFO.
        LOG_FILE:
            Optional path; duplicates all logs to that file (UTF-8, append).
        LOG_DEBUG:
            If ``1``, ``true``, or ``yes``, sets DEBUG on the ``document_ir`` logger
            tree (chunking detail, rewritten queries, etc.).
    """
    level_name = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        _apply_debug_packages()
        _quiet_third_party()
        return

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    root.addHandler(stderr)

    log_file = (os.environ.get("LOG_FILE") or "").strip()
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level)
    _apply_debug_packages()
    _quiet_third_party()


def _apply_debug_packages() -> None:
    """Set DEBUG on first-party packages when ``LOG_DEBUG`` is truthy."""
    flag = (os.environ.get("LOG_DEBUG") or "").lower()
    if flag not in ("1", "true", "yes"):
        return
    logging.getLogger("document_ir").setLevel(logging.DEBUG)


def _quiet_third_party() -> None:
    """Reduce noise from verbose third-party libraries."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

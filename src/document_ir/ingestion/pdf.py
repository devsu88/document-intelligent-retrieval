"""Convert PDF files to Markdown using pymupdf4llm."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from pymupdf4llm import to_markdown

logger = logging.getLogger(__name__)


def _pdf_to_markdown(path: Path) -> str:
    """Render a PDF on disk to a single Markdown string."""
    return to_markdown(path)


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hex digest of ``data``.

    Args:
        data: Raw bytes (e.g. PDF file contents).

    Returns:
        Lowercase hex string of length 64.
    """
    return hashlib.sha256(data).hexdigest()


def convert_pdf_bytes(data: bytes, suffix: str = ".pdf") -> str:
    """Convert PDF bytes to Markdown via a temporary file.

    The underlying library expects a filesystem path, so bytes are written to a
    temp file, converted, then the file is removed.

    Args:
        data: PDF file contents.
        suffix: Temp file suffix (default ``.pdf``).

    Returns:
        Markdown string produced from the PDF.

    Raises:
        Same exceptions as ``pymupdf4llm.to_markdown`` if conversion fails.
    """
    logger.info("PDF: conversion start | size_bytes=%s", len(data))
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        md = _pdf_to_markdown(tmp_path)
        logger.info("PDF: conversion OK | markdown_chars=%s", len(md))
        return md
    finally:
        tmp_path.unlink(missing_ok=True)


def convert_pdf_path(path: str | Path) -> str:
    """Convert a PDF file path to Markdown.

    Args:
        path: Path to an existing PDF file.

    Returns:
        Markdown string.
    """
    return _pdf_to_markdown(Path(path))

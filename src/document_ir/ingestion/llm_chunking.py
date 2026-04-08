"""Semantic document chunking via an LLM (headline + summary + original text).

Used by ``document_ir.kb.indexing`` when ``USE_LLM_DOCUMENT_CHUNKING`` is enabled.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tenacity.before_sleep import before_sleep_log

load_dotenv(override=True)

logger = logging.getLogger(__name__)

LLM_CHUNK_MODEL = os.getenv("LLM_CHUNK_MODEL", "openai/gpt-4.1-nano")
AVERAGE_CHUNK_SIZE = int(os.getenv("LLM_CHUNK_AVG_CHARS", "100"))
wait = wait_exponential(multiplier=1, min=10, max=240)
_log_retry = before_sleep_log(logger, logging.WARNING)


class Result(BaseModel):
    """One chunk as returned by the LLM, with page content and metadata."""

    page_content: str
    metadata: dict


class Chunk(BaseModel):
    """Structured fields for a single semantic chunk."""

    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document: dict) -> Result:
        """Build a ``Result`` with concatenated headline, summary, and original text."""
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    """Wrapper for the model's JSON array of chunks."""

    chunks: list[Chunk]


def make_prompt(document: dict) -> str:
    """Build the user prompt instructing how to split ``document`` into chunks.

    Args:
        document: Dict with keys ``type``, ``source``, and ``text``.

    Returns:
        Full prompt string for the completion API.
    """
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document type is: {document["type"]}
The document source label is: {document["source"]}

A chatbot will use these chunks to answer questions from the user's knowledge base.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


@retry(wait=wait, before_sleep=_log_retry)
def process_document(document: dict) -> list[Result]:
    """Call the LLM to split ``document`` into structured chunks.

    Args:
        document: Must include ``type``, ``source``, and ``text``.

    Returns:
        List of ``Result`` objects (``page_content`` + ``metadata`` per chunk).

    Raises:
        ValidationError: If the model output does not match ``Chunks``.
        API errors: Propagated from ``litellm.completion`` after retries exhaust.
    """
    messages = [{"role": "user", "content": make_prompt(document)}]
    response = completion(model=LLM_CHUNK_MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def _log_chunks_debug(source_label: str, chunks: list[str]) -> None:
    """Emit DEBUG logs with truncated previews for each chunk."""
    preview_len = int(os.getenv("LLM_CHUNK_LOG_PREVIEW", "400"))
    for i, text in enumerate(chunks):
        prev = text if len(text) <= preview_len else text[:preview_len] + "…"
        logger.debug(
            "LLM chunking: chunk[%s] | source=%r len=%s | %r",
            i,
            source_label,
            len(text),
            prev,
        )


def llm_split_document(markdown_text: str, source_label: str, doc_type: str = "markdown") -> list[str]:
    """Split Markdown into chunk strings using the configured LLM.

    Args:
        markdown_text: Full document body.
        source_label: Passed through to chunk metadata (e.g. filename).
        doc_type: Document type label in the prompt (default ``markdown``).

    Returns:
        List of ``page_content`` strings, one per LLM chunk.

    Raises:
        Same as ``process_document`` if the API or validation fails.
    """
    doc = {"type": doc_type, "source": source_label, "text": markdown_text}
    n_char = len(markdown_text)
    how_many = (n_char // AVERAGE_CHUNK_SIZE) + 1 if AVERAGE_CHUNK_SIZE > 0 else 1
    logger.info(
        "LLM chunking: split document | source=%r type=%s chars=%s avg_chars_env=%s suggested_min_chunks=%s",
        source_label,
        doc_type,
        n_char,
        AVERAGE_CHUNK_SIZE,
        how_many,
    )
    results = process_document(doc)
    out = [r.page_content for r in results]
    lens = [len(c) for c in out]
    logger.info(
        "LLM chunking: complete | source=%r n_chunk=%s char_lengths=%s",
        source_label,
        len(out),
        lens,
    )
    _log_chunks_debug(source_label, out)
    return out

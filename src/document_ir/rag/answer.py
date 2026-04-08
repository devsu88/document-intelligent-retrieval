"""RAG pipeline: query rewrite, vector retrieval, LLM rerank, and answer generation."""

import logging

from openai import OpenAI
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tenacity.before_sleep import before_sleep_log

from document_ir.kb.chroma_client import chroma_operation_lock, get_chroma_client
from document_ir.kb.indexing import CHROMA_DIR, EMBEDDING_MODEL
from document_ir.kb.store import chroma_collection_for_slug

load_dotenv(override=True)

logger = logging.getLogger(__name__)

MODEL = "openai/gpt-4.1-nano"
# MODEL = "groq/openai/gpt-oss-120b"

embedding_model = EMBEDDING_MODEL
wait = wait_exponential(multiplier=1, min=10, max=240)
_log_retry = before_sleep_log(logger, logging.WARNING)

openai = OpenAI()

RETRIEVAL_K = 20
FINAL_K = 10

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant answering questions using only the provided document extracts.
Your answer should be accurate, relevant, and complete. If the context does not contain the answer, say so.
For context, here are extracts from the user's knowledge base that may be relevant:
{context}

Answer the user's question based on this context.
"""


class Result(BaseModel):
    """A retrieved passage with text and Chroma metadata."""

    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    """LLM output: 1-based chunk indices ordered by relevance."""

    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait, before_sleep=_log_retry)
def rerank(question, chunks):
    """Reorder ``chunks`` by relevance to ``question`` using the chat model.

    Args:
        question: User question (same as original; used in the rerank prompt).
        chunks: List of ``Result`` in retrieval order.

    Returns:
        Same chunks reranked; length equals ``len(chunks)`` if the model follows instructions.

    Raises:
        ValidationError: If the model JSON does not match ``RankOrder``.
    """
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def make_rag_messages(question, history, chunks, system_prompt_template: str):
    """Build chat messages: system (context-filled), prior turns, then current question.

    Args:
        question: Latest user message.
        history: Prior messages as OpenAI-style dicts (``role``, ``content``).
        chunks: Retrieved ``Result`` objects; each becomes an "Extract from {source}:" block.
        system_prompt_template: Must contain ``{context}`` placeholder.

    Returns:
        List of message dicts ready for ``completion``.
    """
    parts = []
    for chunk in chunks:
        meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        src = meta.get("source", "?")
        parts.append(f"Extract from {src}:\n{chunk.page_content}")
    context = "\n\n".join(parts)
    system_prompt = system_prompt_template.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait, before_sleep=_log_retry)
def rewrite_query(question, history=None):
    """Produce a short search phrase from the question and optional history.

    Args:
        question: Current user question.
        history: Optional prior conversation (string or structured as expected by the prompt).

    Returns:
        Plain-text search query string from the model.

    Raises:
        API errors from ``litellm.completion`` after retries exhaust.
    """
    if history is None:
        history = []
    message = f"""
You are in a conversation with a user. You will search a knowledge base to answer their question.

Conversation history so far:
{history}

Current user question:
{question}

Respond only with a short, refined search query for the knowledge base.
It should be a VERY short specific phrase most likely to retrieve relevant passages.
IMPORTANT: Respond ONLY with the search query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    """Append chunks from ``reranked`` that are not already in ``chunks`` (by ``page_content``).

    Preserves order: base list first, then novel reranked items.
    """
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question, coll):
    """Embed ``question`` and query Chroma for top-``RETRIEVAL_K`` documents.

    Args:
        question: Query text (embedded with ``embedding_model``).
        coll: Chroma collection.

    Returns:
        List of ``Result`` (empty if no hits).
    """
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = coll.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    docs = results["documents"][0] or []
    metas = results["metadatas"][0] or []
    chunks = []
    for result in zip(docs, metas):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    logger.debug(
        "RAG: Chroma query | n_hits=%s question_preview=%r",
        len(chunks),
        (question[:120] + "…") if len(question) > 120 else question,
    )
    return chunks


def fetch_context(original_question, coll):
    """Retrieve with original and rewritten queries, merge, then LLM-rerank to ``FINAL_K``.

    Args:
        original_question: User's question.
        coll: Chroma collection for the active KB.

    Returns:
        At most ``FINAL_K`` ``Result`` instances, reranked by relevance.
    """
    rewritten_question = rewrite_query(original_question)
    logger.debug("RAG: rewritten query | rewritten=%r", rewritten_question[:200] if rewritten_question else "")
    chunks1 = fetch_context_unranked(original_question, coll)
    chunks2 = fetch_context_unranked(rewritten_question, coll)
    chunks = merge_chunks(chunks1, chunks2)
    logger.info(
        "RAG: retrieval | merge_chunks=%s (orig=%s + rewritten=%s)",
        len(chunks),
        len(chunks1),
        len(chunks2),
    )
    if not chunks:
        logger.info("RAG: no chunks from Chroma for this question")
        return []
    reranked = rerank(original_question, chunks)
    out = reranked[:FINAL_K]
    logger.info("RAG: rerank complete | in=%s out_top_k=%s", len(chunks), len(out))
    return out


@retry(wait=wait, before_sleep=_log_retry)
def answer_question_for_collection(
    question: str,
    history: list[dict],
    chroma_path: str,
    collection_name: str,
    system_prompt_template: str | None = None,
) -> tuple[str, list]:
    """Run the full RAG loop on an arbitrary Chroma collection.

    Args:
        question: User question for this turn.
        history: Prior messages (OpenAI format); excluded from retrieval, included in generation.
        chroma_path: Persist path (logged / reserved for callers; client is singleton).
        collection_name: Chroma collection name (e.g. ``kb_{slug}``).
        system_prompt_template: Optional override; must include ``{context}``.

    Returns:
        Tuple of (assistant message text, list of context ``Result`` used in the prompt).

    Raises:
        Same as Chroma / embedding / ``completion`` on failure.
    """
    prompt = system_prompt_template or DEFAULT_SYSTEM_PROMPT
    logger.info(
        "RAG: start answer_question_for_collection | collection=%s question_len=%s history_msgs=%s",
        collection_name,
        len(question),
        len(history),
    )
    with chroma_operation_lock():
        client = get_chroma_client()
        coll = client.get_collection(collection_name)
        chunks = fetch_context(question, coll)
    logger.info("RAG: LLM answer generation | n_context_chunks=%s", len(chunks))
    messages = make_rag_messages(question, history, chunks, prompt)
    response = completion(model=MODEL, messages=messages)
    logger.info("RAG: LLM answer complete | collection=%s", collection_name)
    return response.choices[0].message.content, chunks


@retry(wait=wait, before_sleep=_log_retry)
def answer_question(
    question: str,
    history: list | None = None,
    *,
    kb_slug: str,
    system_prompt_template: str | None = None,
) -> tuple[str, list]:
    """Answer using the SQLite-managed KB identified by ``kb_slug``.

    Args:
        question: User question.
        history: Optional prior turns for the chat model.
        kb_slug: Knowledge base slug (non-empty after strip).
        system_prompt_template: Optional system prompt with ``{context}``.

    Returns:
        ``(answer_text, context_chunks)`` as in ``answer_question_for_collection``.

    Raises:
        ValueError: If ``kb_slug`` is missing or whitespace-only.
    """
    if history is None:
        history = []
    slug = (kb_slug or "").strip()
    if not slug:
        raise ValueError("kb_slug is required for answer_question")
    logger.info("Chat RAG: kb_slug=%s", slug)
    return answer_question_for_collection(
        question,
        history,
        str(CHROMA_DIR),
        chroma_collection_for_slug(slug),
        system_prompt_template,
    )


GENERIC_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

"""Gradio UI: knowledge base management, PDF upload/indexing, and RAG chat."""

from dotenv import load_dotenv

load_dotenv(override=True)

from document_ir.logging_setup import configure_logging

configure_logging()

import logging
from pathlib import Path

import gradio as gr

from document_ir.kb import store as kb_store
from document_ir.ingestion.pdf import convert_pdf_bytes, sha256_bytes
from document_ir.kb.indexing import (
    drop_kb_collection,
    index_document_file,
    remove_document_vectors,
)
from document_ir.kb.store import (
    add_document,
    create_knowledge_base,
    delete_document_row,
    delete_knowledge_base,
    get_document,
    get_knowledge_base_by_slug,
    list_documents,
    list_knowledge_bases,
)
from document_ir.rag.answer import answer_question

logger = logging.getLogger(__name__)


def format_context(context):
    """Render retrieved chunks as HTML for the context panel.

    Args:
        context: Iterable of objects with ``page_content`` and ``metadata`` (``source`` key).

    Returns:
        HTML string with headings and source labels.
    """
    result = "<h2 style='color: #ff7800;'>Retrieved context</h2>\n\n"
    for doc in context:
        src = doc.metadata.get("source", "?")
        result += f"<span style='color: #ff7800;'>Source: {src}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def _chat_content_to_text(content) -> str:
    """Normalize Gradio Chatbot message content to a plain string.

    Gradio 6+ may pass ``content`` as a string or a list of typed parts (dicts with
    ``type``/``text``).

    Args:
        content: Message payload from the chatbot history.

    Returns:
        Concatenated text, or empty string if ``content`` is None.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text") or "")
                elif "text" in item:
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _history_for_llm(history: list | None) -> list[dict]:
    """Convert chat history to OpenAI-style messages with string ``content`` only.

    Args:
        history: Gradio chat messages (dicts with ``role`` and ``content``).

    Returns:
        List of dicts suitable for LiteLLM / OpenAI; entries without ``role`` are skipped.
    """
    if not history:
        return []
    out = []
    for msg in history:
        if not isinstance(msg, dict) or "role" not in msg:
            continue
        out.append(
            {"role": msg["role"], "content": _chat_content_to_text(msg.get("content"))}
        )
    return out


def kb_dropdown_choices():
    """Build Gradio dropdown options for all knowledge bases.

    Returns:
        List of ``(label, value)`` tuples: ``"{name} ({slug})", slug``.
    """
    return [(f"{k.name} ({k.slug})", k.slug) for k in list_knowledge_bases()]


def _kb_slug_if_valid(slug) -> str | None:
    """Return ``slug`` only if it still exists in the database.

    Avoids crashes when the UI holds a stale slug after external deletes.

    Args:
        slug: Candidate slug (any truthy value is coerced with ``str``).

    Returns:
        The slug string if found, else ``None``.
    """
    if not slug:
        return None
    kb = get_knowledge_base_by_slug(str(slug))
    return kb.slug if kb else None


def _kb_dd_sync_on_load(current_slug):
    """On page load, refresh KB dropdown choices and clear invalid selection.

    Args:
        current_slug: Value bound from the KB dropdown.

    Returns:
        ``gr.update`` for the dropdown with refreshed ``choices`` and ``value``.
    """
    kb_store.init_db()
    choices = kb_dropdown_choices()
    valid = _kb_slug_if_valid(current_slug)
    if valid:
        return gr.update(choices=choices, value=valid)
    return gr.update(choices=choices, value=None)


def _kb_choices_update(value=None):
    """Return a dropdown update with latest KB list and optional new value."""
    return gr.update(choices=kb_dropdown_choices(), value=value)


def _upload_path(p):
    """Extract a filesystem path from a Gradio File value (dict or object).

    Args:
        p: File component value (may be dict with ``path``/``name``, or object with ``name``).

    Returns:
        Path string, or ``None`` if not resolvable.
    """
    if p is None:
        return None
    if isinstance(p, dict):
        return p.get("path") or p.get("name")
    return getattr(p, "name", None) or p


def on_index_files(files, new_kb_name, slug):
    """Index uploaded PDFs into the selected KB or a newly created one.

    Skips files whose content hash already exists in the target KB.

    Args:
        files: Single file or list from Gradio File (PDF).
        new_kb_name: If non-empty, creates a KB with this name first.
        slug: Active KB slug from dropdown (used when ``new_kb_name`` is empty).

    Returns:
        Tuple ``(kb_dropdown_update, message, docs_markdown_update, doc_picker_update)``.
    """
    kb_store.init_db()
    new_kb_name = (new_kb_name or "").strip()
    logger.info(
        "UI: index files | new_kb_name=%r slug=%r n_files=%s",
        new_kb_name,
        slug,
        len(files) if isinstance(files, list) else (1 if files else 0),
    )
    if not files:
        s = _kb_slug_if_valid(slug)
        return (
            gr.update(),
            "Select at least one PDF file.",
            docs_table_update(s),
            doc_id_choices(s),
        )
    paths = files if isinstance(files, list) else [files]

    if new_kb_name:
        try:
            kb = create_knowledge_base(new_kb_name)
        except Exception as e:
            logger.exception("UI: KB creation failed | name=%r", new_kb_name)
            return (
                gr.update(),
                f"Could not create the knowledge base: {e}",
                docs_table_update(_kb_slug_if_valid(slug)),
                doc_id_choices(_kb_slug_if_valid(slug)),
            )
        logger.info("UI: KB created for upload | slug=%s", kb.slug)
        u = kb_dropdown_choices()
        kb_dd_upd = gr.update(choices=u, value=kb.slug)
        target_slug = kb.slug
    else:
        resolved = _kb_slug_if_valid(slug)
        if not resolved:
            u = kb_dropdown_choices()
            kb_dd_upd = gr.update(choices=u, value=None) if slug else gr.update()
            msg = (
                "The selected knowledge base is no longer available or none was chosen. "
                "Pick one from the menu above or enter a name to create a new knowledge base."
            )
            return (
                kb_dd_upd,
                msg,
                docs_table_update(None),
                doc_id_choices(None),
            )
        kb_dd_upd = gr.update()
        target_slug = resolved

    kb = get_knowledge_base_by_slug(target_slug)
    n_new = 0
    dup_names: list[str] = []
    errs: list[str] = []
    for p in paths:
        path = _upload_path(p)
        if not path:
            continue
        try:
            data = Path(path).read_bytes()
            h = sha256_bytes(data)
            if any(d.content_hash == h for d in list_documents(kb.id)):
                dup_names.append(Path(path).name)
                continue
            md = convert_pdf_bytes(data)
            orig = Path(path).name
            doc = add_document(kb.id, orig, md, h)
            index_document_file(kb.slug, doc)
            n_new += 1
            logger.info(
                "UI: PDF indexed | slug=%s doc_id=%s file=%r",
                kb.slug,
                doc.id,
                orig,
            )
        except Exception as e:
            logger.exception("UI: PDF indexing failed | path=%s", path)
            errs.append(f"{Path(path).name}: {e}")

    parts: list[str] = []
    if new_kb_name:
        parts.append(f'Knowledge base "{new_kb_name}" created.')
    if n_new:
        parts.append(f"Added and indexed {n_new} document(s).")
    if dup_names:
        listed = ", ".join(dup_names)
        if n_new == 0 and not errs:
            parts.append(
                f"All selected files ({listed}) are already in this knowledge base "
                "(same content); nothing was re-uploaded."
            )
        else:
            parts.append(
                "Some files were already present (same content) and were not re-uploaded: "
                + listed
                + "."
            )
    if errs:
        parts.append("Errors: " + "; ".join(errs))
    if not parts:
        msg = "No files were processed."
    else:
        msg = " ".join(parts)
    return (
        kb_dd_upd,
        msg,
        docs_table_update(target_slug),
        doc_id_choices(target_slug),
    )


def on_delete_kb(slug):
    """Delete the selected knowledge base: Chroma collection, DB row, and files.

    Args:
        slug: KB slug from dropdown.

    Returns:
        ``(kb_dropdown_update, docs_markdown_update, feedback_message)``.
    """
    kb_store.init_db()
    slug = _kb_slug_if_valid(slug)
    logger.info("UI: delete knowledge base | slug=%r", slug)
    if not slug:
        return (
            _kb_choices_update(None),
            docs_table_update(None),
            "Select a knowledge base to delete.",
        )
    kb = get_knowledge_base_by_slug(slug)
    if not kb:
        return (
            _kb_choices_update(None),
            docs_table_update(None),
            "Knowledge base not found.",
        )
    drop_kb_collection(kb.slug)
    delete_knowledge_base(kb.id)
    logger.info("UI: KB deleted | slug=%s name=%r", kb.slug, kb.name)
    u = kb_dropdown_choices()
    return (
        gr.update(choices=u, value=None),
        docs_table_update(None),
        f'Deleted the knowledge base "{kb.name}".',
    )


def docs_table_update(slug):
    """Markdown listing documents for the KB identified by ``slug``.

    Args:
        slug: KB slug, or falsy for empty state.

    Returns:
        ``gr.Markdown`` update with placeholder or numbered document list.
    """
    if not slug:
        return gr.update(value="*Select a knowledge base.*")
    kb = get_knowledge_base_by_slug(slug)
    if not kb:
        return gr.update(value="")
    rows = list_documents(kb.id)
    if not rows:
        return gr.update(
            value="_No PDFs in this knowledge base. Use the upload area above to add some._"
        )
    lines = [
        "**Indexed documents**\n",
    ]
    for i, d in enumerate(rows, 1):
        lines.append(f"{i}. {d.original_filename}")
    return gr.update(value="\n".join(lines))


def doc_id_choices(slug):
    """Dropdown update: document choices for delete picker.

    Args:
        slug: KB slug.

    Returns:
        ``gr.update`` with ``(filename, doc_id)`` choices and cleared selection.
    """
    kb = get_knowledge_base_by_slug(slug) if slug else None
    if not kb:
        return gr.update(choices=[], value=None)
    opts = [(d.original_filename, d.id) for d in list_documents(kb.id)]
    return gr.update(choices=opts, value=None)


def on_delete_doc(slug, doc_id):
    """Remove one document from Chroma and SQLite.

    Args:
        slug: Active KB slug.
        doc_id: Selected document UUID.

    Returns:
        ``(feedback_message, docs_markdown_update, doc_picker_update)``.
    """
    kb_store.init_db()
    logger.info("UI: delete document | slug=%r doc_id=%r", slug, doc_id)
    slug = _kb_slug_if_valid(slug)
    if not slug or not doc_id:
        return "Select a knowledge base and a document.", docs_table_update(slug), doc_id_choices(slug)
    kb = get_knowledge_base_by_slug(slug)
    if not kb:
        return "Knowledge base not found.", docs_table_update(None), doc_id_choices(None)
    doc = get_document(doc_id)
    if not doc or doc.kb_id != kb.id:
        return "Invalid document for this knowledge base.", docs_table_update(slug), doc_id_choices(slug)
    remove_document_vectors(kb.slug, doc.id)
    delete_document_row(doc.id)
    logger.info("UI: document deleted OK | slug=%s doc_id=%s", kb.slug, doc.id)
    return "Document deleted.", docs_table_update(slug), doc_id_choices(slug)


def main():
    """Build and launch the Gradio Blocks app (in-browser)."""
    logger.info("Starting Document Intelligent Retrieval UI")
    kb_store.init_db()

    def chat_append_user(message, history, kb_slug):
        """Append the user message to history and show a loading placeholder in context.

        Returns cleared input, updated history, context panel HTML, and optional KB dropdown fix
        if the stored slug is stale.
        """
        history = list(history or [])
        text = (message or "").strip()
        raw_slug = kb_slug
        kb_slug_valid = _kb_slug_if_valid(kb_slug)
        stale = bool(raw_slug) and kb_slug_valid is None
        kb_dd_fix = (
            gr.update(choices=kb_dropdown_choices(), value=None)
            if stale
            else gr.update()
        )
        if not text:
            return gr.update(), history, gr.update(), kb_dd_fix
        history.append({"role": "user", "content": text})
        return "", history, "<p><i>Working…</i></p>", kb_dd_fix

    def chat_generate_assistant(history, kb_slug):
        """Generate assistant reply and retrieved context after the user turn is visible.

        Calls ``answer_question`` with prior history (excluding the latest user bubble for
        retrieval context construction inside RAG).

        Returns:
            Updated history, context HTML, and a no-op KB dropdown update.
        """
        history = list(history or [])
        if not history or history[-1].get("role") != "user":
            return history, gr.update(), gr.update()
        text = _chat_content_to_text(history[-1].get("content"))
        kb_slug = _kb_slug_if_valid(kb_slug)
        if not kb_slug:
            logger.info("Chat: question with no valid KB after user message append")
            history.append(
                {
                    "role": "assistant",
                    "content": "Select an active knowledge base from the menu above to ask questions about your documents.",
                }
            )
            return history, "<p><i>Select a knowledge base.</i></p>", gr.update()
        prior = _history_for_llm(history[:-1])
        logger.info(
            "Chat: sending question | kb_slug=%s len=%s history_turns=%s",
            kb_slug,
            len(text),
            len(prior),
        )
        try:
            answer, context = answer_question(text, prior, kb_slug=kb_slug)
        except Exception as e:
            logger.exception("Chat: answer_question error | kb_slug=%s", kb_slug)
            answer = f"Could not complete the request: {e}"
            context = []
        else:
            logger.info(
                "Chat: answer ready | kb_slug=%s context_chunks=%s",
                kb_slug,
                len(context),
            )
        history.append({"role": "assistant", "content": answer})
        ctx_html = format_context(context) if context else "<p><i>No passages retrieved.</i></p>"
        return history, ctx_html, gr.update()

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Document Intelligent Retrieval") as ui:
        gr.Markdown(
            "# Document Intelligent Retrieval\n"
            "Build knowledge bases from PDFs, keep them up to date, and query them with RAG."
        )
        kb_dd = gr.Dropdown(
            label="Active knowledge base",
            choices=kb_dropdown_choices(),
            value=None,
            interactive=True,
            allow_custom_value=True,
        )

        with gr.Tabs():
            with gr.Tab("Documents"):
                gr.Markdown(
                    "**Upload and index:** choose PDFs below. "
                    "Use **Active knowledge base** (above) for an existing KB, "
                    "or fill in **New knowledge base name** to create one on the fly."
                )
                new_kb_name = gr.Textbox(
                    label="New knowledge base name (optional)",
                    placeholder="Leave empty to use the knowledge base selected above",
                    lines=1,
                )
                upload_files = gr.File(
                    label="PDFs to upload and index",
                    file_count="multiple",
                    file_types=[".pdf"],
                )
                index_btn = gr.Button("Upload and index", variant="primary")
                docs_tab_feedback = gr.Textbox(
                    label="Messages",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("---")
                gr.Markdown("### Delete documents from the selected knowledge base")
                docs_md = gr.Markdown(value="*Select a knowledge base.*")
                doc_pick = gr.Dropdown(
                    label="Document",
                    choices=[],
                    interactive=True,
                )
                del_doc_btn = gr.Button("Delete document", variant="stop")

                gr.Markdown("### Delete knowledge base")
                gr.Markdown(
                    "_Permanently deletes the knowledge base selected in the menu above, "
                    "its documents, and the vector index._"
                )
                delete_kb_btn = gr.Button("Delete knowledge base", variant="stop")

                def sync_doc_dd(slug):
                    """Refresh document dropdown from KB slug (chain after KB delete)."""
                    return doc_id_choices(slug)

                def on_kb_change(slug):
                    """When KB selection changes, refresh doc list and clear tab feedback."""
                    s = _kb_slug_if_valid(slug)
                    return docs_table_update(s), doc_id_choices(s), gr.update(value="")

                index_btn.click(
                    on_index_files,
                    inputs=[upload_files, new_kb_name, kb_dd],
                    outputs=[kb_dd, docs_tab_feedback, docs_md, doc_pick],
                )

                delete_kb_btn.click(
                    on_delete_kb,
                    inputs=[kb_dd],
                    outputs=[kb_dd, docs_md, docs_tab_feedback],
                ).then(sync_doc_dd, inputs=[kb_dd], outputs=[doc_pick])

                kb_dd.change(
                    on_kb_change,
                    inputs=[kb_dd],
                    outputs=[docs_md, doc_pick, docs_tab_feedback],
                )

                del_doc_btn.click(
                    on_delete_doc,
                    inputs=[kb_dd, doc_pick],
                    outputs=[docs_tab_feedback, docs_md, doc_pick],
                )

            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=600,
                            buttons=["copy"],
                        )
                        message = gr.Textbox(
                            label="Question",
                            placeholder="Ask a question about the documents in the knowledge base…",
                            show_label=False,
                            submit_btn=True,
                        )
                    with gr.Column(scale=1):
                        context_markdown = gr.Markdown(
                            label="Retrieved context",
                            value="<p><i>Retrieved context will appear here.</i></p>",
                            container=True,
                            height=600,
                        )

                message.submit(
                    chat_append_user,
                    inputs=[message, chatbot, kb_dd],
                    outputs=[message, chatbot, context_markdown, kb_dd],
                ).then(
                    chat_generate_assistant,
                    inputs=[chatbot, kb_dd],
                    outputs=[chatbot, context_markdown, kb_dd],
                )

        ui.load(_kb_dd_sync_on_load, inputs=[kb_dd], outputs=[kb_dd])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()

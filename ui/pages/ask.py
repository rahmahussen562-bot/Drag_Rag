"""
ui/pages/ask.py — the RAG chat. The page the product is actually judged on.

Flow, unchanged from the single-file version:
    interaction pre-check (when ≥2 drugs are named)
    → staged retrieval + generation via 07_prompting.answer_question
    → answer, provenance badges, source cards
    → optional evaluation panel

This module owns FLOW only. Every RAG decision — how to score, when to refuse,
what enters the prompt — lives in the numbered stage modules.
"""
from __future__ import annotations

import re
from datetime import datetime

import streamlit as st

from ui.components import (FIRST_LABEL, answer_badges, error_box,
                           fallback_warning, finalise, make_on_stage,
                           render_evaluation, render_sources)
from ui.pages.audit import record
from ui.pages.interactions import render_interaction_block

# ── Interaction-intent detection for the chat path ───────────────────────────
_INTENT_WORDS = {"interact", "interacts", "interaction", "interactions", "together",
                 "combine", "combined", "combination", "mix", "concurrent",
                 "concurrently", "coadminister", "coadministration", "alongside"}


def interaction_intent(q: str) -> bool:
    ql = q.lower()
    tokens = set(re.findall(r"[a-z]+", ql))
    return bool(tokens & _INTENT_WORDS) or "safe to take" in ql or "take with" in ql


def render(ctx) -> None:
    persona = ctx.persona
    st.title("Ask a question")

    if persona is not None:
        # State the persona in the page, not just the sidebar. The whole claim is
        # that the same question gets a different answer depending on who asks —
        # that is only legible if the reader can see which one is active.
        fields = ("all label sections" if persona.allowed_fields is None
                  else ", ".join(sorted(persona.allowed_fields)))
        st.caption(f"**Answering as: {persona.label}** — {persona.description}  \n"
                   f"Reads: {fields} · k={persona.k}")

    has_uploads = bool(st.session_state.uploaded_docs)
    st.caption("e.g. *“How does metformin work?”*, *“Does warfarin interact with naproxen?”*"
               + (", or ask about your uploaded documents." if has_uploads else ""))

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg.get("chunks"):
                render_sources(msg["chunks"])

    picked = None
    if not st.session_state.messages:
        st.markdown("")
        st.caption("TRY ONE")
        # The third example is chosen deliberately: aspirin is absent from the
        # corpus, so it demonstrates the refusal behaviour rather than hiding it.
        examples = [("💊", "How does metformin work?"),
                    ("⚠️", "Does warfarin interact with naproxen?"),
                    ("🛑", "Side effects of aspirin?")]
        for col, (icon, example) in zip(st.columns(3), examples):
            if col.button(f"{icon}  {example}", use_container_width=True):
                picked = example

    prompt_text = st.chat_input("Ask about a drug or your documents…") or picked
    if not prompt_text:
        return

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        # Structured interaction block first, when the question names ≥2 drugs.
        interaction_md = ""
        if ctx.interactions_available and interaction_intent(prompt_text):
            matched = sorted(ctx.retriever.ground(prompt_text)["matched"])
            if len(matched) >= 2:
                res = ctx.idb.check_list(matched)
                interaction_md = ("#### ⚠️ Interaction check\n"
                                  + render_interaction_block(res) + "\n\n---\n")
                st.markdown(interaction_md, unsafe_allow_html=True)

        # ── Staged progress ──────────────────────────────────────────────────
        result = None
        with st.status(FIRST_LABEL, expanded=False) as box:
            try:
                # The persona owns the answer. It supplies the field policy, the
                # packing order, the prompt and the post-generation enforcement;
                # stage 07 still runs the loop, so every caller shares one path.
                result = ctx.agent.answer(
                    prompt_text, ctx.retriever, k=ctx.top_k, use_llm=True,
                    prefer=ctx.prefer, on_stage=make_on_stage(box))
            except Exception as exc:  # noqa: BLE001 — last line of defence
                # No traceback ever reaches the user. This is the only place a
                # pipeline exception can surface, and it is turned into a
                # sentence plus a next action, with the detail folded away.
                box.update(label="⚠️ Could not complete this request",
                           state="error", expanded=False)
                error_box(
                    "Something went wrong while answering.",
                    detail=f"{type(exc).__name__}: {exc}",
                    action="Try rephrasing your question, or reload the page. "
                           "Your uploaded documents are preserved.")

            if result is not None:
                finalise(box, result)

        # Answer renders inside the SAME assistant bubble as the status box — a
        # second st.chat_message would draw a second avatar for one reply.
        if result is None:
            return

        # A configured provider that failed is an incident, not a mode. Show the
        # provider's own error ABOVE the answer so a bad key or a retired model
        # slug can never hide behind a plausible-looking sourced answer.
        if result.llm_failed:
            error_box(
                "LLM generation failed — showing retrieved sources instead.",
                detail=result.llm_error,
                action="Check `OPENROUTER_API_KEY` in **Settings → Secrets**. "
                       "Retrieval and citations below are unaffected.")

        fallback_warning(result)

        # Red-flag triage gets its own treatment: it is not a refusal-with-
        # suggestions, it is an instruction to stop using the software.
        if result.status == "red_flag":
            st.error(result.text, icon="🚑")
        else:
            st.markdown(f'<div class="answer">{result.text}</div>',
                        unsafe_allow_html=True)

        badges = answer_badges(result)
        if badges:
            st.markdown(badges, unsafe_allow_html=True)

        render_sources(result.chunks, query=prompt_text,
                       alpha=ctx.retriever.alpha,
                       reranked=ctx.retriever.reranker is not None)

        if ctx.evaluation_mode:
            render_evaluation(result)

        # Every answer is filed, including refusals — a refusal is exactly the
        # kind of event a pharmacy needs a record of.
        record(result, prompt_text,
               ctx.persona.label if ctx.persona else "—",
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Guarded by `result is not None`: when the pipeline raised, there is no
        # answer to persist, and appending would crash on the next rerun while
        # replaying history.
        st.session_state.messages.append({
            "role": "assistant",
            "content": interaction_md + result.text,
            "chunks": result.chunks,
        })

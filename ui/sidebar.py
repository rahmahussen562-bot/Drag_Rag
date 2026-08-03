"""
ui/sidebar.py — navigation, retrieval settings and the live status panel.

The sidebar is where the app's settings are resolved, so it is also where the
`AppContext` every page consumes is built. Returning (page_label, ctx) keeps the
router in `streamlit_app.py` down to a dict lookup.

# WHY does the status panel exist?
# Because every degradation in this app is silent by design — BM25-only when torch
# is missing, extractive when no LLM key is present, a NumPy matmul when faiss has
# no wheel. Each of those is the correct behaviour AND invisible from the answers
# alone. The panel is what turns "it still works" into "it still works, and here
# is exactly which path it took".
"""
from __future__ import annotations

import streamlit as st

import llm_providers
from agents import DEFAULT_AGENT, available, config_of, get_agent
from ui.components import error_box
from ui.pages import PAGES
from ui.state import AppContext, get_retriever, get_rxnorm, load_interactions
from ui.theme import pill, pills, stat_row


def render():
    """Draw the sidebar. Returns (page_label, AppContext) for the router."""
    with st.sidebar:
        st.title("💊 PhARMA RAG")
        st.caption("Grounded drug Q&A over an offline corpus + your own documents. "
                   "Every answer is retrieved, cited and verifiable.")

        page_label = st.radio("Tool", list(PAGES), index=0)

        # ── Persona ──────────────────────────────────────────────────────────
        # The single most important control in the app: it changes which label
        # sections are readable, how many sources are retrieved, which prompt is
        # used, and what is enforced on the answer afterwards.
        st.subheader("Who is asking?")
        agent_ids = available()
        agent_id = st.radio(
            "Persona",
            agent_ids,
            format_func=lambda a: config_of(a).label,
            index=agent_ids.index(DEFAULT_AGENT) if DEFAULT_AGENT in agent_ids else 0,
            label_visibility="collapsed")
        cfg = config_of(agent_id)
        st.caption(cfg.description)

        st.subheader("Retrieval")
        use_embeddings = st.toggle(
            "Semantic + hybrid retrieval", value=True,
            help="Sentence embeddings blended with BM25. Off = fast BM25 keyword search only.")
        use_reranker = st.toggle(
            "Cross-encoder reranker", value=False,
            help="Highest quality, slower. Re-scores the top 40 candidates jointly with the query.")
        # Default k follows the persona (patient 4, practitioner 8) but stays
        # user-overridable — the slider is how the divergence is demonstrated.
        top_k = st.slider("Sources per answer (k)", 3, 10, cfg.k,
                          key=f"k_{agent_id}")
        evaluation_mode = st.toggle(
            "🔬 Evaluation mode", value=False,
            help="Show the retrieved chunks, component scores, the exact prompt, "
                 "and groundedness verification.")

        st.subheader("Answer generation")
        status = llm_providers.llm_status()
        provider_choice = st.selectbox(
            "LLM provider",
            ["auto"] + llm_providers.PROVIDER_ORDER,
            help="`auto` uses the priority cascade: OpenRouter → Gemini → Groq → Ollama. "
                 "A pinned provider still falls back if it fails.")
        prefer = None if provider_choice == "auto" else provider_choice

        # Building the index touches disk, the corpus JSON and (optionally) the
        # embedding model. If any of that fails the app cannot function at all, so
        # it gets a plain explanation and a stop — never a traceback in the sidebar.
        try:
            retriever = get_retriever(use_embeddings, use_reranker)
        except Exception as exc:  # noqa: BLE001 — startup must not crash
            error_box(
                "Could not build the retrieval index.",
                detail=f"{type(exc).__name__}: {exc}",
                action="Check that `data/drugs_large.json` and `data/doc_embeddings.npy` "
                       "are present in the deployment, then reload.")
            st.stop()

        idb = load_interactions()

        use_rxnorm = st.toggle(
            "RxNorm name normalization", value=False,
            help="Resolve brand names/typos online (Tylenol → acetaminophen). Needs internet.")
        if use_rxnorm and not isinstance(idb, Exception):
            idb.rxnorm = get_rxnorm()

        _status_panel(retriever, idb, status)

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("⚠️ Educational demo. Not medical advice.")

    ctx = AppContext(
        retriever=retriever, idb=idb, top_k=top_k, prefer=prefer,
        evaluation_mode=evaluation_mode, use_embeddings=use_embeddings,
        use_reranker=use_reranker, agent=get_agent(agent_id))
    return page_label, ctx


def _status_panel(retriever, idb, status) -> None:
    st.subheader("Status")
    n_uploaded = sum(len(v) for v in st.session_state.uploaded_docs.values())
    dense_backend = (retriever.store.dense.backend if retriever.store.has_dense
                     else "BM25 only")

    rows = [
        stat_row("Chunks indexed", f"{len(retriever.df):,}"),
        stat_row("Sources", f"{retriever.df['drug'].nunique():,}"),
    ]
    if n_uploaded:
        rows.append(stat_row(
            "Your documents",
            f"{len(st.session_state.uploaded_docs)} file(s) · {n_uploaded} section(s)"))
    rows.append(stat_row("Vector store", dense_backend,
                         tone="ok" if retriever.store.has_dense else "warn"))
    rows.append(stat_row("Lexical", retriever.store.lexical.backend, tone="ok"))
    if retriever.reranker is not None:
        rows.append(stat_row("Reranker", "cross-encoder", tone="ok"))
    rows.append(stat_row(
        "Interactions",
        "not loaded" if isinstance(idb, Exception) else f"{len(idb.pairs):,} pairs",
        tone="err" if isinstance(idb, Exception) else "ok"))
    rows.append(stat_row("LLM", status["active"] or "not configured",
                         tone="ok" if status["active"] else "warn"))
    st.markdown("".join(rows), unsafe_allow_html=True)

    if status["active"]:
        st.caption(f"Model: `{status['model']}`")
    else:
        st.info("No LLM key found — answers fall back to **retrieved text**, still "
                "cited. Add `OPENROUTER_API_KEY` in Streamlit **Settings → Secrets** "
                "(or a local `.streamlit/secrets.toml`) for generated answers.",
                icon="ℹ️")

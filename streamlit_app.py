"""
streamlit_app.py — STAGE 8 (final):  **Streamlit UI**

    Raw Documents → Preprocessing → Chunking → Vector Representation → Vector Store
    → Context Retrieval → Prompt Building → LLM Generation → >>> Streamlit UI <<<

# WHY is this file thin?
# It contains NO pipeline logic. Every RAG decision — how to chunk, how to score,
# when to refuse, what to put in the prompt — lives in the numbered stage modules
# and is testable from the command line without Streamlit. This file only:
#   1. builds the pipeline objects (cached, so the 15k-chunk corpus is encoded once),
#   2. renders three tools over them,
#   3. exposes the internals in "evaluation mode" so the pipeline is inspectable.
# That separation is what lets `python 06_retrieve_context.py` and the eval harness
# exercise exactly the same code the UI runs.
#
# This app is the evolution of the previous `app.py`, not a rewrite: the sidebar,
# the Ask chat, the DDInter interaction checker and the graceful-degradation
# behaviour are carried over. What is new is document upload, evaluation mode, and
# the multi-provider LLM layer.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import load  # noqa: E402

import llm_providers  # noqa: E402

documents_stage = load("01_documents")
chunking = load("03_chunking")
representation = load("04_vector_representation")
vector_store = load("05_create_vector_store")
retrieval = load("06_retrieve_context")
prompting = load("07_prompting")

from interactions import InteractionDB, DDINTER_CITATION  # noqa: E402

st.set_page_config(page_title="PhARMA RAG — Grounded Drug Q&A",
                   page_icon="💊", layout="centered")

DISCLAIMER = ("⚠️ **Educational tool — not medical advice.** Interaction and drug data can be "
              "incomplete. Always confirm with a pharmacist or prescriber.")

# ── Interaction-intent detection for the chat path (carried over from app.py) ─
_INTENT_WORDS = {"interact", "interacts", "interaction", "interactions", "together",
                 "combine", "combined", "combination", "mix", "concurrent",
                 "concurrently", "coadminister", "coadministration", "alongside"}


def interaction_intent(q: str) -> bool:
    ql = q.lower()
    tokens = set(re.findall(r"[a-z]+", ql))
    return bool(tokens & _INTENT_WORDS) or "safe to take" in ql or "take with" in ql


# ═════════════════════════════════════════════════════════════════════════════
# Cached pipeline construction
#
# # WHY split "base corpus" from "retriever"?
# Encoding 14,955 chunks costs ~40s (or a 23 MB np.load from cache). That work is
# identical for every user and every session, so it is @st.cache_resource'd once.
# The retriever, by contrast, changes whenever THIS user uploads a document, so it
# lives in session_state and is rebuilt only when the upload set actually changes.
# Caching them together would either re-encode the whole corpus on every upload or
# leak one user's uploads into another's session.
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading drug corpus and building the index…")
def load_base(use_embeddings: bool):
    """Chunk + encode the built-in drug corpus. Shared across all sessions."""
    records = documents_stage.load_drug_records()
    docs = documents_stage.documents_from_drug_records(records)
    df = chunking.build_chunks(docs)

    embedder = representation.Embedder() if use_embeddings else None
    vectors, info = representation.embed_corpus(
        df["text"].tolist(), embedder=embedder, use_dense=use_embeddings)
    return {"records": records, "df": df, "vectors": vectors,
            "embedder": embedder, "info": info}


@st.cache_resource(show_spinner="Loading interaction database (DDInter)…")
def load_interactions():
    try:
        return InteractionDB()
    except Exception as exc:
        return exc  # surfaced in the UI rather than crashing the app


@st.cache_resource
def get_rxnorm():
    from rxnorm import RxNormResolver
    return RxNormResolver()


def uploads_signature() -> tuple:
    """Identity of the current upload set — the rebuild trigger."""
    return tuple(sorted(st.session_state.get("uploaded_docs", {}).keys()))


def get_retriever(use_embeddings: bool, use_reranker: bool):
    """Build (or reuse) a retriever over the drug corpus + this session's uploads."""
    signature = (uploads_signature(), use_embeddings, use_reranker)
    if st.session_state.get("retriever_sig") == signature:
        return st.session_state["retriever"]

    base = load_base(use_embeddings)
    df, vectors = base["df"], base["vectors"]
    embedder = base["embedder"]

    upload_docs = []
    for docs in st.session_state.get("uploaded_docs", {}).values():
        upload_docs.extend(docs)

    if upload_docs:
        # Chunk the uploads through the SAME stage-03 code as the drug corpus, then
        # encode ONLY the new chunks and stack them onto the cached matrix. Never
        # re-encode the 15k base chunks — that is the whole point of the split.
        upload_df = chunking.build_chunks(upload_docs)
        df = chunking.merge_chunks(base["df"], upload_df)
        if vectors is not None and embedder is not None and embedder.available:
            with st.spinner(f"Embedding {len(upload_df)} chunks from your documents…"):
                upload_vectors = embedder.encode(upload_df["text"].tolist())
            vectors = np.vstack([base["vectors"], upload_vectors])
        else:
            vectors = None

    store = vector_store.HybridStore(df, vectors)
    retriever = retrieval.Retriever(store, embedder=embedder, use_reranker=use_reranker)

    st.session_state["retriever"] = retriever
    st.session_state["retriever_sig"] = signature
    return retriever


# ═════════════════════════════════════════════════════════════════════════════
# Shared renderers
# ═════════════════════════════════════════════════════════════════════════════
def render_interactions(res: dict) -> str:
    ints = res["interactions"]
    lines = []
    if ints:
        counts = ", ".join(f"{res['summary'][l]} {l}"
                           for l in ("Major", "Moderate", "Minor", "Unknown")
                           if res["summary"][l])
        lines.append(f"**{len(ints)} documented interaction(s)** ({counts}):\n")
        for r in ints:
            lines.append(f"- {r['emoji']} **{r['level']}** — **{r['a']}** + **{r['b']}**")
    elif not res["unresolved"]:
        lines.append(f"✅ **No documented interactions** among these drugs "
                     f"({res['safe_pairs']} pair(s) checked in DDInter).")
    if res["unresolved"]:
        lines.append(f"\n⚠️ Not in the interaction database (skipped): "
                     f"*{', '.join(res['unresolved'])}*.")
    lines.append(f"\n<small>Source: {DDINTER_CITATION}</small>")
    return "\n".join(lines)


def render_sources(chunks) -> None:
    """The citation list. Source numbering matches [Source N] in the answer.

    # WHY show this by default rather than behind evaluation mode?
    # "The answer cites its sources" is only meaningful if the user can READ those
    # sources. This expander is what turns a citation from a claim into something
    # verifiable in one click.
    """
    if not chunks:
        return
    with st.expander(f"📚 Sources used ({len(chunks)})"):
        for rank, chunk in enumerate(chunks, 1):
            st.markdown(f"**[Source {rank}]** `{chunk.title}` · *{chunk.field}* "
                        f"· score={chunk.score:.3f}"
                        + (f" · _{chunk.origin}_" if chunk.origin != "drug-corpus" else ""))
            st.caption(chunk.text)


def render_evaluation(result) -> None:
    """Evaluation mode — expose every intermediate the pipeline produced.

    # WHY does a product ship a debug panel?
    # Because the project's central claim is "the answer comes from the retrieved
    # context". This panel is the evidence: the exact chunks, their component
    # scores, the verbatim prompt, and the mechanical verification of the answer.
    # Without it, groundedness is a promise; with it, it is auditable.
    """
    st.divider()
    st.markdown("### 🔬 Evaluation mode")

    verification = result.verification
    if verification:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sources cited", f"{len(verification.citations)}/{verification.n_sources}")
        c2.metric("Context overlap", f"{verification.overlap:.0%}")
        c3.metric("Disclaimer", "yes" if verification.has_disclaimer else "NO")
        c4.metric("Grounded", "yes" if verification.grounded else "no")
        if verification.invalid_citations:
            st.error(f"⚠️ The answer cited sources that do not exist: "
                     f"{verification.invalid_citations}. That is a fabricated citation.")
        elif result.mode == "llm" and not verification.citations:
            st.warning("The answer cited no sources. For a refusal that is correct; "
                       "for a factual claim it means the answer is not traceable.")

    st.markdown(f"**Generation:** `{result.mode}`"
                + (f" · provider `{result.provider}` · model `{result.model}`"
                   if result.provider else ""))
    if result.attempts:
        st.caption("Provider cascade: " +
                   " → ".join(f"{p} ({o})" for p, o in result.attempts))

    st.markdown("**Retrieved chunks (what the LLM was allowed to see):**")
    if not result.chunks:
        st.info("No chunks retrieved — the system abstained instead of guessing.")
    for rank, chunk in enumerate(result.chunks, 1):
        st.markdown(f"`[Source {rank}]` **{chunk.title}** · *{chunk.field}* · "
                    f"final={chunk.score:.3f} · bm25={chunk.bm25:.2f} · dense={chunk.dense:.3f}")
        st.code(chunk.text, language=None)

    if result.prompt:
        with st.expander("🧾 The exact prompt sent to the LLM"):
            st.code(result.prompt, language=None)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar — tool, settings, status
# ═════════════════════════════════════════════════════════════════════════════
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("💊 PhARMA RAG")
    st.caption("Grounded drug Q&A over an offline corpus + your own documents. "
               "Every answer is retrieved, cited and verifiable.")

    mode = st.radio("Tool", ["💬 Ask a question", "⚠️ Interaction checker",
                             "📄 Upload documents"], index=0)

    st.subheader("Retrieval")
    use_embeddings = st.toggle(
        "Semantic + hybrid retrieval", value=True,
        help="Sentence embeddings blended with BM25. Off = fast BM25 keyword search only.")
    use_reranker = st.toggle(
        "Cross-encoder reranker", value=False,
        help="Highest quality, slower. Re-scores the top 40 candidates jointly with the query.")
    top_k = st.slider("Sources per answer (k)", 3, 10, 5)
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

    retriever = get_retriever(use_embeddings, use_reranker)
    idb = load_interactions()

    use_rxnorm = st.toggle(
        "RxNorm name normalization", value=False,
        help="Resolve brand names/typos online (Tylenol → acetaminophen). Needs internet.")
    if use_rxnorm and not isinstance(idb, Exception):
        idb.rxnorm = get_rxnorm()

    st.subheader("Status")
    n_uploaded = sum(len(v) for v in st.session_state.uploaded_docs.values())
    st.markdown(f"- **Chunks indexed:** {len(retriever.df):,}")
    st.markdown(f"- **Sources:** {retriever.df['drug'].nunique():,}")
    if n_uploaded:
        st.markdown(f"- **Your documents:** {len(st.session_state.uploaded_docs)} "
                    f"file(s), {n_uploaded} section(s)")
    dense_backend = (retriever.store.dense.backend if retriever.store.has_dense
                     else "off (BM25 only)")
    st.markdown(f"- **Vector store:** `{dense_backend}`")
    st.markdown(f"- **Lexical:** `{retriever.store.lexical.backend}`")
    if retriever.reranker is not None:
        st.markdown("- **Reranker:** 🟢 cross-encoder")

    if isinstance(idb, Exception):
        st.markdown("- **Interactions:** 🔴 not loaded")
    else:
        st.markdown(f"- **Interactions:** 🟢 {len(idb.pairs):,} pairs · {len(idb._canon):,} drugs")

    if status["active"]:
        st.markdown(f"- **LLM:** 🟢 {status['active']} · `{status['model']}`")
    else:
        st.markdown("- **LLM:** 🔴 none → extractive answers")
        st.info("No LLM key found. The app still works and still cites sources — it "
                "returns the retrieved text directly. For generated answers add "
                "`OPENROUTER_API_KEY`, `GEMINI_API_KEY` or `GROQ_API_KEY` to Streamlit "
                "secrets (or a local `.env`).", icon="ℹ️")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("⚠️ Educational demo. Not medical advice.")


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 1 — Upload documents
# ═════════════════════════════════════════════════════════════════════════════
if mode == "📄 Upload documents":
    st.title("📄 Upload your own documents")
    st.caption("Uploaded files run through the identical pipeline as the built-in "
               "corpus: chunked at 90 words with 25-word overlap, embedded, indexed, "
               "and cited by filename and section.")

    uploaded = st.file_uploader(
        "Add documents",
        type=documents_stage.SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help=f"Supported: {', '.join(documents_stage.SUPPORTED_UPLOAD_TYPES)}")

    if uploaded:
        added, failed = [], []
        for file in uploaded:
            if file.name in st.session_state.uploaded_docs:
                continue
            try:
                docs = documents_stage.documents_from_upload(file.name, file.getvalue())
                st.session_state.uploaded_docs[file.name] = docs
                added.append((file.name, len(docs)))
            except Exception as exc:
                failed.append((file.name, str(exc)))
        for name, n in added:
            st.success(f"✅ **{name}** — {n} section(s) ingested.")
        for name, err in failed:
            st.error(f"❌ **{name}** — {err}")
        if added:
            st.rerun()

    if st.session_state.uploaded_docs:
        st.subheader("Indexed documents")
        for name, docs in list(st.session_state.uploaded_docs.items()):
            words = sum(d.word_count for d in docs)
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{name}** — {len(docs)} section(s), {words:,} words")
                with st.expander("Preview sections"):
                    for d in docs[:12]:
                        st.markdown(f"*{d.field}* — {d.text[:220]}…")
            with col2:
                if st.button("Remove", key=f"rm_{name}"):
                    del st.session_state.uploaded_docs[name]
                    st.rerun()

        st.info("Switch to **💬 Ask a question** to query these documents alongside "
                "the drug corpus. Answers will cite the filename and section.", icon="💡")
    else:
        st.info("No documents uploaded yet. The app still works on the built-in "
                f"corpus of {retriever.df['drug'].nunique():,} FDA drug labels.", icon="ℹ️")

    st.divider()
    st.caption(DISCLAIMER)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 2 — Interaction checker (carried over from app.py)
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "⚠️ Interaction checker":
    st.title("⚠️ Drug interaction checker")
    st.caption("Enter the drugs on a prescription — one per line or comma-separated. "
               "Every pair is checked against DDInter 2.0. This is a deterministic "
               "lookup, not retrieval: it cannot miss a pair or invent one.")

    if isinstance(idb, Exception):
        st.error(f"Interaction database not available: {idb}\n\nDownload the 8 DDInter "
                 "CSVs into `data/ddinter/` (see README).")
    else:
        default = st.session_state.get("rx_text", "warfarin\nnaproxen\nmetformin\nomeprazole")
        text = st.text_area("Drugs / prescription", value=default, height=140,
                            placeholder="warfarin\nnaproxen\nmetformin")
        if st.button("🔎 Check interactions", type="primary"):
            names = [t for t in re.split(r"[\n,;]+", text) if t.strip()]
            if len(names) < 2:
                st.warning("Enter at least two drugs to check for interactions.")
            else:
                res = idb.check_list(names)
                summary = res["summary"]
                if summary["Major"]:
                    st.error(f"🔴 {summary['Major']} MAJOR interaction(s) found — "
                             f"review before dispensing.")
                elif summary["Moderate"]:
                    st.warning(f"🟠 {summary['Moderate']} moderate interaction(s) found.")
                elif res["interactions"]:
                    st.info("🟡 Only minor / unclassified interactions found.")
                elif not res["unresolved"]:
                    st.success("✅ No documented interactions among these drugs.")

                st.markdown(render_interactions(res), unsafe_allow_html=True)

                # Explain the top interaction with RAG narrative from the labels.
                top = res["interactions"][0] if res["interactions"] else None
                if top:
                    with st.expander(f"📖 Label context for {top['a']} + {top['b']}"):
                        result = prompting.answer_question(
                            f"{top['a']} interaction with {top['b']}",
                            retriever, k=top_k, use_llm=False)
                        st.markdown(result.text)
        st.divider()
        st.caption(DISCLAIMER)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 3 — Ask a question (the RAG chat)
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.title("Ask a question")
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
        st.markdown("**Try one:**")
        examples = ["How does metformin work?",
                    "Does warfarin interact with naproxen?",
                    "Side effects of aspirin?"]   # aspirin absent → honest refusal
        for col, example in zip(st.columns(3), examples):
            if col.button(example, use_container_width=True):
                picked = example

    prompt_text = st.chat_input("Ask about a drug or your documents…") or picked

    if prompt_text:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            # Structured interaction block first, when the question names ≥2 drugs.
            interaction_md = ""
            if not isinstance(idb, Exception) and interaction_intent(prompt_text):
                matched = sorted(retriever.ground(prompt_text)["matched"])
                if len(matched) >= 2:
                    res = idb.check_list(matched)
                    interaction_md = ("#### ⚠️ Interaction check\n"
                                      + render_interactions(res) + "\n\n---\n")
                    st.markdown(interaction_md, unsafe_allow_html=True)

            with st.spinner("Retrieving sources and composing an answer…"):
                result = prompting.answer_question(
                    prompt_text, retriever, k=top_k, use_llm=True, prefer=prefer)

            st.markdown(result.text, unsafe_allow_html=True)
            render_sources(result.chunks)

            if result.mode == "extractive" and not result.refused:
                st.caption("↳ retrieval-only answer (no LLM reachable)")

            if evaluation_mode:
                render_evaluation(result)

        st.session_state.messages.append({
            "role": "assistant",
            "content": interaction_md + result.text,
            "chunks": result.chunks,
        })

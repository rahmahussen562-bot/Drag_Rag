"""
ui/state.py — cached resource construction and session bootstrap.

# WHY does this exist as its own module?
# `streamlit_app.py` is now only page config + session bootstrap + router. The
# expensive, cached objects it used to build in-file (the chunk table, the
# embedding matrix, the retriever, the interaction DB) are shared by several
# pages, so they belong somewhere every page can import from — not in the router.
#
# This module holds NO pipeline logic. Every RAG decision lives in the numbered
# stage modules and is testable from the command line without Streamlit. What is
# here is purely *lifecycle*: what gets built once per container, what gets built
# once per session, and what triggers a rebuild.
"""
from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stages import load  # noqa: E402

documents_stage = load("01_documents")
chunking = load("03_chunking")
representation = load("04_vector_representation")
vector_store = load("05_create_vector_store")
retrieval = load("06_retrieve_context")
prompting = load("07_prompting")

from interactions import InteractionDB  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Cached construction
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
    # `records` and `docs` are intermediates only. Holding them in the returned
    # dict would pin ~23 MB of raw label text in the cross-session cache for the
    # life of the container, and nothing downstream reads them — the chunk table
    # carries every field the UI and retriever need.
    docs = documents_stage.documents_from_drug_records(
        documents_stage.load_drug_records())
    df = chunking.build_chunks(docs)
    del docs

    embedder = representation.Embedder() if use_embeddings else None
    vectors, info = representation.embed_corpus(
        df["text"].tolist(), embedder=embedder, use_dense=use_embeddings)

    # Chunking allocates millions of short-lived token strings during stage-02
    # cleaning. Returning those arenas before the app starts serving keeps the
    # steady-state footprint measurably below the container limit.
    gc.collect()
    return {"df": df, "vectors": vectors, "embedder": embedder, "info": info}


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
    retriever_obj = retrieval.Retriever(store, embedder=embedder,
                                        use_reranker=use_reranker)

    st.session_state["retriever"] = retriever_obj
    st.session_state["retriever_sig"] = signature
    return retriever_obj


def bootstrap() -> None:
    """Initialise the session keys every page assumes exist."""
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []


# ═════════════════════════════════════════════════════════════════════════════
# The context handed to every page
#
# # WHY pass a context object instead of letting pages read st.session_state?
# A page that reaches into global session state is untestable and can silently
# depend on a key some other page happens to set. An explicit context makes each
# page's inputs visible in its signature — which is also what will let the two
# agents in Phase 2 be swapped per page without touching page code.
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class AppContext:
    """Everything a page needs, resolved once per rerun by the sidebar."""
    retriever: object                 # 06_retrieve_context.Retriever
    idb: object                       # InteractionDB, or the Exception that stopped it
    top_k: int
    prefer: Optional[str]             # pinned LLM provider, or None for the cascade
    evaluation_mode: bool
    use_embeddings: bool
    use_reranker: bool
    agent: object = None              # the active persona (agents.Agent)

    @property
    def interactions_available(self) -> bool:
        return not isinstance(self.idb, Exception)

    @property
    def persona(self):
        """The active persona's AgentConfig."""
        return self.agent.config if self.agent is not None else None

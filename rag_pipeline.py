"""
rag_pipeline.py — compatibility facade over the numbered pipeline stages.

# WHY does this file still exist after the refactor?
# It used to BE the pipeline: a single 600-line module holding loading, cleaning,
# chunking, retrieval and generation. That logic now lives in the seven numbered
# stage files, where each step is documented and independently runnable.
#
# But `eval/run_eval.py` imports `RAGEngine` from here, and the project rule is to
# refactor rather than rewrite. So this module keeps its public API exactly as it
# was and DELEGATES every call to the stages. There is no duplicated logic below —
# only adaptation:
#
#     RAGEngine.__init__  →  01_documents + 03_chunking + 05_vector_store
#     RAGEngine.retrieve  →  06_retrieve_context
#     RAGEngine.answer    →  07_prompting
#     llm_status          →  llm_providers
#
# New code should import the stages directly (see streamlit_app.py). This file is
# the bridge that keeps the existing evaluation harness running unchanged.

Public API preserved:
    RAGEngine, llm_status, preprocess_text, build_chunks, sliding_windows,
    BM25_SOURCE, SYSTEM_PROMPT, DISCLAIMER, _drug_identity_tokens
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import load  # noqa: E402

import llm_providers  # noqa: E402

_documents = load("01_documents")
_preprocessing = load("02_preprocessing")
_chunking = load("03_chunking")
_representation = load("04_vector_representation")
_store = load("05_create_vector_store")
_retrieval = load("06_retrieve_context")
_prompting = load("07_prompting")

# ── Re-exported names, so `from rag_pipeline import X` keeps working ─────────
preprocess_text = _preprocessing.preprocess_text
build_chunks = _chunking.build_chunks
sliding_windows = _chunking.sliding_windows
SYSTEM_PROMPT = _prompting.SYSTEM_PROMPT
DISCLAIMER = _prompting.DISCLAIMER
EMBED_MODEL_NAME = _representation.EMBED_MODEL_NAME
CHUNK_WORDS = _chunking.CHUNK_WORDS
CHUNK_OVERLAP = _chunking.CHUNK_OVERLAP
ALPHA = _retrieval.ALPHA
_drug_identity_tokens = _retrieval.identity_tokens

# Reported in the UI sidebar; resolved once the lexical store is first built.
BM25_SOURCE = "rank_bm25"
try:
    import rank_bm25  # noqa: F401
except Exception:
    BM25_SOURCE = "numpy-fallback"

# Retrieval statuses (stage 06) → the legacy `meta["mode"]` strings that
# eval/run_eval.py matches on.
_STATUS_TO_MODE = {
    "unknown_entity": "refused",
    "no_entity": "refused",
    "no_relevant_context": "no-relevant-context",
}


def llm_status() -> dict:
    """Legacy shape: {"backend": <provider or None>, "model": <slug or None>}."""
    status = llm_providers.llm_status()
    return {"backend": status["active"], "model": status["model"],
            "available": status["available"], "all": status["all"]}


def active_llm_backend() -> Optional[str]:
    return llm_providers.llm_status()["active"]


class RAGEngine:
    """The original engine interface, now composed from the numbered stages.

    Parameters are unchanged from the pre-refactor version so existing callers
    (eval/run_eval.py) need no edits.
    """

    def __init__(self, data_path: Optional[str] = None, use_embeddings: bool = True,
                 use_reranker: bool = False, cache_embeddings: bool = True):
        # STAGE 1 + 3 — load raw records, expand to documents, chunk them.
        self.drugs_data = _documents.load_drug_records(data_path)
        docs = _documents.documents_from_drug_records(self.drugs_data)
        self.df = _chunking.build_chunks(docs)

        # STAGE 4 + 5 + 6 — represent, index, and build the grounded retriever.
        embedder = _representation.Embedder() if use_embeddings else None
        store = _store.build_store(self.df, embedder=embedder, use_cache=cache_embeddings,
                                   use_dense=use_embeddings)
        self.retriever = _retrieval.Retriever(
            store, embedder=embedder, use_reranker=use_reranker)

        self.store = store
        self.embed_model = embedder.model if (embedder and embedder.available) else None
        self.reranker = self.retriever.reranker
        self.drug_vocab = self.retriever.entity_vocab

        self.retriever_name = (
            "hybrid+rerank" if self.retriever.reranker is not None else
            "hybrid" if store.has_dense else "bm25"
        )

    # ---- grounding ---------------------------------------------------------
    def ground_query(self, query: str) -> dict:
        return self.retriever.ground(query)

    # ---- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, k: int = 5,
                 restrict_tokens: Optional[set] = None) -> list[tuple[int, float]]:
        """Legacy return shape: [(chunk_index, score), …]."""
        chunks = self.retriever.retrieve(query, k=k, restrict_tokens=restrict_tokens)
        return [(c.index, c.score) for c in chunks]

    # ---- context + prompt --------------------------------------------------
    def pack_context(self, results) -> str:
        chunks = [self._chunk_from_pair(idx, score) for idx, score in results]
        return _prompting.pack_context(chunks)

    def build_prompt(self, query: str, context: str) -> str:
        return _prompting.build_prompt(query, context)

    def _chunk_from_pair(self, idx: int, score: float):
        row = self.df.loc[idx]
        return _retrieval.RetrievedChunk(
            index=int(idx), score=float(score), text=row["text"], drug=row["drug"],
            title=row["title"], field=row["field"], origin=row["origin"])

    # ---- generation --------------------------------------------------------
    def answer(self, query: str, k: int = 5, use_llm: bool = True):
        """Legacy return shape: (answer_text, [(idx, score), …], meta_dict)."""
        result = _prompting.answer_question(query, self.retriever, k=k, use_llm=use_llm)
        results = [(c.index, c.score) for c in result.chunks]

        if result.mode == "refused":
            mode = _STATUS_TO_MODE.get(result.status, "refused")
            meta = {"mode": mode, "unknown": []}
        elif result.mode == "llm":
            meta = {"mode": result.provider, "provider": result.provider,
                    "model": result.model}
        elif result.mode == "llm_failed":
            # A configured provider failed. Carry the error through instead of
            # flattening it into "extractive", so a CLI/eval caller can tell a
            # real outage apart from the intentional no-credentials mode.
            meta = {"mode": "llm_failed", "attempts": result.attempts,
                    "llm_error": result.llm_error}
        else:
            meta = {"mode": "extractive", "attempts": result.attempts}

        if result.verification is not None:
            meta["verification"] = result.verification
        meta["answer"] = result
        return result.text, results, meta


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test:  python rag_pipeline.py "what are the side effects of aspirin?"
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    query = " ".join(sys.argv[1:]) or "what is metformin used for?"
    engine = RAGEngine(use_embeddings=False)   # fast path for a quick check
    print(f"Loaded {len(engine.drugs_data)} drugs, {len(engine.df)} chunks. "
          f"Retriever={engine.retriever_name}, BM25={BM25_SOURCE}\n")
    text, results, meta = engine.answer(query, use_llm=False)
    print("Q:", query, "\n")
    print(text)

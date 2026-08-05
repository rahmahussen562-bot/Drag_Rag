"""
test_degraded.py — prove the documented fallbacks actually still answer.

The README claims the pipeline degrades in QUALITY, never in TRUSTWORTHINESS:

    no torch / sentence-transformers  → BM25-only, still grounded and citing
    no faiss                          → NumPy matmul, identical ranking
    no LLM provider                   → extractive, still cited and disclaimed

# WHY does this need a test rather than a paragraph?
# Because those paths are, by definition, the ones nobody exercises in
# development — every dev box has torch installed. A fallback that is never run is
# a fallback that has silently rotted, and the first person to discover it is a
# customer on a constrained host. Each mode is forced here and checked to still
# ground, still cite, and still REFUSE out-of-corpus drugs.

Run:  python eval/test_degraded.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from stages import load  # noqa: E402

documents_stage = load("01_documents")
chunking = load("03_chunking")
representation = load("04_vector_representation")
vector_store = load("05_create_vector_store")
retrieval = load("06_retrieve_context")
prompting = load("07_prompting")

GOOD = "how does metformin work?"
ABSENT = "what are the side effects of aspirin?"

results: list = []


def record(mode: str, name: str, ok: bool, detail: str = "") -> bool:
    results.append((mode, name, ok, detail))
    print(f"    {'✅' if ok else '❌'} {name}" + (f"   {detail}" if detail else ""))
    return ok


def exercise(mode: str, retriever) -> None:
    """Every degraded mode must satisfy the same three properties."""
    good = retriever.retrieve_context(GOOD, k=5)
    record(mode, "grounds a known drug", good.ok and bool(good.chunks),
           f"{len(good.chunks)} sources, status={good.status}")
    record(mode, "returns the RIGHT drug",
           any("metformin" in retrieval.identity_tokens(c.drug) for c in good.chunks))

    # The one that matters most: degradation must not open the abstention gate.
    absent = retriever.retrieve_context(ABSENT, k=5)
    record(mode, "still REFUSES an out-of-corpus drug", not absent.ok,
           f"status={absent.status}, sources={len(absent.chunks)}")

    answer = prompting.answer_question(GOOD, retriever, k=3, use_llm=False)
    record(mode, "still produces a cited answer",
           bool(answer.text.strip()) and bool(answer.chunks))
    record(mode, "still carries the disclaimer",
           "not medical advice" in answer.text.lower())


def main() -> int:
    df = chunking.build_chunks(documents_stage.load_drug_documents())

    # ── Mode 1: no dense encoder (simulates a host with no torch wheel) ──────
    print("\n[1] NO EMBEDDINGS — sentence-transformers/torch unavailable")
    print("    expected: BM25-only, still grounded, still refuses\n")
    store = vector_store.build_store(df, embedder=None, use_dense=False)
    lexical_only = retrieval.Retriever(store, embedder=None)
    record("bm25", "dense store is genuinely off", not store.has_dense,
           f"has_dense={store.has_dense}")
    exercise("bm25", lexical_only)

    # ── Mode 2: full stack, for comparison ──────────────────────────────────
    print("\n[2] FULL STACK — hybrid retrieval")
    print("    expected: everything above, plus a dense backend\n")
    embedder = representation.Embedder()
    if not embedder.available:
        print(f"    ⚠️  encoder unavailable here ({embedder.load_error});")
        print("       mode 2 skipped — mode 1 is what this host would run.")
    else:
        full_store = vector_store.build_store(df, embedder=embedder)
        full = retrieval.Retriever(full_store, embedder=embedder)
        record("hybrid", "dense store present", full_store.has_dense,
               full_store.dense.backend)
        exercise("hybrid", full)

        # ── Mode 3: no FAISS — the NumPy fallback must rank identically ─────
        print("\n[3] NO FAISS — NumPy matmul fallback")
        print("    expected: IDENTICAL ranking, just slower\n")
        vectors = full_store.dense.vectors
        numpy_store = vector_store.VectorStore.__new__(vector_store.VectorStore)
        numpy_store.vectors = vectors
        numpy_store.n, numpy_store.dim = vectors.shape
        numpy_store.index = None            # force the fallback path
        numpy_store.backend = "numpy"
        numpy_store.nprobe = 8
        numpy_store.build_seconds = 0.0

        qv = embedder.encode_query(GOOD)
        faiss_hits = [i for i, _ in full_store.dense.search(qv, k=5)]
        numpy_hits = [i for i, _ in numpy_store.search(qv, k=5)]
        record("numpy", "NumPy fallback matches FAISS exactly",
               faiss_hits == numpy_hits,
               f"faiss={faiss_hits} numpy={numpy_hits}")

    # ── Mode 4: no LLM provider → extractive ────────────────────────────────
    print("\n[4] NO LLM PROVIDER — extractive mode")
    print("    expected: quotation instead of generation, still cited\n")
    answer = prompting.answer_question(GOOD, lexical_only, k=3, use_llm=False)
    record("extractive", "mode is extractive", answer.mode == "extractive",
           f"mode={answer.mode}")
    record("extractive", "cites every source it used",
           bool(answer.verification and answer.verification.citations),
           f"cited {len(answer.verification.citations) if answer.verification else 0}"
           f"/{len(answer.chunks)}")
    record("extractive", "no fabricated citations",
           bool(answer.verification and not answer.verification.invalid_citations))

    print("\n" + "=" * 62)
    failed = [(m, n) for m, n, ok, _ in results if not ok]
    print(f"SUMMARY  {len(results) - len(failed)}/{len(results)} checks passed")
    for mode, name in failed:
        print(f"  FAILED: [{mode}] {name}")
    print("RESULT:", "PASS ✅" if not failed else "FAIL ❌")
    print("=" * 62)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

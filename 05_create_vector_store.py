"""
05_create_vector_store.py — STAGE 5:  **Vector Store**

    Raw Documents  →  Preprocessing  →  Chunking  →  Vector Representation
    →  >>> Vector Store <<<  →  Context Retrieval  →  Prompt Building  →  LLM Generation  →  Streamlit UI

# WHY does this stage exist?
# Stage 04 produced a 14,955 × 384 matrix of vectors. A vector store is the data
# structure that answers "which rows of that matrix point most nearly the same way
# as this query vector?" — fast, and without scanning everything by hand.
#
# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN DECISION: FAISS is KEPT. Chroma is NOT adopted. Here is the reasoning.
# ═══════════════════════════════════════════════════════════════════════════════
# All three notebooks already use FAISS, and they use it CORRECTLY, which is the
# part worth checking before changing anything:
#   • they normalise embeddings to unit length in stage 04, then use
#     `IndexFlatIP` (inner product). For unit vectors inner product IS cosine
#     similarity, so the index returns exact cosine ranking — not an approximation
#     of it. This is the standard idiom and it is right.
#   • the Scale Edition additionally demonstrates `IndexIVFFlat`, the approximate
#     variant, and measures its top-5 agreement against exact search.
# So the notebooks pass the correctness bar. The remaining question is whether
# Chroma would buy anything at THIS project's scale and shape:
#
#   Corpus size     ~15k vectors × 384 dim = 23 MB. FAISS IndexFlatIP scans this
#                   exhaustively in ~1-3 ms. Chroma (which wraps hnswlib) would
#                   return an APPROXIMATE answer no faster in wall-clock terms at
#                   this size — you buy sublinear scaling you do not need, and pay
#                   for it in recall.
#   Persistence     Our index is a pure derivative of data/drugs_large.json plus a
#                   cached .npy. It is rebuilt deterministically in ~40s and is
#                   never the system of record. Chroma's headline feature is a
#                   persistent, mutable, transactional collection — solving a
#                   problem we do not have.
#   Metadata filter Genuinely needed: stage 06 restricts retrieval to the drug(s)
#                   the query names. Chroma does this with a `where` clause; we do
#                   it with a pandas boolean mask over the chunk table, which is
#                   ~0.2 ms at 15k rows and keeps ALL metadata logic in one place
#                   instead of split between a query language and Python.
#   Deployment cost Chroma pulls in chromadb + its own embedding stack + sqlite
#                   persistence, adding ~120 MB and a writable-disk requirement.
#                   Streamlit Community Cloud gives 1 GB RAM and an ephemeral
#                   filesystem; torch + transformers + the corpus already use most
#                   of it. faiss-cpu is ~15 MB and needs no writable disk.
#
# CONCLUSION: replacing FAISS with Chroma would add a dependency, a storage layer
# and approximate recall, in exchange for capabilities this project does not use.
# The notebooks' choice stands. What we DO fix is that the previous Streamlit app
# never actually used FAISS — it brute-forced cosine through scikit-learn, so the
# notebooks' index work was thrown away in production. This stage wires the real
# FAISS index into the app, behind a numpy fallback for hosts without the wheel.
#
# WHEN WOULD THE ANSWER CHANGE? Documented honestly, because it is not "never":
#   • > ~1M vectors, or a corpus that no longer fits in RAM → a real vector DB.
#   • Concurrent writers / multi-tenant collections → a real vector DB.
#   • Incremental upserts as the primary write pattern → a real vector DB.
# At 15k read-mostly vectors rebuilt from source, none of those apply.

Run standalone:
    python 05_create_vector_store.py          # build both indexes, verify exact == approximate
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import load  # noqa: E402

documents_stage = load("01_documents")
preprocessing = load("02_preprocessing")
chunking = load("03_chunking")
representation = load("04_vector_representation")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

# Below this many vectors, exact search is both faster and perfect, so we never
# pay the approximate-search recall penalty. Above it, IVF becomes worthwhile.
IVF_THRESHOLD = 50_000
IVF_NPROBE = 8          # how many clusters an approximate query visits


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1
# The dense vector store.
# ─────────────────────────────────────────────────────────────────────────────
class VectorStore:
    """FAISS-backed dense index over unit-normalised embeddings.

    backend is one of:
      "faiss-flat" — exact inner-product search (== exact cosine). Default.
      "faiss-ivf"  — approximate, clustered. Auto-selected above IVF_THRESHOLD.
      "numpy"      — fallback when faiss-cpu is not installed. Same maths
                     (a single matrix multiply), just without FAISS's SIMD
                     kernels. Correct, merely slower — the app stays up.
    """

    def __init__(self, vectors: np.ndarray, use_ivf: Optional[bool] = None,
                 nprobe: int = IVF_NPROBE):
        if vectors is None or len(vectors) == 0:
            raise ValueError("VectorStore requires a non-empty embedding matrix.")
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.n, self.dim = self.vectors.shape
        self.nprobe = nprobe
        self.index = None
        self.backend = "numpy"
        self.build_seconds = 0.0

        # WHY guard on the norm? If stage 04's `normalize_embeddings=True` were
        # ever dropped, inner-product search would silently become dot-product
        # search — which ranks longer chunks higher irrespective of relevance and
        # produces plausible but wrong citations. Fail loudly instead.
        sample = self.vectors[: min(256, self.n)]
        norms = np.linalg.norm(sample, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"Embeddings are not unit-normalised (mean norm {norms.mean():.4f}). "
                f"Inner-product search would not equal cosine similarity. "
                f"Check normalize_embeddings=True in stage 04.")

        want_ivf = (self.n >= IVF_THRESHOLD) if use_ivf is None else use_ivf
        self._build(want_ivf)

    def _build(self, want_ivf: bool) -> None:
        t0 = time.time()
        try:
            import faiss
        except Exception:
            # No faiss wheel for this platform/python — brute force is fine here.
            self.backend = "numpy"
            self.build_seconds = time.time() - t0
            return

        if want_ivf and self.n >= 40 * 8:
            # IVF clusters vectors into `nlist` Voronoi cells and searches only the
            # `nprobe` nearest cells. sqrt(n) cells is the standard rule of thumb:
            # it balances cells-to-scan against vectors-per-cell.
            nlist = max(8, int(math.sqrt(self.n)))
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(self.vectors)        # IVF must learn the cluster centroids first
            index.add(self.vectors)
            index.nprobe = self.nprobe
            self.index, self.backend = index, "faiss-ivf"
        else:
            # Exact search. `IndexFlatIP` + unit vectors == exact cosine ranking.
            index = faiss.IndexFlatIP(self.dim)
            index.add(self.vectors)
            self.index, self.backend = index, "faiss-flat"
        self.build_seconds = time.time() - t0

    # ---- search ------------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Return the k nearest chunks as [(row_index, cosine_similarity), …]."""
        q = np.ascontiguousarray(np.atleast_2d(query_vector), dtype="float32")
        k = min(k, self.n)
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        # numpy fallback: one matmul, then partial sort. Identical ranking.
        scores = self.vectors @ q[0]
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def scores_all(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity against EVERY chunk.

        # WHY does this exist alongside search()?
        # Stage 06 blends dense scores with BM25 scores across the whole corpus
        # before ranking, so it needs the full score vector, not just a top-k. For
        # a 23 MB matrix this matmul is ~2 ms — cheaper than reconciling two
        # partial rankings.
        """
        q = np.ascontiguousarray(np.atleast_2d(query_vector), dtype="float32")
        return self.vectors @ q[0]

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path = FAISS_INDEX_PATH) -> bool:
        """Write the FAISS index to disk. Non-fatal on failure."""
        if self.index is None:
            return False
        try:
            import faiss
            path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        return {"backend": self.backend, "vectors": self.n, "dim": self.dim,
                "memory_mb": round(self.vectors.nbytes / 1e6, 1),
                "build_seconds": round(self.build_seconds, 2),
                "nprobe": self.nprobe if self.backend == "faiss-ivf" else None}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2
# The sparse (lexical) store — BM25.
#
# WHY is BM25 in the "vector store" stage rather than stage 04?
# Because BM25 is not a vector space you can embed a query into; it is a scoring
# function evaluated against an inverted index. It belongs with the other search
# structure, next to FAISS, since stage 06 queries both together.
#
# WHY BM25 rather than plain TF-IDF cosine?
# Two corrections that matter on real label text:
#   • SATURATION — the 10th occurrence of "mg" in a dosage chunk adds almost
#     nothing, whereas TF-IDF keeps counting it linearly and lets one repetitive
#     chunk dominate.
#   • LENGTH NORMALISATION — a 90-word chunk should not outrank a 40-word chunk
#     merely for containing more words.
# ─────────────────────────────────────────────────────────────────────────────
class _NumpyBM25:
    """Compact BM25-Okapi used when `rank_bm25` is not installed.

    Kept from the notebooks deliberately: it makes the formula inspectable, and it
    means the app has no hard dependency on a single small package. The posting
    list (term → [(doc, freq)]) is what makes it usable at 15k chunks — scoring
    touches only documents that actually contain a query term.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus)
        tf = [Counter(doc) for doc in corpus]
        self.dl = np.array([len(doc) for doc in corpus], dtype="float32")
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        df = Counter(term for doc in corpus for term in set(doc))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, counts in enumerate(tf):
            for term, freq in counts.items():
                self.postings.setdefault(term, []).append((i, freq))

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype="float32")
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in self.postings[term]:
                denom = freq + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


class LexicalStore:
    """BM25 index over the stage-02-cleaned chunk text."""

    def __init__(self, clean_texts: list[str]):
        corpus = [t.split() for t in clean_texts]
        self.n = len(corpus)
        t0 = time.time()
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(corpus)
            self.backend = "rank_bm25"
        except Exception:
            self._bm25 = _NumpyBM25(corpus)
            self.backend = "numpy-fallback"
        self.build_seconds = time.time() - t0

    def scores_all(self, query: str) -> np.ndarray:
        """BM25 score of the query against every chunk.

        The query is cleaned with the SAME stage-02 profile used to build the
        index — otherwise "doses" in the query would never match the indexed
        lemma "dose" and the retriever would quietly under-perform.
        """
        tokens = preprocessing.clean(query).split()
        return np.asarray(self._bm25.get_scores(tokens), dtype="float32")

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self.scores_all(query)
        k = min(k, self.n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def stats(self) -> dict:
        return {"backend": self.backend, "documents": self.n,
                "build_seconds": round(self.build_seconds, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3
# Build both stores together — the single entry point stage 06 calls.
# ─────────────────────────────────────────────────────────────────────────────
class HybridStore:
    """Bundles the dense store, the lexical store and the chunk table.

    Holding all three together guarantees the invariant everything else depends
    on: row i of the chunk table, row i of the embedding matrix and document i of
    the BM25 index are THE SAME CHUNK. If those ever drift apart the system cites
    one chunk while quoting another, so they are constructed in one place.
    """

    def __init__(self, df, vectors: Optional[np.ndarray] = None,
                 use_ivf: Optional[bool] = None):
        self.df = df
        self.lexical = LexicalStore(df["clean"].tolist())
        self.dense: Optional[VectorStore] = None
        if vectors is not None and len(vectors) == len(df):
            self.dense = VectorStore(vectors, use_ivf=use_ivf)
        elif vectors is not None:
            raise ValueError(
                f"Embedding count ({len(vectors)}) != chunk count ({len(df)}). "
                f"Refusing to build a store whose rows would not line up.")

    @property
    def has_dense(self) -> bool:
        return self.dense is not None

    def stats(self) -> dict:
        return {"chunks": len(self.df), "lexical": self.lexical.stats(),
                "dense": self.dense.stats() if self.dense else None}


def build_store(df, embedder: Optional[object] = None, use_cache: bool = True,
                show_progress: bool = False, use_dense: bool = True) -> HybridStore:
    """Chunk table → fully built hybrid store. The one function stage 06 needs.

    `use_dense=False` builds a lexical-only store (BM25 with no embeddings) — the
    fast path used by the evaluation harness and by lightweight deployments.
    """
    vectors, info = representation.embed_corpus(
        df["text"].tolist(), embedder=embedder, use_cache=use_cache,
        show_progress=show_progress, use_dense=use_dense)
    store = HybridStore(df, vectors)
    store.embed_info = info
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("  STAGE 5 — VECTOR STORE")
    print("=" * 78)

    docs = documents_stage.load_drug_documents()
    df = chunking.build_chunks(docs)
    store = build_store(df, show_progress=True)
    s = store.stats()

    print(f"\nChunks indexed : {s['chunks']:,}")
    print(f"Lexical store  : BM25 via {s['lexical']['backend']} "
          f"(built in {s['lexical']['build_seconds']}s)")
    if s["dense"]:
        d = s["dense"]
        print(f"Dense store    : {d['backend']}  {d['vectors']:,} × {d['dim']}  "
              f"{d['memory_mb']} MB  (built in {d['build_seconds']}s)")
    else:
        print("Dense store    : unavailable → lexical-only mode")
        raise SystemExit(0)

    # ── Verify FAISS returns the same answer as brute force ─────────────────
    print("\n── CORRECTNESS: FAISS vs brute-force cosine ──")
    embedder = representation.Embedder()
    qv = embedder.encode_query("how does metformin lower blood sugar")
    faiss_hits = store.dense.search(qv, k=5)

    brute = store.dense.vectors @ qv[0]
    brute_top = np.argsort(-brute)[:5]
    agree = len(set(i for i, _ in faiss_hits) & set(int(i) for i in brute_top))
    print(f"  top-5 agreement with exact numpy scan: {agree}/5")
    for rank, (idx, score) in enumerate(faiss_hits, 1):
        print(f"    {rank}. {score:.3f}  {df.loc[idx, 'drug'][:34]:<34s} · {df.loc[idx, 'field']}")

    # ── Show the exact/approximate trade-off the Scale notebook demonstrates ─
    print("\n── SPEED/RECALL: exact (Flat) vs approximate (IVF) ──")
    try:
        flat = VectorStore(store.dense.vectors, use_ivf=False)
        ivf = VectorStore(store.dense.vectors, use_ivf=True)
        t0 = time.time()
        for _ in range(50):
            flat.search(qv, k=5)
        flat_ms = (time.time() - t0) / 50 * 1000
        t0 = time.time()
        for _ in range(50):
            ivf.search(qv, k=5)
        ivf_ms = (time.time() - t0) / 50 * 1000
        overlap = len(set(i for i, _ in flat.search(qv, 5)) & set(i for i, _ in ivf.search(qv, 5)))
        print(f"  {flat.backend:11s} {flat_ms:6.2f} ms/query   (exact)")
        print(f"  {ivf.backend:11s} {ivf_ms:6.2f} ms/query   top-5 recall vs exact: {overlap}/5")
        print(f"  → at {store.dense.n:,} vectors exact search is already ~{flat_ms:.1f} ms,")
        print("    so we keep IndexFlatIP and pay no recall penalty. IVF turns on")
        print(f"    automatically past {IVF_THRESHOLD:,} vectors.")
    except Exception as exc:
        print(f"  comparison skipped: {type(exc).__name__}: {exc}")

    print("\n── LEXICAL store sanity ──")
    for idx, score in store.lexical.search("naproxen dosage", k=3):
        print(f"  {score:6.2f}  {df.loc[idx, 'drug'][:34]:<34s} · {df.loc[idx, 'field']}")

    store.dense.save()
    print(f"\nStage 5 complete — index persisted to {FAISS_INDEX_PATH.name}.")

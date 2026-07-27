"""
04_vector_representation.py — STAGE 4:  **Vector Representation**

    Raw Documents  →  Preprocessing  →  Chunking  →  >>> Vector Representation <<<
    →  Vector Store  →  Context Retrieval  →  Prompt Building  →  LLM Generation  →  Streamlit UI

# WHY does this stage exist?
# A computer cannot compare "how does metformin lower blood sugar?" to a paragraph
# of text. It can only compare NUMBERS. This stage turns every chunk into a vector
# so that "similar meaning" becomes "small angle between vectors".
#
# # WHY TWO representations instead of one?
# Because sparse and dense vectors fail in exactly opposite situations, and stage
# 06 blends them to cover both. This is the single highest-leverage design choice
# in the retrieval half of the project.
#
#   SPARSE (BM25 / TF-IDF) — one dimension per vocabulary word.
#     ✅ exact tokens: drug names, "500 mg", "CYP3A4", "SSRI"
#     ❌ paraphrase: a query saying "high blood sugar" shares ZERO words with a
#        chunk saying "hyperglycemia", so the score is exactly 0.
#
#   DENSE (sentence embeddings) — 384 learned dimensions of meaning.
#     ✅ paraphrase: "high blood sugar" lands near "hyperglycemia"/"diabetes"
#     ❌ exact tokens and negation: embeddings are fuzzy by construction, so
#        "no known interactions" sits very close to "known interactions", and a
#        rare drug name the model never saw in training becomes near-noise.
#
# A pharmaceutical corpus needs both at once: drug names are exact tokens (sparse
# wins) but users ask in lay paraphrase (dense wins). Hence both are built here.
#
# # WHY all-MiniLM-L6-v2 as the embedding model?
#   • 384 dimensions, 22 MB — it fits in Streamlit Cloud's 1 GB memory limit, which
#     larger models (768-1024 dim) do not once torch and the corpus are also loaded.
#   • Trained with a symmetric sentence-similarity objective, which matches how we
#     use it (short question vs. short passage).
#   • ~2,000 chunks/sec on CPU, so the 15k-chunk corpus encodes in well under a
#     minute — and then we cache it to disk and never pay again.
#   Cost of the choice: a 256-token input limit. That is precisely why stage 03
#   chunks to 90 words (~120 tokens) — anything longer would be silently truncated
#   by the model and therefore invisible to retrieval.

Run standalone:
    python 04_vector_representation.py        # encode the corpus, show both representations
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import load  # noqa: E402

documents_stage = load("01_documents")
preprocessing = load("02_preprocessing")
chunking = load("03_chunking")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384                 # fixed by the model above
ENCODE_BATCH_SIZE = 64          # sweet spot on CPU: big enough to amortise
                                # per-call overhead, small enough to stay in cache

# Where the encoded corpus is cached between runs.
EMB_CACHE = DATA_DIR / "doc_embeddings.npy"
EMB_META = DATA_DIR / "doc_embeddings.meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1
# The dense encoder.
#
# WHY wrap SentenceTransformer in a class instead of calling it directly?
# Three reasons the notebooks handled ad-hoc and a production app must handle
# properly: (a) the model must load exactly once per process — it is ~90 MB of
# torch weights; (b) if it cannot load, the app must keep working on BM25 alone
# rather than crash; (c) encoding the corpus must be cached to disk, with a
# cache key that is actually correct.
# ─────────────────────────────────────────────────────────────────────────────
class Embedder:
    """Dense sentence-embedding model with graceful degradation and disk caching.

    `available` is False when sentence-transformers/torch could not be imported or
    the model weights could not be downloaded. Every caller must check it; when it
    is False the pipeline runs in lexical-only mode (still fully functional).
    """

    def __init__(self, model_name: str = EMBED_MODEL_NAME, lazy: bool = False):
        self.model_name = model_name
        self.model = None
        self.available = False
        self.load_error: Optional[str] = None
        if not lazy:
            self.ensure_loaded()

    def ensure_loaded(self) -> bool:
        """Load the model on first use. Returns True if the encoder is usable."""
        if self.model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.available = True
            self.load_error = None
        except Exception as exc:
            # WHY swallow this? On a constrained host (no torch wheel, no network
            # for the weights, out of memory) the correct product behaviour is
            # "search still works, just lexically" — not a 500 page.
            self.model = None
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"
        return self.available

    # ---- encoding ----------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = ENCODE_BATCH_SIZE,
               show_progress: bool = False) -> Optional[np.ndarray]:
        """Encode texts into an (n, 384) float32 matrix of UNIT vectors.

        # WHY normalize_embeddings=True?
        # It rescales every vector to length 1. Then cosine similarity — the metric
        # we actually want — reduces to a plain dot product:
        #       cos(a,b) = (a·b) / (|a||b|)  and  |a| = |b| = 1  ⟹  cos(a,b) = a·b
        # That matters twice over: the dot product is far cheaper than computing
        # norms per query, and it lets stage 05 use FAISS's IndexFlat*IP* (inner
        # product) and get exact cosine search for free. Skipping normalisation
        # here would silently turn that index into a *dot-product* index, which
        # ranks long chunks higher regardless of relevance.
        """
        if not self.ensure_loaded():
            return None
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        # float32: FAISS requires it, and it halves memory versus float64 for no
        # measurable loss in retrieval quality.
        return np.asarray(vectors, dtype="float32")

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """Encode a single query into a (1, 384) matrix.

        # WHY a separate method for queries?
        # To make the asymmetry explicit and impossible to get wrong. Documents are
        # encoded once and cached; queries are encoded per request and never cached
        # against the corpus. This mirrors the sparse rule in step 2 — documents
        # are `fit`, queries are only ever `transform`ed.
        """
        if not self.ensure_loaded():
            return None
        return self.encode([query])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2
# The sparse (lexical) representation.
#
# # THE GOLDEN RULE, enforced structurally below:
#     documents → fit_transform()      queries → transform()   ONLY
# Fitting a vectoriser on the query would rebuild the vocabulary and IDF weights
# from a single 6-word string, so the document matrix and the query vector would
# live in different spaces and every score would be garbage. Because it produces
# plausible-looking-but-wrong numbers rather than an exception, it is one of the
# most common silent bugs in RAG code. `LexicalRepresentation` therefore exposes
# `fit(documents)` and `transform_query(q)` as separate methods — there is no API
# by which a caller can fit on a query.
# ─────────────────────────────────────────────────────────────────────────────
class LexicalRepresentation:
    """TF-IDF representation of the corpus (unigrams + bigrams).

    BM25 — the retriever we actually rank with — lives in stage 05 because it is an
    index rather than a vector space. TF-IDF is kept here because it *is* a vector
    representation, it is the baseline the notebooks measure everything against,
    and stage 06 falls back to it if `rank_bm25` is unavailable.
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1):
        # WHY bigrams? "blood pressure" and "side effects" are two-word concepts;
        # unigrams alone split them and lose the distinction from "blood" + "pressure"
        # appearing separately.
        # WHY min_df is caller-controlled: on the 15k-chunk corpus we pass min_df=2
        # to drop one-off OCR noise, which cuts the vocabulary roughly in half. On a
        # tiny uploaded document min_df=2 would delete almost everything, so it
        # defaults to 1.
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None
        self.matrix = None

    def fit(self, clean_texts: list[str]) -> "LexicalRepresentation":
        """Build the vocabulary and document matrix. DOCUMENTS ONLY."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        min_df = self.min_df if len(clean_texts) > 50 else 1
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=min_df)
        self.matrix = self.vectorizer.fit_transform(clean_texts)
        return self

    def transform_query(self, query: str) -> "np.ndarray":
        """Project a query into the EXISTING vocabulary. Never re-fits."""
        if self.vectorizer is None:
            raise RuntimeError("LexicalRepresentation.fit() must be called first.")
        cleaned = preprocessing.clean(query)
        return self.vectorizer.transform([cleaned])

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every document."""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.transform_query(query), self.matrix).flatten()

    @property
    def vocabulary_size(self) -> int:
        return 0 if self.vectorizer is None else len(self.vectorizer.vocabulary_)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3
# Encode the corpus once and cache it to disk.
#
# WHY cache? Encoding 15,000 chunks costs ~40s of CPU. On Streamlit Cloud the app
# restarts on every deploy and whenever the container is recycled, so without a
# cache every user would wait for a cold encode. With one, startup is a 23 MB
# np.load (~50 ms).
#
# # WHY a CONTENT FINGERPRINT instead of just comparing lengths?
# Comparing `len(cached) == len(chunks)` — the notebook approach — is unsound: edit
# the corpus in a way that preserves the chunk count (fix a typo, swap a drug for
# another) and the stale vectors load happily. Retrieval then scores the NEW text
# using the OLD meanings, and every answer cites the wrong chunk while looking
# perfectly healthy. Hashing the actual text makes that class of bug impossible.
# ─────────────────────────────────────────────────────────────────────────────
def corpus_fingerprint(texts: list[str], model_name: str = EMBED_MODEL_NAME) -> str:
    """Stable hash of (model, corpus content). Changes ⟹ cache must be rebuilt."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def load_cached_embeddings(fingerprint: str,
                           cache_path: Path = EMB_CACHE,
                           meta_path: Path = EMB_META) -> Optional[np.ndarray]:
    """Return cached vectors only if they were built from exactly this corpus."""
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            return None                       # corpus or model changed → stale
        vectors = np.load(cache_path)
        if meta.get("count") != len(vectors):
            return None                       # truncated/corrupt file
        return np.asarray(vectors, dtype="float32")
    except Exception:
        return None                           # unreadable cache is just a miss


def save_cached_embeddings(vectors: np.ndarray, fingerprint: str,
                           cache_path: Path = EMB_CACHE,
                           meta_path: Path = EMB_META) -> bool:
    """Persist vectors + fingerprint. Failure is non-fatal (read-only FS, etc.)."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vectors)
        meta_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "count": int(len(vectors)),
            "dim": int(vectors.shape[1]),
            "model": EMBED_MODEL_NAME,
        }, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def embed_corpus(texts: list[str], embedder: Optional[Embedder] = None,
                 use_cache: bool = True, show_progress: bool = False,
                 use_dense: bool = True) -> tuple[Optional[np.ndarray], dict]:
    """Encode the corpus, using the disk cache when it is provably valid.

    Returns (vectors_or_None, info_dict). `None` means dense retrieval is
    unavailable and the caller should run lexical-only — never an error.

    `use_dense=False` disables dense retrieval outright (the lexical-only mode the
    evaluation harness and the lightweight deploy use). It is a SEPARATE flag from
    `embedder=None`, which merely means "create the default embedder for me" —
    conflating the two silently turned the BM25-only path into a hybrid one.
    """
    info = {"cached": False, "encoded": 0, "seconds": 0.0, "model": EMBED_MODEL_NAME}
    if not texts or not use_dense:
        if not use_dense:
            info["disabled"] = True
        return None, info

    embedder = embedder or Embedder()
    if not embedder.available:
        info["error"] = embedder.load_error
        return None, info

    fingerprint = corpus_fingerprint(texts, embedder.model_name)
    if use_cache:
        cached = load_cached_embeddings(fingerprint)
        if cached is not None:
            info.update(cached=True, encoded=len(cached))
            return cached, info

    t0 = time.time()
    vectors = embedder.encode(texts, show_progress=show_progress)
    info["seconds"] = round(time.time() - t0, 1)
    info["encoded"] = 0 if vectors is None else len(vectors)
    if vectors is not None and use_cache:
        info["cache_written"] = save_cached_embeddings(vectors, fingerprint)
    return vectors, info


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("  STAGE 4 — VECTOR REPRESENTATION")
    print("=" * 78)

    docs = documents_stage.load_drug_documents()
    df = chunking.build_chunks(docs)
    print(f"Corpus: {len(df):,} chunks from {df['drug'].nunique():,} sources\n")

    # ── Sparse ───────────────────────────────────────────────────────────────
    print("── SPARSE (TF-IDF) ──")
    t0 = time.time()
    lexical = LexicalRepresentation(min_df=2).fit(df["clean"].tolist())
    print(f"  matrix     : {lexical.matrix.shape[0]:,} × {lexical.matrix.shape[1]:,}")
    print(f"  non-zeros  : {lexical.matrix.nnz:,} "
          f"({lexical.matrix.nnz / (lexical.matrix.shape[0] * lexical.matrix.shape[1]):.4%} dense)")
    print(f"  vocabulary : {lexical.vocabulary_size:,} terms")
    print(f"  build time : {time.time() - t0:.1f}s")
    print("  ↳ sparse because 99.97% of entries are zero — stored compressed.\n")

    # ── Dense ────────────────────────────────────────────────────────────────
    print("── DENSE (sentence embeddings) ──")
    embedder = Embedder()
    if not embedder.available:
        print(f"  UNAVAILABLE: {embedder.load_error}")
        print("  → pipeline would run lexical-only (still fully functional).")
        raise SystemExit(0)

    vectors, info = embed_corpus(df["text"].tolist(), embedder, show_progress=True)
    print(f"  model      : {info['model']}")
    print(f"  matrix     : {vectors.shape[0]:,} × {vectors.shape[1]}")
    print(f"  memory     : {vectors.nbytes / 1e6:.1f} MB (float32)")
    source = "disk cache" if info["cached"] else f"freshly encoded in {info['seconds']}s"
    print(f"  source     : {source}")

    norms = np.linalg.norm(vectors[:1000], axis=1)
    print(f"  unit norm  : mean {norms.mean():.6f} (must be 1.0 → cosine == dot product)")

    # ── Prove the two representations fail in opposite ways ──────────────────
    print("\n── WHY BOTH: the same query, scored two ways ──")
    probes = [
        ("paraphrase (dense should win)", "medicine for high blood sugar"),
        ("exact token (sparse should win)", "naproxen"),
    ]
    for label, q in probes:
        sparse_scores = lexical.scores(q)
        qv = embedder.encode_query(q)
        dense_scores = (vectors @ qv[0])
        s_top, d_top = int(sparse_scores.argmax()), int(dense_scores.argmax())
        print(f"\n  {label}: \"{q}\"")
        print(f"    TF-IDF best  {sparse_scores[s_top]:.3f}  "
              f"{df.loc[s_top, 'drug'][:28]:<28s} · {df.loc[s_top, 'field']}")
        print(f"    dense  best  {dense_scores[d_top]:.3f}  "
              f"{df.loc[d_top, 'drug'][:28]:<28s} · {df.loc[d_top, 'field']}")
        print(f"    TF-IDF found nothing at all: {sparse_scores.max() == 0.0}")

    print("\nStage 4 complete — vectors ready for the vector store (stage 05).")

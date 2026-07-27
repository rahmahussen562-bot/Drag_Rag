"""
06_retrieve_context.py — STAGE 6:  **Context Retrieval**

    Raw Documents  →  Preprocessing  →  Chunking  →  Vector Representation  →  Vector Store
    →  >>> Context Retrieval <<<  →  Prompt Building  →  LLM Generation  →  Streamlit UI

# WHY does this stage exist?
# This is where the "R" in RAG happens, and it is the stage that decides whether
# the whole system is trustworthy. The LLM in stage 07 is forbidden from using
# outside knowledge, so it can only ever be as correct as what arrives here. Every
# hallucination that survives the prompt rules is really a retrieval failure.
#
# Four mechanisms, each earning its place with a measurement:
#
#   1. HYBRID SCORING — blend BM25 (exact tokens) with dense cosine (paraphrase).
#      They fail in opposite cases, so the blend beats either alone.
#
#   2. ENTITY GROUNDING — before retrieving anything, ask "does this query even
#      name a drug we have?". # WHY? Vector search ALWAYS returns its k nearest
#      neighbours, even when the nearest neighbour is irrelevant. Ask about a drug
#      not in the corpus and naive RAG happily retrieves five chunks about a
#      different drug and answers confidently from them. For a pharmacy assistant
#      that is the single most dangerous failure mode, so we refuse up front.
#
#   3. RELEVANCE FLOOR — drop results scoring below a threshold. # WHY? Grounding
#      catches "wrong drug"; the floor catches "right drug, wrong question" (e.g.
#      asking for a price, which FDA labels do not contain). If nothing clears the
#      floor we return an EMPTY list, which stage 07 turns into an honest refusal.
#      Retrieval returning fewer than k results is a feature here, not a bug.
#
#   4. CROSS-ENCODER RERANKING (optional) — the measured biggest quality win.
#      Bi-encoders embed query and chunk SEPARATELY, so the model never sees them
#      together and can only compare two summaries. A cross-encoder reads the pair
#      jointly and scores true relevance, but costs a full transformer pass per
#      candidate — far too slow for 15k chunks. Hence the standard cascade:
#             retrieve ~40 cheaply  →  rerank those 40  →  keep the best k.
#      Measured on this corpus (Scale-Edition notebook, 60 queries): MRR 0.42 →
#      0.80, Hit@5 0.72 → 0.88.
#
# # WHY IS ALPHA 0.20 AND NOT THE NOTEBOOKS' 0.6?
# The notebooks set α=0.6 (lexical-heavy), justified by "drug names are exact
# tokens, where BM25 shines". That was correct FOR THE 12-DRUG corpus it was tuned
# on. It does not survive the real 506-drug corpus, where BM25 alone collapses to
# MRR 0.22 while embeddings hold 0.56.
#
# `eval/tune_alpha.py` sweeps α over the ground-truth set and measures it:
#
#     ALPHA   drug recall@5   field recall@5   field MRR
#     0.00         95%             95%           0.867     ← best fields, loses a DRUG
#     0.20        100%             91%           0.822     ← chosen
#     0.60        100%             68%           0.502     ← the notebooks' value
#     1.00        100%             41%           0.409     ← BM25 only
#
# The choice is lexicographic, not a single score. Missing the right FIELD answers
# the wrong section of the right drug — unhelpful, and visible in the citation.
# Missing the right DRUG answers from a different medicine entirely — fluent,
# confidently cited, and clinically wrong. So drug recall is maximised first, and
# field recall optimised only within the settings that keep it at 100%.
#
# That rules out pure-dense (α=0) despite its better field recall: with no lexical
# weight an exact drug-name match earns no score of its own. Keeping BM25 at 0.20
# is what guarantees the named drug is retrieved — which is precisely the reason
# to run a hybrid at all rather than embeddings alone.
#
# Net effect vs. the notebooks' constant: +23 points of field recall (68% → 91%).

Run standalone:
    python 06_retrieve_context.py                     # demo retrieval + abstention
    python 06_retrieve_context.py "how does metformin work?"
"""
from __future__ import annotations

import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import load  # noqa: E402

documents_stage = load("01_documents")
preprocessing = load("02_preprocessing")
chunking = load("03_chunking")
representation = load("04_vector_representation")
vector_store = load("05_create_vector_store")

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval configuration
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = 0.20           # hybrid weight on BM25; (1-ALPHA) goes to dense. Measured,
                       # not guessed — see the sweep table in the header and
                       # `python eval/tune_alpha.py` to reproduce it.
RETRIEVE_POOL = 40     # candidates fetched before the cross-encoder reranks them
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Relevance floors — a result must clear ONE of these to count as relevant.
# Calibrated so that on-topic questions keep all k results while off-topic ones
# (pricing, unrelated drugs) return nothing at all.
EMB_FLOOR = 0.25       # minimum raw cosine similarity
BM25_FLOOR = 1.0       # minimum raw BM25 score


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1
# Entity grounding vocabulary.
#
# The corpus stores generic names WITH salt forms ("Metformin Hydrochloride",
# "Rizatriptan Benzoate"). A user types "metformin". So we index each drug by its
# IDENTITY tokens — the name with salt/dosage-form words stripped — and treat a
# leftover content word in the query that matches no identity token as an unknown
# drug. That is the signal to refuse.
# ─────────────────────────────────────────────────────────────────────────────
_SALT_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "sulfate", "sulphate", "succinate", "fumarate", "tartrate", "maleate",
    "mesylate", "besylate", "citrate", "phosphate", "acetate", "dipropionate",
    "bromide", "chloride", "hydrobromide", "monohydrate", "dihydrate", "hydrate",
    "benzoate", "carbonate", "nitrate", "tromethamine", "dihydrochloride",
    "and", "with", "for", "in", "of", "tablets", "tablet", "capsules", "capsule",
    "injection", "oral", "topical", "extended", "release", "cream", "ointment",
    "solution", "suspension", "gel", "spray", "er", "xr", "sr",
}

# Pharmacological and interrogative vocabulary — these words appear in almost
# every question and identify no drug, so they must never trigger a refusal.
_GENERIC_Q_WORDS = {
    "side", "effect", "effects", "adverse", "reaction", "reactions", "dose",
    "dosage", "doses", "dosing", "mg", "interaction", "interactions", "interact",
    "interacts", "drug", "drugs", "medication", "medicine", "medicines", "use",
    "used", "uses", "using", "usage", "work", "works", "working", "mechanism",
    "action", "indication", "indications", "indicated", "treat", "treats",
    "treatment", "contraindication", "contraindications", "warning", "warnings",
    "precaution", "precautions", "take", "taking", "taken", "give", "given",
    "patient", "patients", "common", "serious", "severe", "risk", "risks",
    "cause", "causes", "caused", "symptom", "symptoms", "what", "how", "why",
    "when", "who", "which", "whom", "does", "do", "did", "is", "are", "was",
    "can", "could", "should", "would", "the", "a", "an", "about", "tell", "me",
    "my", "you", "your", "i", "list", "name", "names", "between", "versus", "vs",
    "or", "on", "to", "from", "any", "some", "all", "there", "have", "has",
    "safe", "safety", "avoid", "during", "while", "pregnancy", "pregnant",
    "document", "documents", "file", "text", "say", "says", "said", "explain",
    "summarize", "summary", "describe", "according", "mentioned",
}

# ≥3 alphabetic characters, hyphens allowed ("co-trimoxazole").
_ALPHA_TOKEN = re.compile(r"[a-z][a-z\-]{2,}")


def identity_tokens(name: str) -> set:
    """Identity tokens of a source name, with salt/dosage-form words removed."""
    return {t for t in _ALPHA_TOKEN.findall(str(name).lower()) if t not in _SALT_WORDS}


def salient_query_terms(query: str) -> list:
    """Content words in a query that could plausibly BE a drug name."""
    stop = preprocessing._STOP
    seen, out = set(), []
    for t in _ALPHA_TOKEN.findall(query.lower()):
        if t in stop or t in _GENERIC_Q_WORDS or t in _SALT_WORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2
# Result types — explicit so the UI and the evaluator agree on what happened.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """One retrieved chunk plus everything needed to cite and display it."""
    index: int          # row in the chunk table — the join key for everything
    score: float        # final ranking score (hybrid, or reranker score)
    text: str
    drug: str
    title: str
    field: str
    origin: str
    bm25: float = 0.0   # component scores, surfaced in the UI's evaluation mode
    dense: float = 0.0

    def citation(self) -> str:
        return f"{self.title} · {self.field}"


@dataclass
class RetrievalOutcome:
    """The full result of a retrieval attempt, including WHY it returned nothing.

    status:
      "ok"                  — chunks found above the relevance floor
      "unknown_entity"      — the query names something not in the corpus → refuse
      "no_entity"           — the query names nothing retrievable → ask for a drug
      "no_relevant_context" — entity known, but nothing cleared the floor → refuse
    """
    status: str
    chunks: list[RetrievedChunk]
    matched: set
    unknown: list
    suggestions: list
    query: str

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.chunks)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3
# The retriever.
# ─────────────────────────────────────────────────────────────────────────────
class Retriever:
    """Grounded hybrid retrieval with abstention and optional reranking."""

    def __init__(self, store, embedder=None, alpha: float = ALPHA,
                 use_reranker: bool = False):
        self.store = store
        self.df = store.df
        self.embedder = embedder
        self.alpha = alpha
        self.reranker = None
        self._rerank_error = None

        # 3a — Build the grounding vocabulary from the corpus itself, so it can
        #      never drift out of sync with what is actually indexed.
        self.entity_vocab: set = set()
        self.chunk_tokens: list[set] = []
        for name in self.df["drug"]:
            toks = identity_tokens(name)
            self.chunk_tokens.append(toks)
            self.entity_vocab |= toks

        # 3b — Chunks that came from an upload are exempt from drug grounding:
        #      a user who uploads a report is entitled to ask about its contents
        #      using words that are not drug names.
        self.has_uploads = bool((self.df["origin"] != "drug-corpus").any())
        self.upload_mask = (self.df["origin"] != "drug-corpus").to_numpy()

        if use_reranker:
            self._load_reranker()

    def _load_reranker(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)
            return True
        except Exception as exc:
            self.reranker = None
            self._rerank_error = f"{type(exc).__name__}: {exc}"
            return False

    # ---- grounding ---------------------------------------------------------
    def ground(self, query: str) -> dict:
        """Which corpus entities does this query name, and which are unknown?"""
        terms = salient_query_terms(query)
        matched = {t for t in terms if t in self.entity_vocab}
        unknown = [t for t in terms if t not in self.entity_vocab]
        return {"terms": terms, "matched": matched, "unknown": unknown}

    def suggest(self, unknown: list) -> list:
        """Fuzzy "did you mean…?" for unknown terms. Improves a refusal from a
        dead end into something actionable."""
        vocab = sorted(self.entity_vocab)
        out = []
        for term in unknown:
            match = difflib.get_close_matches(term, vocab, n=1, cutoff=0.82)
            if match:
                out.append((term, match[0]))
        return out

    # ---- scoring -----------------------------------------------------------
    @staticmethod
    def _minmax(a: np.ndarray) -> np.ndarray:
        """Rescale to [0,1] so two incomparable score scales can be summed.

        # WHY is this necessary? BM25 is unbounded (0 to ~20 here) while cosine
        # lives in [-1,1]. Summing them raw would let BM25 dominate purely because
        # its numbers are bigger, which is a units bug, not a ranking decision.
        """
        mn, mx = float(a.min()), float(a.max())
        if mx - mn < 1e-12:
            return np.full_like(a, 0.5)
        return (a - mn) / (mx - mn)

    def _raw_scores(self, query: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
        bm25 = self.store.lexical.scores_all(query)
        dense = None
        if self.store.has_dense and self.embedder is not None and self.embedder.available:
            qv = self.embedder.encode_query(query)
            if qv is not None:
                dense = self.store.dense.scores_all(qv)
        return bm25, dense

    # ---- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, k: int = 5,
                 restrict_tokens: Optional[set] = None) -> list[RetrievedChunk]:
        """Rank chunks and return up to k that clear the relevance floor.

        May return FEWER than k, or none at all — that is how abstention works.
        """
        if self.df.empty:
            return []

        raw_bm25, raw_dense = self._raw_scores(query)
        if raw_dense is None:
            combined = raw_bm25                      # lexical-only mode
        else:
            combined = (self.alpha * self._minmax(raw_bm25)
                        + (1 - self.alpha) * self._minmax(raw_dense))

        n = len(combined)
        candidates = np.arange(n)

        # 3c — Metadata filter: restrict to chunks belonging to the named drug(s).
        #
        # # WHY restrict instead of trusting the ranking?
        # "What are the side effects of metformin?" must not return atorvastatin's
        # side-effect chunk just because it is textually similar — the two answers
        # are equally fluent and one is wrong. Restricting first turns a hard
        # 15,000-way problem into an easy ~30-way one (which FIELD of this drug),
        # which is where the measured accuracy gain comes from.
        if restrict_tokens:
            mask = np.fromiter(
                (bool(self.chunk_tokens[i] & restrict_tokens) for i in range(n)),
                dtype=bool, count=n)
            if self.has_uploads:
                # Uploaded material stays eligible: it may well be what the user
                # is asking about, and it is not covered by the drug vocabulary.
                mask = mask | self.upload_mask
            if mask.any():
                candidates = np.where(mask)[0]

        order = candidates[np.argsort(combined[candidates])[::-1]]

        def clears_floor(i: int) -> bool:
            if raw_dense is not None:
                return bool(raw_dense[i] >= EMB_FLOOR or raw_bm25[i] >= BM25_FLOOR)
            return bool(raw_bm25[i] >= BM25_FLOOR)

        # 3d — Optional rerank cascade over the floor-passing pool.
        if self.reranker is not None:
            pool = [int(i) for i in order[:RETRIEVE_POOL] if clears_floor(int(i))]
            if not pool:
                return []
            pairs = [(query, self.df.at[i, "text"]) for i in pool]
            scores = self.reranker.predict(pairs)
            best = np.argsort(scores)[::-1][:k]
            return [self._make(pool[int(j)], float(scores[int(j)]), raw_bm25, raw_dense)
                    for j in best]

        out = []
        for i in order:
            i = int(i)
            if clears_floor(i):
                out.append(self._make(i, float(combined[i]), raw_bm25, raw_dense))
                if len(out) == k:
                    break
        return out

    def _make(self, i: int, score: float, raw_bm25, raw_dense) -> RetrievedChunk:
        row = self.df.loc[i]
        return RetrievedChunk(
            index=i, score=score, text=row["text"], drug=row["drug"],
            title=row["title"], field=row["field"], origin=row["origin"],
            bm25=float(raw_bm25[i]),
            dense=float(raw_dense[i]) if raw_dense is not None else 0.0,
        )

    # ---- the full grounded flow -------------------------------------------
    def retrieve_context(self, query: str, k: int = 5) -> RetrievalOutcome:
        """Ground → restrict → retrieve → floor. The entry point stage 07 calls."""
        ground = self.ground(query)
        matched, unknown = ground["matched"], ground["unknown"]

        # An upload-bearing corpus skips drug grounding entirely — the user is
        # asking about their own document, not about our drug database.
        if not matched and not self.has_uploads:
            status = "unknown_entity" if unknown else "no_entity"
            return RetrievalOutcome(status, [], matched, unknown,
                                    self.suggest(unknown), query)

        chunks = self.retrieve(query, k=k, restrict_tokens=matched or None)
        if not chunks:
            return RetrievalOutcome("no_relevant_context", [], matched, unknown,
                                    [], query)
        return RetrievalOutcome("ok", chunks, matched, unknown, [], query)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4
# Convenience builder — wires stages 01→06 together in one call.
# ─────────────────────────────────────────────────────────────────────────────
def build_retriever(df=None, use_embeddings: bool = True, use_reranker: bool = False,
                    alpha: float = ALPHA, show_progress: bool = False) -> Retriever:
    """Build a ready-to-query Retriever over the built-in corpus (or a given df)."""
    if df is None:
        df = chunking.build_chunks(documents_stage.load_drug_documents())
    embedder = representation.Embedder() if use_embeddings else None
    store = vector_store.build_store(df, embedder=embedder, show_progress=show_progress,
                                     use_dense=use_embeddings)
    return Retriever(store, embedder=embedder, alpha=alpha, use_reranker=use_reranker)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("  STAGE 6 — CONTEXT RETRIEVAL")
    print("=" * 78)

    retriever = build_retriever(show_progress=True)
    print(f"Corpus      : {len(retriever.df):,} chunks")
    print(f"Entities    : {len(retriever.entity_vocab):,} identity tokens")
    print(f"Dense store : {retriever.store.has_dense}")
    print(f"Alpha       : {retriever.alpha}  "
          f"(BM25 {retriever.alpha:.0%} / dense {1 - retriever.alpha:.0%})")

    queries = sys.argv[1:] or [
        "how does metformin work?",                 # known drug → should answer
        "what are the side effects of naproxen?",   # known drug → should answer
        "what are the side effects of aspirin?",    # NOT in corpus → should refuse
        "what is the price of metformin in Egypt?", # known drug, unanswerable field
        "hello there",                              # names nothing → should ask
    ]

    for q in queries:
        print("\n" + "─" * 78)
        print(f"Q: {q}")
        outcome = retriever.retrieve_context(q, k=5)
        print(f"   status  : {outcome.status}")
        print(f"   matched : {sorted(outcome.matched) or '—'}")
        if outcome.unknown:
            print(f"   unknown : {outcome.unknown}")
        if outcome.suggestions:
            print(f"   suggest : {outcome.suggestions}")
        for rank, c in enumerate(outcome.chunks, 1):
            print(f"   [{rank}] {c.score:.3f} (bm25={c.bm25:6.2f} dense={c.dense:.3f})  "
                  f"{c.citation()[:52]}")
        if not outcome.chunks:
            print("   → no context returned; stage 07 will refuse rather than guess.")

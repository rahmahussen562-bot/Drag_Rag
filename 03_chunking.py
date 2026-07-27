"""
03_chunking.py — STAGE 3 of the RAG pipeline:  **Chunking**

    Raw Documents  →  Preprocessing  →  >>> Chunking <<<  →  Vector Representation
    →  Vector Store  →  Context Retrieval  →  Prompt Building  →  LLM Generation  →  Streamlit UI

# WHY does this stage exist?
# Chunking sets the ceiling on everything downstream. Retrieval returns *chunks*,
# the LLM sees only the chunks retrieval returned, and a citation points at a
# chunk. So chunk boundaries decide what the system is even capable of saying.
#
# Two failure modes bracket the design:
#
#   TOO BIG  — embed a whole 3,000-word drug label as one vector and you get a
#              blurry average. It matches every drug question weakly and none
#              strongly, and the one relevant sentence is diluted by 2,900 words
#              of unrelated text. The LLM then has to find the needle itself.
#   TOO SMALL— a 15-word chunk loses its own context: "reduce the dose by half"
#              is useless when you can no longer see which drug or which patient
#              group it refers to.
#
# This project solves it in two passes:
#   PASS 1 (free, done in stage 01) — split on the document's OWN structure. An
#          FDA label already separates mechanism / dosage / side effects, and a
#          markdown file already has headings. Author-declared boundaries are the
#          best chunk boundaries available and they cost nothing.
#   PASS 2 (here) — anything still too long gets a SLIDING WINDOW of 90 words
#          with 25 words of overlap.
#
# # WHY overlap at all?
# Without it a fact landing on a boundary is cut in half and neither half is
# retrievable: "...do not exceed | 4 g per day..." becomes two chunks, one saying
# "do not exceed" and one saying "4 g per day". The 25-word overlap guarantees any
# fact shorter than 25 words appears intact in at least one window.
#
# # WHY 90 / 25 specifically?
# 90 words ≈ 120 tokens, comfortably inside the 256-token window of the
# all-MiniLM-L6-v2 embedding model used in stage 04 — text past that limit is
# silently truncated by the model, i.e. invisible to retrieval. 25/90 ≈ 28%
# overlap is the usual sweet spot: enough to survive boundary cuts, not so much
# that the index bloats (overlap cost is 1/(1-0.28) ≈ 1.4× more chunks).

Run standalone:
    python 03_chunking.py             # chunk the real corpus and print the statistics
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import load  # noqa: E402

# Reuse the previous stages instead of copying their code (see stages.py).
documents_stage = load("01_documents")
preprocessing = load("02_preprocessing")

Document = documents_stage.Document

# ─────────────────────────────────────────────────────────────────────────────
# Chunking configuration — these two numbers are the most consequential knobs in
# the whole project, so they live in one obvious place and are quoted verbatim
# in the README.
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_WORDS = 90      # sliding-window size, in words
CHUNK_OVERLAP = 25    # words shared between neighbouring windows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1
# The sliding window itself.
#
# WHY words and not characters or tokens?
# Characters cut mid-word. Real tokenisers are model-specific and would couple
# this stage to whichever embedding model stage 04 happens to use. Words are a
# stable, model-agnostic unit that maps predictably onto tokens (~1.3 tokens per
# English word), which is all the precision this decision needs.
# ─────────────────────────────────────────────────────────────────────────────
def sliding_windows(text: str, size: int = CHUNK_WORDS,
                    overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split `text` into overlapping word windows.

    Short text is returned as a single window — we never pad, because a padded
    chunk would put noise into the index.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size}); "
                         f"otherwise the window never advances and this loops forever.")
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [text]

    step = size - overlap          # how far the window advances each iteration
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        # Stop once this window already reached the end, so we never emit a final
        # sliver that is pure duplicate of the previous window's tail.
        if i + size >= len(words):
            break
        i += step
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2
# Turn Documents into the chunk table the rest of the pipeline works on.
#
# WHY a pandas DataFrame rather than a list of objects?
# Because stage 06 needs three things that DataFrames give for free: positional
# indexing (retrieval returns row numbers that must line up 1:1 with the embedding
# matrix and the FAISS index), vectorised boolean masks (metadata filtering, e.g.
# "only chunks belonging to metformin"), and cheap groupby for the statistics the
# UI displays.
# ─────────────────────────────────────────────────────────────────────────────
def build_chunks(docs: list[Document], size: int = CHUNK_WORDS,
                 overlap: int = CHUNK_OVERLAP, profile: str | None = None) -> pd.DataFrame:
    """Chunk a document list into the canonical chunk table.

    Returns a DataFrame with one row per chunk and these columns:
        doc_id   — id of the parent Document (lets us group chunks back together)
        drug     — lowercased source name; the citation/grouping key
        title    — display name shown to the user
        category — coarse group, used for filtering
        field    — which part of the source ("dosage", "page 3", …)
        win      — window number within that field (0 for un-split documents)
        text     — the chunk sent to the embedding model and to the LLM (RAW)
        clean    — the same chunk after stage-02 cleaning (lexical index only)
        origin   — provenance ("drug-corpus" / "upload:notes.pdf")
    """
    profile = profile or preprocessing.DEFAULT_PROFILE_NAME
    records = []

    for doc in docs:
        windows = sliding_windows(doc.text, size=size, overlap=overlap)
        for j, window in enumerate(windows):
            # STEP 2a — make every chunk SELF-DESCRIBING.
            #
            # WHY prefix each chunk with "Name — field: "?
            # A retrieved chunk is torn out of its document and handed to the LLM
            # alone. Without the prefix, a dosage window reads "500 mg twice daily
            # with food" — true of hundreds of drugs and therefore useless and
            # dangerous. The prefix also injects the drug name into the chunk's
            # own embedding and BM25 tokens, which is what lets a query naming a
            # drug pull back that drug's chunks at all.
            records.append({
                "doc_id": doc.doc_id,
                "drug": doc.title.lower(),
                "title": doc.title,
                "category": doc.category.lower(),
                "field": doc.field,
                "win": j,
                "text": f"{doc.title} — {doc.field}: {window}",
                "origin": doc.origin,
            })

    if not records:
        # An empty-but-correctly-shaped frame keeps stages 04-06 from having to
        # special-case "no documents yet" (which happens before the first upload).
        return pd.DataFrame(columns=["doc_id", "drug", "title", "category", "field",
                                     "win", "text", "origin", "clean", "word_count"])

    df = pd.DataFrame(records).reset_index(drop=True)

    # STEP 2b — attach the cleaned variant for the lexical retrievers.
    #
    # WHY store both `text` and `clean`?
    # They feed different consumers and must not be confused:
    #   `clean` → BM25 / TF-IDF, which need normalised tokens to match;
    #   `text`  → the embedding model AND the LLM context, which both want natural
    #             language. Sending stemmed, stopword-stripped text to either one
    #             measurably degrades results.
    df["clean"] = preprocessing.clean_many(df["text"].tolist(), profile=profile)
    df["word_count"] = df["text"].str.split().str.len()
    return df


def chunk_documents(docs: list[Document], **kwargs) -> pd.DataFrame:
    """Alias kept for readability at call sites."""
    return build_chunks(docs, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3
# Merge chunk tables — needed when a user uploads files on top of the built-in
# corpus.
#
# WHY re-index instead of concatenating naively?
# Row position IS the identity of a chunk downstream: the FAISS index in stage 05
# stores integer ids that are row numbers, and stage 06 looks results up with
# `df.loc[idx]`. Duplicate or stale index values would silently mis-cite — the
# answer would quote one chunk and the UI would display a different one.
# ─────────────────────────────────────────────────────────────────────────────
def merge_chunks(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate chunk tables into one contiguously indexed frame."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return build_chunks([])
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def chunk_stats(df: pd.DataFrame) -> dict:
    """Summary used by the standalone run and the Streamlit sidebar."""
    if df.empty:
        return {"chunks": 0, "sources": 0, "mean_words": 0.0, "max_words": 0,
                "windowed_chunks": 0, "by_field": {}}
    return {
        "chunks": len(df),
        "sources": df["drug"].nunique(),
        "mean_words": round(float(df["word_count"].mean()), 1),
        "min_words": int(df["word_count"].min()),
        "max_words": int(df["word_count"].max()),
        # How many chunks exist only because a field was long enough to split?
        # This is the direct, measurable payoff of the sliding window.
        "windowed_chunks": int((df["win"] > 0).sum()),
        "by_field": df["field"].value_counts().head(10).to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("  STAGE 3 — CHUNKING")
    print("=" * 78)

    docs = documents_stage.load_drug_documents()
    df = build_chunks(docs)
    stats = chunk_stats(df)

    print(f"Window size / overlap : {CHUNK_WORDS} words / {CHUNK_OVERLAP} words "
          f"({CHUNK_OVERLAP / CHUNK_WORDS:.0%})")
    print(f"Documents in          : {len(docs):,}")
    print(f"Chunks out            : {stats['chunks']:,}  "
          f"({stats['chunks'] / len(docs):.2f} chunks per document)")
    print(f"Sources               : {stats['sources']:,}")
    print(f"Words per chunk       : mean {stats['mean_words']}, "
          f"min {stats['min_words']}, max {stats['max_words']}")
    print(f"Created by windowing  : {stats['windowed_chunks']:,} chunks "
          f"(i.e. fields long enough to need splitting)")

    print("\nChunks per field:")
    for fld, n in stats["by_field"].items():
        print(f"  {fld:<20s} {n:>7,}")

    # ── Demonstrate the overlap guarantee on a real long field ───────────────
    long_docs = [d for d in docs if d.word_count > CHUNK_WORDS * 2]
    if long_docs:
        demo = long_docs[0]
        wins = sliding_windows(demo.text)
        print(f"\nOverlap demo — '{demo.title}' · {demo.field} "
              f"({demo.word_count} words → {len(wins)} windows)")
        # The LAST `overlap` words of window 0 must be the FIRST `overlap` words
        # of window 1 — that shared span is what makes boundary facts survivable.
        tail = wins[0].split()[-CHUNK_OVERLAP:]
        head = wins[1].split()[:CHUNK_OVERLAP]
        shared = sum(1 for a, b in zip(tail, head) if a == b)
        print(f"  window 0 tail   : …{' '.join(tail[:8])} …")
        print(f"  window 1 head   : …{' '.join(head[:8])} …")
        print(f"  shared words    : {shared}/{CHUNK_OVERLAP}  "
              f"← identical span, so a fact straddling the cut survives intact")

    print("\nSample chunk (self-describing prefix in bold):")
    print(" ", df.loc[0, "text"][:220], "…")
    print("\nSame chunk after stage-02 cleaning (what BM25 indexes):")
    print(" ", df.loc[0, "clean"][:220], "…")

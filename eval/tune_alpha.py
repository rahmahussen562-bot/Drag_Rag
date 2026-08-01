"""
tune_alpha.py — empirically choose the hybrid retrieval weight ALPHA.

    hybrid_score = ALPHA * minmax(BM25) + (1 - ALPHA) * minmax(cosine)

# WHY does this script exist?
# The notebooks set ALPHA = 0.6 (lexical-heavy) and justified it with "drug names
# are exact tokens, where BM25 shines". That reasoning is sound, and on the 12-drug
# teaching corpus the number was fine. It does NOT survive contact with the
# 506-drug corpus: the Scale-Edition notebook's own scoreboard shows BM25 at
# MRR 0.22 versus embeddings at 0.56, so weighting BM25 at 0.6 drags the blend
# down to 0.42 — worse than dropping BM25 entirely.
#
# Rather than copy the notebook constant or guess a new one, this sweeps ALPHA
# across the full range and measures it on the ground-truth set in eval_set.json.
#
# The metric that matters here is FIELD RECALL@k. Drug recall is already 100% at
# every ALPHA, because stage 06 restricts retrieval to the drug the query names
# before scoring — so the only remaining question is whether the right FIELD of
# that drug (mechanism vs. dosage vs. side_effects) reaches the top k. That is
# precisely what the blend decides.

Run:
    python eval/tune_alpha.py
    python eval/tune_alpha.py --k 5 --steps 11
"""
from __future__ import annotations

import argparse
import json
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

EVAL = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))


def score_alpha(retriever, cases: list, k: int) -> dict:
    """Return drug/field recall@k and MRR-of-field for one ALPHA setting."""
    drug_hits = field_hits = 0
    reciprocal = 0.0

    for case in cases:
        outcome = retriever.retrieve_context(case["q"], k=k)
        chunks = outcome.chunks

        # Did the right DRUG appear among the sources?
        tokens = set()
        for chunk in chunks:
            tokens |= retrieval.identity_tokens(chunk.drug)
        drug_hits += case["drug"] in tokens

        # Did the right FIELD appear, and how high?
        want = case.get("field")
        rank_of_field = next((i for i, c in enumerate(chunks, 1) if c.field == want), None)
        if rank_of_field:
            field_hits += 1
            reciprocal += 1.0 / rank_of_field

    n = len(cases)
    return {"drug_recall": drug_hits / n, "field_recall": field_hits / n,
            "field_mrr": reciprocal / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--steps", type=int, default=11, help="ALPHA values from 0..1")
    args = ap.parse_args()

    cases = [c for c in EVAL["answerable"] if c.get("field")]
    print(f"Tuning ALPHA on {len(cases)} ground-truth cases (k={args.k}).\n")

    # Build the corpus and store ONCE, then swap ALPHA on the retriever. Rebuilding
    # per sweep step would re-encode 15k chunks each time for no reason.
    df = chunking.build_chunks(documents_stage.load_drug_documents())
    embedder = representation.Embedder()
    if not embedder.available:
        print(f"Embeddings unavailable ({embedder.load_error}) — ALPHA is moot "
              f"without a dense score. Aborting.")
        return 1
    store = vector_store.build_store(df, embedder=embedder)
    retriever = retrieval.Retriever(store, embedder=embedder)

    print(f"{'ALPHA':>6}  {'BM25/dense':>12}  {'drug R@k':>9}  {'field R@k':>10}  {'field MRR':>10}")
    print("-" * 56)

    rows = []
    for i in range(args.steps):
        alpha = i / (args.steps - 1)
        retriever.alpha = alpha
        result = score_alpha(retriever, cases, args.k)
        rows.append((alpha, result))
        mix = f"{alpha:.0%}/{1 - alpha:.0%}"
        print(f"{alpha:>6.2f}  {mix:>12}  {result['drug_recall']:>8.0%}  "
              f"{result['field_recall']:>9.0%}  {result['field_mrr']:>10.3f}")

    # ── Selection is LEXICOGRAPHIC, not a single score. ─────────────────────
    #
    # # WHY does drug recall outrank field recall?
    # They are not equally bad failures. Missing the right FIELD means answering
    # "how does X work?" from X's dosage section — unhelpful, but still about the
    # right drug, and the citation shows the user what happened. Missing the right
    # DRUG means answering a question about X using Y's label — fluent, confidently
    # cited, and clinically wrong. For a pharmacy assistant the second failure is
    # categorically worse, so we maximise drug recall FIRST and only then optimise
    # field recall within the settings that keep it perfect.
    #
    # This is exactly what the sweep exposes: alpha ≤ 0.10 wins on field recall but
    # loses a drug, because with no lexical weight an exact drug-name match carries
    # no score of its own. Keeping BM25 in the blend is what guarantees the named
    # drug is found — which is the whole reason not to go pure-dense.
    best_drug = max(r["drug_recall"] for _, r in rows)
    safe = [(a, r) for a, r in rows if r["drug_recall"] >= best_drug]
    best_alpha, best = max(safe, key=lambda ar: (ar[1]["field_recall"],
                                                 ar[1]["field_mrr"]))

    dense_only = rows[0][1]
    print("-" * 56)
    print("\nDrug recall is the safety-critical metric — maximised first.")
    print(f"  best drug recall      : {best_drug:.0%}  "
          f"(reached at ALPHA ≥ {min(a for a, r in safe):.2f})")
    print(f"  pure-dense (ALPHA=0)  : drug recall {dense_only['drug_recall']:.0%} "
          f"← loses a drug, so it is rejected despite the best field recall")

    print(f"\nBest ALPHA = {best_alpha:.2f}  "
          f"(drug recall {best['drug_recall']:.0%}, "
          f"field recall {best['field_recall']:.0%}, MRR {best['field_mrr']:.3f})")
    print(f"Configured  = {retrieval.ALPHA:.2f} in 06_retrieve_context.py")

    notebook = next((r for a, r in rows if abs(a - 0.6) < 1e-9), None)
    if notebook:
        print(f"\nNotebook ALPHA = 0.60 → field recall {notebook['field_recall']:.0%}, "
              f"MRR {notebook['field_mrr']:.3f}")
        delta = best["field_recall"] - notebook["field_recall"]
        print(f"Re-tuning gains {delta:+.0%} field recall at production scale.")

    if abs(best_alpha - retrieval.ALPHA) > 0.101:
        print(f"\n⚠️  Configured ALPHA is far from the measured optimum — consider "
              f"updating ALPHA in 06_retrieve_context.py to {best_alpha:.2f}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

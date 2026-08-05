"""
tune_field_boost.py — empirically choose FIELD_BOOST, the field-routing weight.

    boosted = hybrid_score * (1 + FIELD_BOOST * normalised_field_preference)

# WHY does this script exist?
# Because "routing by intent should help field recall" is a hypothesis, and this
# project's rule is that a parameter earns its value from a table or it does not
# ship. FIELD_BOOST starts at 0.0 — a no-op — and only moves if this sweep says it
# should.
#
# THE METRIC THAT MATTERS is field recall@k. Drug recall is already 100% at every
# setting, because stage 06 restricts to the named drug BEFORE scoring, and the
# boost is applied after that restriction. So the only question the boost can
# affect is whether the right SECTION of the right drug reaches the top k.
#
# THE CONSTRAINT: drug recall@5 must stay 100%. It is checked at every step and
# any setting that breaks it is rejected regardless of its field recall — the same
# lexicographic rule tune_alpha.py uses, for the same clinical reason. Missing the
# right field answers the wrong section of the right drug; missing the right drug
# answers from a different medicine.

Run:
    python eval/tune_field_boost.py
    python eval/tune_field_boost.py --k 5 --steps 9
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

from retrieval.field_router import field_weights  # noqa: E402
from retrieval.query_understanding import classify_intents  # noqa: E402

documents_stage = load("01_documents")
chunking = load("03_chunking")
representation = load("04_vector_representation")
vector_store = load("05_create_vector_store")
retrieval = load("06_retrieve_context")

EVAL = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))


def score_boost(retriever, cases: list, boost: float, k: int) -> dict:
    """drug/field recall@k and field MRR at one FIELD_BOOST setting."""
    drug_hits = field_hits = 0
    reciprocal = 0.0
    misses = []

    for case in cases:
        intents = classify_intents(case["q"])
        weights = field_weights(intents)
        outcome = retriever.retrieve_context(
            case["q"], k=k, field_weights=weights, field_boost=boost)
        chunks = outcome.chunks

        tokens = set()
        for chunk in chunks:
            tokens |= retrieval.identity_tokens(chunk.drug)
        drug_hits += case["drug"] in tokens

        want = case.get("field")
        rank = next((i for i, c in enumerate(chunks, 1) if c.field == want), None)
        if rank:
            field_hits += 1
            reciprocal += 1.0 / rank
        else:
            misses.append(case["q"])

    n = len(cases)
    return {"drug_recall": drug_hits / n, "field_recall": field_hits / n,
            "field_mrr": reciprocal / n, "misses": misses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--steps", type=int, default=9, help="FIELD_BOOST values 0..0.8")
    args = ap.parse_args()

    cases = [c for c in EVAL["answerable"] if c.get("field")]
    print(f"Tuning FIELD_BOOST on {len(cases)} ground-truth cases (k={args.k}).\n")

    df = chunking.build_chunks(documents_stage.load_drug_documents())
    embedder = representation.Embedder()
    if not embedder.available:
        print(f"Embeddings unavailable ({embedder.load_error}). Aborting.")
        return 1
    store = vector_store.build_store(df, embedder=embedder)
    retriever = retrieval.Retriever(store, embedder=embedder)

    # Show what the classifier decided, so a wrong routing is visible as a
    # classification problem rather than being blamed on the boost.
    print("Intent classification (top intent per case):")
    for case in cases:
        intents = classify_intents(case["q"])
        top = max(intents, key=intents.get) if intents else "—"
        mark = "✔" if top == case.get("field") else " "
        print(f"  {mark} {case['q'][:46]:46s} → {top:20s} (want {case['field']})")
    print()

    print(f"{'BOOST':>6}  {'drug R@k':>9}  {'field R@k':>10}  {'field MRR':>10}")
    print("-" * 42)

    rows = []
    for i in range(args.steps):
        boost = round(i * 0.1, 2)
        result = score_boost(retriever, cases, boost, args.k)
        rows.append((boost, result))
        print(f"{boost:>6.2f}  {result['drug_recall']:>8.0%}  "
              f"{result['field_recall']:>9.0%}  {result['field_mrr']:>10.3f}")

    print("-" * 42)

    baseline = rows[0][1]
    best_drug = max(r["drug_recall"] for _, r in rows)
    safe = [(b, r) for b, r in rows if r["drug_recall"] >= best_drug]
    best_boost, best = max(safe, key=lambda br: (br[1]["field_recall"],
                                                 br[1]["field_mrr"]))

    print(f"\nbaseline (BOOST=0.00) : field recall {baseline['field_recall']:.0%}, "
          f"MRR {baseline['field_mrr']:.3f}")
    print(f"best safe setting     : BOOST={best_boost:.2f} → field recall "
          f"{best['field_recall']:.0%}, MRR {best['field_mrr']:.3f}")
    print(f"drug recall held at   : {best['drug_recall']:.0%}")

    delta = best["field_recall"] - baseline["field_recall"]
    mrr_delta = best["field_mrr"] - baseline["field_mrr"]
    print(f"\ndelta vs no routing   : {delta:+.0%} field recall, "
          f"{mrr_delta:+.3f} MRR")

    if best["misses"]:
        print(f"\nstill missing the right field at BOOST={best_boost:.2f}:")
        for q in best["misses"]:
            print(f"  · {q}")

    # THE VERDICT. A parameter that does not improve a metric does not ship —
    # and the negative result gets written down rather than quietly dropped.
    print("\n" + "=" * 60)
    if delta > 0 or (delta == 0 and mrr_delta > 0.01):
        print(f"VERDICT: ADOPT FIELD_BOOST = {best_boost:.2f}")
        print(f"         {delta:+.0%} field recall at unchanged drug recall.")
    else:
        print("VERDICT: KEEP FIELD_BOOST = 0.0 — routing did not help.")
        print("         Record the negative result in README.md; do not ship a")
        print("         parameter that buys nothing.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

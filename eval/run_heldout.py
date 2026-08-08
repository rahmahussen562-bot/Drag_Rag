"""
run_heldout.py — score field routing on phrasings it was NOT tuned against.

# WHY does this exist?
# `tune_field_boost.py` reported field recall 100% and field MRR exactly 1.000 on
# eval_set.json. That number is not trustworthy on its own: the intent classifier
# was written while looking at that file, so the sweep measures memorisation as
# much as generalisation. An MRR of 1.000 is a red flag, not a result.
#
# This script runs the same machinery over eval_heldout.json — alternative natural
# phrasings of the same information needs, over drugs already verified present in
# the corpus. If the gain survives here, field routing generalises. If it collapses,
# the honest conclusion is that the classifier memorised its tuning set, and the
# number to publish is this one.

Run:
    python eval/run_heldout.py
    python eval/run_heldout.py --boost 0.2
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

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boost", type=float, default=0.20)
    ap.add_argument("--set", default="eval_heldout2.json",
                    help="which held-out file to score")
    args = ap.parse_args()

    # # WHY default to the SECOND held-out file?
    # Because eval_heldout.json stopped being held-out the moment the intent
    # patterns were widened knowing which of its cases they missed. That is the
    # same trap that produced a fake MRR of 1.000 on eval_set.json. It is kept as
    # a regression check; improvements are reported on the fresh set.
    path = Path(__file__).parent / args.set
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    print(f"Held-out evaluation — {len(cases)} cases from {path.name}\n")

    df = chunking.build_chunks(documents_stage.load_drug_documents())
    embedder = representation.Embedder()
    if not embedder.available:
        print(f"Embeddings unavailable ({embedder.load_error}). Aborting.")
        return 1
    store = vector_store.build_store(df, embedder=embedder)
    retriever = retrieval.Retriever(store, embedder=embedder)

    # ── 1. Does the classifier get the intent right on unseen phrasings? ────
    correct = unmatched = 0
    print("INTENT CLASSIFICATION")
    for case in cases:
        intents = classify_intents(case["q"])
        top = max(intents, key=intents.get) if intents else None
        if top is None:
            unmatched += 1
            mark = "∅"
        elif top == case["field"]:
            correct += 1
            mark = "✔"
        else:
            mark = "✘"
        print(f"  {mark} {case['q'][:48]:48s} → {str(top):20s} (want {case['field']})")
    n = len(cases)
    print(f"\n  intent accuracy : {correct}/{n} = {correct/n:.0%}"
          f"   (no match at all: {unmatched})\n")

    # ── 2. Does routing actually improve retrieval here? ────────────────────
    def measure(boost: float) -> dict:
        field_hits = drug_hits = 0
        reciprocal = 0.0
        misses = []
        for case in cases:
            weights = field_weights(classify_intents(case["q"])) if boost else None
            out = retriever.retrieve_context(case["q"], k=args.k,
                                             field_weights=weights, field_boost=boost)
            toks = set()
            for c in out.chunks:
                toks |= retrieval.identity_tokens(c.drug)
            drug_hits += case["drug"] in toks
            rank = next((i for i, c in enumerate(out.chunks, 1)
                         if c.field == case["field"]), None)
            if rank:
                field_hits += 1
                reciprocal += 1.0 / rank
            else:
                misses.append(case["q"])
        return {"drug_recall": drug_hits / n, "field_recall": field_hits / n,
                "field_mrr": reciprocal / n, "misses": misses}

    off, on = measure(0.0), measure(args.boost)

    print("RETRIEVAL")
    print(f"{'setting':>22}  {'drug R@k':>9}  {'field R@k':>10}  {'field MRR':>10}")
    print("-" * 58)
    print(f"{'no routing (0.00)':>22}  {off['drug_recall']:>8.0%}  "
          f"{off['field_recall']:>9.0%}  {off['field_mrr']:>10.3f}")
    print(f"{f'routing ({args.boost:.2f})':>22}  {on['drug_recall']:>8.0%}  "
          f"{on['field_recall']:>9.0%}  {on['field_mrr']:>10.3f}")
    print("-" * 58)

    d_field = on["field_recall"] - off["field_recall"]
    d_mrr = on["field_mrr"] - off["field_mrr"]
    print(f"\ndelta : {d_field:+.0%} field recall, {d_mrr:+.3f} MRR")

    if on["misses"]:
        print(f"\nstill missing the right field ({len(on['misses'])}):")
        for q in on["misses"]:
            print(f"  · {q}")

    print("\n" + "=" * 60)
    if on["drug_recall"] < 1.0:
        print("VERDICT: REJECT — drug recall fell below 100%. Safety-critical.")
    elif d_field > 0 or d_mrr > 0.01:
        print(f"VERDICT: field routing GENERALISES. {d_field:+.0%} field recall,")
        print(f"         {d_mrr:+.3f} MRR on phrasings it was not tuned on.")
    else:
        print("VERDICT: routing did NOT generalise. The gain on eval_set.json was")
        print("         memorisation. Report this number, not that one.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

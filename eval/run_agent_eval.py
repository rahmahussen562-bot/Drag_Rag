"""
run_agent_eval.py — evaluate what the APP actually does, per persona.

# WHY does this exist alongside run_eval.py?
# Because they stopped measuring the same thing, and that gap is a defect.
#
# `run_eval.py` drives `rag_pipeline.RAGEngine` — the raw retrieval path, with no
# persona, no field routing and no enforcement. After Phase 1 the app routes by
# intent and after Phase 2 it answers through an agent, so run_eval.py now reports
# a number the product no longer produces. A harness that measures a path nobody
# ships is worse than no harness: it is a green tick for the wrong system.
#
# This script drives `agents.get_agent(...).answer(...)` — byte-identical to what
# the Streamlit app calls. Both are kept: run_eval.py is the un-routed BASELINE,
# this is the shipped configuration. Reporting them side by side is what makes the
# routing gain legible instead of asserted.
#
# # WHY is the patient's "false refusal" rate expected to be non-zero?
# Because the persona is SUPPOSED to refuse some answerable questions. A dosage
# titration question is answerable from the corpus and must still be declined for
# a lay reader. So this script separates:
#     policy refusals   — the persona's screens fired (correct behaviour)
#     retrieval misses  — nothing was found (a real failure)
# Collapsing those into one number would make the safest persona look the worst.

Run:
    python eval/run_agent_eval.py
    python eval/run_agent_eval.py --markdown     # paste-ready table for the README
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from stages import load  # noqa: E402

from agents import available, config_of, get_agent  # noqa: E402

documents_stage = load("01_documents")
chunking = load("03_chunking")
representation = load("04_vector_representation")
vector_store = load("05_create_vector_store")
retrieval = load("06_retrieve_context")

EVAL = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))

#: Statuses produced by a persona POLICY rather than by retrieval failing.
POLICY_STATUSES = {"red_flag", "medication_change", "persona_field_policy",
                   "interaction_referral"}


def in_policy(config, field: str) -> bool:
    """Can this persona read the label section the ground truth expects?

    # WHY does field recall have to be conditioned on this?
    # eval_set.json's expected fields were written for a full-access reader: five
    # cases expect `mechanism`, two expect `dosage`, three expect `interactions`.
    # The patient persona is DELIBERATELY forbidden those sections. Scoring it on
    # finding them measures the safety policy, not the retrieval — the first run of
    # this harness reported the patient at 32% field recall, which reads as "the
    # safe persona is worse" when it actually means "the metric does not apply".
    #
    # So field recall is computed only over cases the persona is allowed to answer
    # fully, and `out_of_policy` reports how many were excluded. Coverage and
    # accuracy are different questions and are reported as different numbers.
    """
    if field is None:
        return False
    if field in config.blocked_fields:
        return False
    if config.allowed_fields is not None and field not in config.allowed_fields:
        return False
    return True


def evaluate(agent, cases: list, refuse_cases: list, k: int) -> dict:
    config = agent.config
    drug_hits = field_hits = 0
    reciprocal = 0.0
    policy_refusals = retrieval_misses = out_of_policy = 0
    in_policy_answered = 0
    context_chars = []
    latencies = []

    for case in cases:
        t0 = time.perf_counter()
        result = agent.answer(case["q"], RETRIEVER, k=k, use_llm=False)
        latencies.append((time.perf_counter() - t0) * 1000)

        if result.refused:
            if result.status in POLICY_STATUSES:
                policy_refusals += 1
            else:
                retrieval_misses += 1
            continue

        context_chars.append(len(result.context or ""))
        tokens = set()
        for chunk in result.chunks:
            tokens |= retrieval.identity_tokens(chunk.drug)
        drug_hits += case["drug"] in tokens

        # Field recall only counts where the persona is ALLOWED to reach the
        # expected section — see in_policy() for why.
        if not in_policy(config, case.get("field")):
            out_of_policy += 1
            continue
        in_policy_answered += 1
        rank = next((i for i, c in enumerate(result.chunks, 1)
                     if c.field == case.get("field")), None)
        if rank:
            field_hits += 1
            reciprocal += 1.0 / rank

    # Refusal accuracy on out-of-corpus questions — must be 100% for every persona.
    correct_refusals = 0
    for case in refuse_cases:
        out = agent.answer(case["q"], RETRIEVER, k=k, use_llm=False)
        correct_refusals += out.refused

    n, m = len(cases), len(refuse_cases)
    answered = n - policy_refusals - retrieval_misses
    return {
        "answered": answered,
        "in_policy": in_policy_answered,
        "out_of_policy": out_of_policy,
        "drug_recall": drug_hits / answered if answered else 0.0,
        "field_recall": (field_hits / in_policy_answered
                         if in_policy_answered else float("nan")),
        "field_mrr": (reciprocal / in_policy_answered
                      if in_policy_answered else float("nan")),
        "policy_refusals": policy_refusals,
        "retrieval_misses": retrieval_misses,
        "refusal_accuracy": correct_refusals / m if m else 0.0,
        "mean_context_chars": (sum(context_chars) / len(context_chars)
                               if context_chars else 0),
        "mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="override each persona's k")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    global RETRIEVER
    df = chunking.build_chunks(documents_stage.load_drug_documents())
    embedder = representation.Embedder()
    if not embedder.available:
        print(f"Embeddings unavailable ({embedder.load_error}). Aborting.")
        return 1
    store = vector_store.build_store(df, embedder=embedder)
    RETRIEVER = retrieval.Retriever(store, embedder=embedder)

    answerable, refuse = EVAL["answerable"], EVAL["refuse"]
    print(f"Per-persona evaluation — {len(answerable)} answerable, "
          f"{len(refuse)} must-refuse. Both personas share ONE retriever.\n")

    results = {}
    for agent_id in available():
        cfg = config_of(agent_id)
        agent = get_agent(agent_id)
        k = args.k or cfg.k
        results[agent_id] = evaluate(agent, answerable, refuse, k)
        results[agent_id]["k"] = k

    rows = []
    for agent_id, r in results.items():
        cfg = config_of(agent_id)
        rows.append({
            "persona": cfg.label,
            "k": r["k"],
            "answered": f"{r['answered']}/{len(answerable)}",
            "drug R@k": f"{r['drug_recall']:.0%}",
            "in-policy": f"{r['in_policy']}/{r['answered']}",
            "field R@k*": f"{r['field_recall']:.0%}",
            "field MRR*": f"{r['field_mrr']:.3f}",
            "policy ref": r["policy_refusals"],
            "misses": r["retrieval_misses"],
            "refusal acc": f"{r['refusal_accuracy']:.0%}",
            "ctx chars": f"{r['mean_context_chars']:,.0f}",
            "ms": f"{r['mean_ms']:.0f}",
        })

    headers = list(rows[0])
    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            print("| " + " | ".join(str(row[h]) for h in headers) + " |")
    else:
        widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
        print("  " + "  ".join(h.ljust(widths[h]) for h in headers))
        print("  " + "  ".join("-" * widths[h] for h in headers))
        for row in rows:
            print("  " + "  ".join(str(row[h]).ljust(widths[h]) for h in headers))

    print("\nHow to read this:")
    print("  in-policy   — cases whose EXPECTED field this persona is allowed to")
    print("                read. eval_set.json was written for a full-access")
    print("                reader: 5 cases expect `mechanism`, 2 `dosage`, 3")
    print("                `interactions` — all blocked for the patient.")
    print("  field R@k*  — computed over IN-POLICY cases only. Scoring a persona on")
    print("                sections its own safety policy forbids measures the")
    print("                policy, not the retrieval. Before this correction the")
    print("                patient reported 32%, which reads as 'the safe persona")
    print("                is worse' and means nothing of the kind.")
    print("  policy ref  — a persona screen fired (dose-change, emergency). CORRECT")
    print("                behaviour; excluded from the recall denominators.")
    print("  misses      — nothing was retrieved at all. A real failure.")
    print("  refusal acc — out-of-corpus questions refused. Must be 100%.")
    print("\n  NOTE: the personas are NOT directly comparable on field R@k — they")
    print("  are scored over different case subsets. Proper per-persona ground")
    print("  truth (brief section 4.5) is the fix; this is the honest interim.")

    failures = [a for a, r in results.items()
                if r["refusal_accuracy"] < 1.0 or r["drug_recall"] < 0.95]
    print("\nRESULT:", "PASS ✅" if not failures else f"FAIL ❌ {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

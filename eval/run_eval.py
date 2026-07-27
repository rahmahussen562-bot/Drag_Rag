"""
run_eval.py — validate PhARMA RAG retrieval + abstention against a ground-truth set.

Two things we care about for a *pharmacist* assistant:
  1. Abstention  — does it REFUSE drugs it doesn't cover (instead of answering
     from unrelated sources)? This is the safety-critical metric.
  2. Retrieval   — for drugs it DOES cover, does the right drug show up in the
     top-k sources (drug recall@k), and the right field (field recall@k)?

Run:
    python eval/run_eval.py                # fast BM25-only (matches cloud default)
    python eval/run_eval.py --embeddings   # hybrid semantic
    python eval/run_eval.py --rerank       # hybrid + cross-encoder
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

from rag_pipeline import RAGEngine, _drug_identity_tokens  # noqa: E402

EVAL = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))
REFUSAL_MODES = {"refused", "no-relevant-context"}


def source_drug_tokens(engine, results):
    toks = set()
    for idx, _ in results:
        toks |= _drug_identity_tokens(engine.df.loc[idx, "drug"])
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true", help="use semantic + hybrid retrieval")
    ap.add_argument("--rerank", action="store_true", help="use cross-encoder reranker")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    use_emb = args.embeddings or args.rerank
    eng = RAGEngine(use_embeddings=use_emb, use_reranker=args.rerank)
    print(f"Engine: retriever={eng.retriever_name}  drugs={len(eng.drugs_data)}  "
          f"chunks={len(eng.df)}  k={args.k}\n")

    # ── Answerable: must NOT refuse; expect the right drug in top-k ──────────
    ans = EVAL["answerable"]
    drug_hits = field_hits = false_refusals = 0
    print("ANSWERABLE (should answer, right drug in sources)")
    for c in ans:
        text, results, meta = eng.answer(c["q"], k=args.k, use_llm=False)
        refused = meta["mode"] in REFUSAL_MODES
        got = source_drug_tokens(eng, results)
        drug_ok = c["drug"] in got
        field_ok = any(eng.df.loc[i, "field"] == c.get("field") for i, _ in results)
        false_refusals += refused
        drug_hits += drug_ok
        field_hits += field_ok
        flag = "REFUSED!" if refused else ("ok" if drug_ok else "miss-drug")
        mark = "✔" if (drug_ok and not refused) else "✗"
        print(f"  {mark} [{flag:9s}] {c['q'][:48]:48s} drug={drug_ok} field={field_ok}")

    n = len(ans)
    print(f"\n  drug recall@{args.k} : {drug_hits}/{n} = {drug_hits/n:.0%}")
    print(f"  field recall@{args.k}: {field_hits}/{n} = {field_hits/n:.0%}")
    print(f"  false refusals    : {false_refusals}/{n} = {false_refusals/n:.0%}  (lower is better)\n")

    # ── Refuse: must refuse (unknown/out-of-corpus drug or no drug) ──────────
    ref = EVAL["refuse"]
    correct_refusals = 0
    print("REFUSE (should refuse, drug not in corpus)")
    for c in ref:
        text, results, meta = eng.answer(c["q"], k=args.k, use_llm=False)
        refused = meta["mode"] in REFUSAL_MODES
        correct_refusals += refused
        mark = "✔" if refused else "✗"
        print(f"  {mark} [{meta['mode']:20s}] {c['q'][:44]:44s} ({c['why']})")

    m = len(ref)
    print(f"\n  refusal accuracy  : {correct_refusals}/{m} = {correct_refusals/m:.0%}\n")

    # ── Interactions: structured DDInter lookup must be exactly right ────────
    inter_acc = 1.0
    inter = EVAL.get("interactions", [])
    if inter:
        from interactions import InteractionDB
        db = InteractionDB()
        inter_ok = 0
        print("INTERACTIONS (structured DDInter lookup, exact-match)")
        for c in inter:
            r = db.check_pair(c["a"], c["b"])
            if "level" in c:
                good = r["status"] == "ok" and r.get("level") == c["level"]
                exp = f"ok/{c['level']}"
            else:
                good = r["status"] == c["status"]
                exp = c["status"]
            inter_ok += good
            got = r["status"] + (f"/{r.get('level')}" if r["status"] == "ok" else "")
            print(f"  {'✔' if good else '✗'} {c['a']:12s}+{c['b']:14s} expect={exp:16s} got={got}")
        inter_acc = inter_ok / len(inter)
        print(f"\n  interaction accuracy: {inter_ok}/{len(inter)} = {inter_acc:.0%}\n")

    # ── Headline ────────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"SUMMARY  drug_recall@{args.k}={drug_hits/n:.0%}  "
          f"refusal_acc={correct_refusals/m:.0%}  "
          f"false_refusal={false_refusals/n:.0%}  "
          f"interaction_acc={inter_acc:.0%}")
    ok = (drug_hits / n >= 0.8) and (correct_refusals / m >= 0.9) \
        and (false_refusals / n <= 0.1) and (inter_acc >= 0.9)
    print("RESULT:", "PASS ✅" if ok else "NEEDS TUNING ⚠️")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

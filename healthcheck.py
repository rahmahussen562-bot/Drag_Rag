"""
healthcheck.py — is this deployment actually able to answer?

    python -m healthcheck            # human-readable
    python -m healthcheck --json     # for a container HEALTHCHECK / monitor
    python -m healthcheck --strict   # degraded modes count as failure

# WHY does this exist separately from the eval suite?
# They answer different questions. `run_e2e.py` asks "is the pipeline CORRECT?" and
# takes minutes. This asks "is this container ABLE TO SERVE?" and takes seconds.
# An ops team pages on the second one.
#
# # WHY are degraded modes not failures by default?
# Because they are documented, intentional behaviour: no torch → BM25-only, no LLM
# → extractive, no faiss → NumPy matmul. Every one of those still answers, cites
# and refuses correctly. Reporting them as DOWN would train an operator to ignore
# the alert. They are reported as DEGRADED with the reason, and `--strict` is there
# for the deployment that genuinely requires the full stack.

Exit codes:  0 = healthy (or degraded)   1 = unhealthy   2 = degraded under --strict
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, DEGRADED, FAIL = "ok", "degraded", "fail"


def _check(name: str, fn) -> dict:
    t0 = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as exc:  # a check must never crash the health check
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    return {"check": name, "status": status, "detail": detail,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


def check_config() -> tuple:
    from config import CONFIG
    if CONFIG.warnings:
        return DEGRADED, f"{len(CONFIG.warnings)} warning(s): {CONFIG.warnings[0]}"
    return OK, f"loaded from {Path(CONFIG.source).name if Path(CONFIG.source).exists() else CONFIG.source}"


def check_corpus() -> tuple:
    from stages import load
    docs = load("01_documents")
    path = docs.default_corpus_path()
    if not path.exists():
        return FAIL, f"corpus missing at {path}"
    records = docs.load_drug_records()
    return OK, f"{len(records):,} drug records from {path.name}"


def check_embeddings() -> tuple:
    from stages import load
    rep = load("04_vector_representation")
    if not rep.EMB_CACHE.exists():
        return DEGRADED, ("embedding cache absent — first boot will re-encode "
                          "(~300 s, ~1.2 GB peak). Commit data/doc_embeddings.npy.")
    mb = rep.EMB_CACHE.stat().st_size / 1e6
    emb = rep.Embedder()
    if not emb.available:
        return DEGRADED, f"encoder unavailable ({emb.load_error}) → BM25-only mode"
    return OK, f"{rep.EMBED_MODEL_NAME} · cache {mb:.1f} MB"


# ─────────────────────────────────────────────────────────────────────────────
# Deep checks share ONE index build.
#
# # WHY does this need saying?
# The first version of this file built the whole index twice — once in
# check_index and again in check_retrieval — which took minutes. A health check
# that is slower than a deploy is a health check nobody runs. Build once, reuse.
# ─────────────────────────────────────────────────────────────────────────────
_STORE = None


def _shared_store():
    global _STORE
    if _STORE is None:
        from stages import load
        docs, chunking = load("01_documents"), load("03_chunking")
        rep, vs = load("04_vector_representation"), load("05_create_vector_store")
        df = chunking.build_chunks(docs.load_drug_documents())
        embedder = rep.Embedder()
        _STORE = (vs.build_store(df, embedder=embedder), embedder)
    return _STORE


def check_index() -> tuple:
    store, _ = _shared_store()
    rows = len(store.df)
    if store.has_dense and len(store.dense.vectors) != rows:
        return FAIL, (f"ROW MISALIGNMENT: {rows:,} chunks vs "
                      f"{len(store.dense.vectors):,} vectors — would mis-cite")
    backend = store.dense.backend if store.has_dense else "lexical-only"
    return (OK if store.has_dense else DEGRADED), \
        f"{rows:,} chunks · dense={backend} · lexical={store.lexical.backend}"


def check_retrieval() -> tuple:
    """Does a known-good query still ground, and a known-absent one still refuse?"""
    from stages import load
    ret = load("06_retrieve_context")
    store, embedder = _shared_store()
    retriever = ret.Retriever(store, embedder=embedder)

    good = retriever.retrieve_context("how does metformin work?", k=3)
    if not good.ok:
        return FAIL, f"a known-good query returned nothing (status={good.status})"
    # The abstention gate is the product. If it stops firing, the deployment is
    # unsafe even though every other check is green.
    bad = retriever.retrieve_context("what are the side effects of aspirin?", k=3)
    if bad.ok:
        return FAIL, "ABSTENTION GATE OPEN — an out-of-corpus drug returned sources"
    return OK, f"grounded ok · abstention ok ({len(retriever.entity_vocab)} entities)"


def check_agents() -> tuple:
    from agents import available, config_of
    ids = available()
    if not ids:
        return FAIL, "no personas registered"
    missing = [i for i in ids if not config_of(i).prompt().strip()]
    if missing:
        return FAIL, f"prompt file missing/empty for: {missing}"
    return OK, f"{len(ids)} personas: {', '.join(ids)}"


def check_interactions() -> tuple:
    from config import CONFIG
    if not CONFIG.interactions.enabled:
        return DEGRADED, "interaction lookup disabled in config"
    from interactions import get_provider, licence_of
    db = get_provider(CONFIG.interactions.provider)
    lic = licence_of(CONFIG.interactions.provider)
    note = "" if lic["commercial_use"] else "  ⚠️ NON-COMMERCIAL licence"
    return OK, f"{lic['name']} · {len(db.pairs):,} pairs · {lic['licence']}{note}"


def check_llm() -> tuple:
    import llm_providers
    status = llm_providers.llm_status()
    if status.get("active"):
        return OK, f"{status['active']} · {status.get('model') or 'default model'}"
    return DEGRADED, ("no provider configured → extractive mode (retrieved text, "
                      "still cited and disclaimed)")


# FAST: file/config/provider checks. Seconds. Safe to run on an interval.
FAST_CHECKS = [
    ("config", check_config),
    ("corpus", check_corpus),
    ("embeddings", check_embeddings),
    ("agents", check_agents),
    ("interactions", check_interactions),
    ("llm", check_llm),
]

# DEEP: builds the index and exercises the abstention gate. Minutes, not seconds.
#
# # WHY is this NOT the default?
# Because a container HEALTHCHECK runs every 30 s and would spend its life
# rebuilding a 15k-chunk index. Use --deep at build time and after a deploy, where
# "can this thing actually answer and still refuse?" is the question worth minutes.
DEEP_CHECKS = [
    ("index", check_index),
    ("retrieval", check_retrieval),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="PhARMA RAG deployment health check.")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true",
                    help="treat degraded modes as failure")
    ap.add_argument("--deep", action="store_true",
                    help="also build the index and test the abstention gate "
                         "(minutes — use at build/deploy time, not on an interval)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    checks = FAST_CHECKS + (DEEP_CHECKS if args.deep else [])
    results = [_check(name, fn) for name, fn in checks]
    failed = [r for r in results if r["status"] == FAIL]
    degraded = [r for r in results if r["status"] == DEGRADED]

    overall = FAIL if failed else (DEGRADED if degraded else OK)
    code = 1 if failed else (2 if (degraded and args.strict) else 0)

    if args.json:
        print(json.dumps({"status": overall, "exit": code, "checks": results},
                         indent=2, ensure_ascii=False))
        return code

    icon = {OK: "✅", DEGRADED: "⚠️ ", FAIL: "❌"}
    width = max(len(r["check"]) for r in results)
    print("PhARMA RAG — health check\n")
    for r in results:
        print(f"  {icon[r['status']]} {r['check'].ljust(width)}  {r['detail']}"
              f"   ({r['ms']:.0f} ms)")
    print(f"\nOVERALL: {overall.upper()}"
          + (f"  ({len(degraded)} degraded)" if degraded else "")
          + ("" if args.deep else
             "   [fast checks only — add --deep to test retrieval + abstention]"))
    if degraded and not args.strict:
        print("  Degraded modes are documented, intentional fallbacks — the app "
              "still answers,\n  cites and refuses correctly. Use --strict if this "
              "deployment requires the full stack.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

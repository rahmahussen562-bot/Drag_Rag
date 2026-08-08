"""
run_brand_eval.py — brand resolution grounds what it should, refuses what it must.

# WHY is the REFUSAL half of this suite the important half?
# Because the dangerous version of brand resolution is the one that works "better".
# In this corpus `acetaminophen` exists only inside opioid combination products, so
# a naive map would answer "is Tylenol safe?" from an Oxycodone label — fluent,
# correctly cited, and about a completely different medicine. Testing only that
# Glucophage resolves would pass while that bug shipped.

Run:  python eval/run_brand_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from stages import load  # noqa: E402

retrieval = load("06_retrieve_context")
prompting = load("07_prompting")

ok = True


def check(label: str, cond, detail: str = "") -> None:
    global ok
    ok = ok and bool(cond)
    print(f"  {'✅' if cond else '❌'} {label}" + (f"   {detail}" if detail else ""))


#: (query, generic that must be grounded)
MUST_GROUND = [
    ("what is Glucophage used for?", "metformin"),
    ("Lipitor side effects", "atorvastatin"),
    ("Coumadin contraindications", "warfarin"),
    ("how does Prilosec work?", "omeprazole"),
    ("Zoloft indications", "sertraline"),
]

#: Brands whose generic exists ONLY inside combination products.
MUST_REFUSE_COMBINATION = ["is Tylenol safe?", "paracetamol dosage",
                           "adderall interactions"]

#: Brands (or generics) genuinely absent from the corpus.
MUST_REFUSE_ABSENT = ["what are the side effects of aspirin?",
                      "how does ibuprofen work?", "loratadine side effects",
                      "advil dosage"]


def main() -> int:
    r = retrieval.build_retriever()
    print(f"brand map: {len(r.brand_resolve)} resolvable, "
          f"{len(r.brand_refuse)} refuse-only\n")
    check("a brand map is actually loaded", bool(r.brand_resolve))

    print("\n[1] MUST GROUND — generic exists as a standalone record")
    for q, want in MUST_GROUND:
        out = r.retrieve_context(q, k=5)
        got = set()
        for c in out.chunks:
            got |= retrieval.identity_tokens(c.drug)
        check(f"{q[:40]:40s} → {want}", out.ok and want in got,
              f"aliases={out.aliases}")

    print("\n[2] MUST REFUSE — generic is combination-only "
          "(resolving would answer about a DIFFERENT medicine)")
    for q in MUST_REFUSE_COMBINATION:
        out = r.retrieve_context(q, k=5)
        check(f"{q[:40]:40s} refused", not out.ok, f"status={out.status}")
        ans = prompting.answer_question(q, r, k=5, use_llm=False)
        check("    …and the refusal explains why",
              "combination" in ans.text.lower())

    print("\n[3] MUST REFUSE — generic genuinely absent from the corpus")
    for q in MUST_REFUSE_ABSENT:
        out = r.retrieve_context(q, k=5)
        check(f"{q[:40]:40s} refused", not out.ok, f"status={out.status}")

    print("\n[4] a rewrite is never silent")
    out = r.retrieve_context("what is Glucophage used for?", k=5)
    check("the alias is reported for the UI",
          out.aliases == {"glucophage": "metformin"}, str(out.aliases))

    print("\n[5] a missing map degrades safely (no mapping > wrong mapping)")
    res, ref = retrieval.load_brand_map(Path("does_not_exist.json"))
    check("missing file → empty maps", res == {} and ref == {})

    print("\n" + "=" * 62)
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

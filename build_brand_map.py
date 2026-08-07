"""
build_brand_map.py — generate data/brand_map.json from the corpus itself.

    python build_brand_map.py            # rebuild and report
    python build_brand_map.py --check    # verify the committed map is still valid

# WHY does the brand map need GENERATING rather than hand-writing?
# Because whether a mapping is safe depends on what is actually in the corpus, and
# that changes every time `01_documents.py --refresh` runs. A hand-written map goes
# stale silently, and a stale brand map does not fail — it MIS-GROUNDS.
#
# ═══════════════════════════════════════════════════════════════════════════
# THE SAFETY RULE: resolve a brand only to a STANDALONE record.
# ═══════════════════════════════════════════════════════════════════════════
# The obvious implementation — "is the generic's token anywhere in the corpus?" —
# is dangerous, and this corpus proves it:
#
#   acetaminophen appears ONLY inside:
#       Benzhydrocodone And Acetaminophen
#       Hydrocodone Bitartrate And Acetaminophen
#       Oxycodone And Acetaminophen          ← all OPIOID combination products
#
# So `tylenol -> acetaminophen` would ground a question about paracetamol to an
# opioid label, and answer it with opioid dosing, warnings and contraindications.
# Fluent, correctly cited, and about a completely different medicine — with no
# signal to the reader that anything went wrong.
#
# A brand therefore resolves only when some record's identity tokens are EXACTLY
# {generic}. Everything else is still recorded, but as `refuse_only`: the name is
# recognised so the refusal can explain itself ("Tylenol is acetaminophen, which I
# only hold as combination products"), without ever grounding the query.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stages import load  # noqa: E402

documents_stage = load("01_documents")
retrieval = load("06_retrieve_context")

OUT_PATH = ROOT / "data" / "brand_map.json"

# Curated brand → generic pairs. Provenance matters more than volume: these are
# common brand names a user actually types. RxNorm (rxnorm.py) covers the long
# tail at runtime when a network is available; this file is the offline floor.
BRANDS = {
    "glucophage": "metformin", "fortamet": "metformin", "riomet": "metformin",
    "glumetza": "metformin",
    "tylenol": "acetaminophen", "panadol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nurofen": "ibuprofen",
    "aleve": "naproxen", "naprosyn": "naproxen", "anaprox": "naproxen",
    "coumadin": "warfarin", "jantoven": "warfarin",
    "lipitor": "atorvastatin", "zocor": "simvastatin", "crestor": "rosuvastatin",
    "prilosec": "omeprazole", "nexium": "esomeprazole",
    "zoloft": "sertraline", "prozac": "fluoxetine", "lexapro": "escitalopram",
    "celexa": "citalopram", "paxil": "paroxetine",
    "xanax": "alprazolam", "valium": "diazepam", "ativan": "lorazepam",
    "klonopin": "clonazepam",
    "zestril": "lisinopril", "prinivil": "lisinopril",
    "neurontin": "gabapentin", "lyrica": "pregabalin",
    "amoxil": "amoxicillin", "augmentin": "amoxicillin",
    "deltasone": "prednisone", "zyprexa": "olanzapine",
    "zofran": "ondansetron", "flomax": "tamsulosin",
    "amaryl": "glimepiride", "glucotrol": "glipizide",
    "lamictal": "lamotrigine", "requip": "ropinirole",
    "chantix": "varenicline", "pepcid": "famotidine",
    "singulair": "montelukast", "synthroid": "levothyroxine",
    "plavix": "clopidogrel", "norvasc": "amlodipine",
    "cozaar": "losartan", "diovan": "valsartan",
    "viagra": "sildenafil", "cialis": "tadalafil",
    "adderall": "amphetamine", "ritalin": "methylphenidate",
    "claritin": "loratadine", "zyrtec": "cetirizine",
    "ventolin": "albuterol", "proventil": "albuterol",
    "aspirin": "acetylsalicylic acid", "asa": "acetylsalicylic acid",
}


def standalone_generics(records) -> dict:
    """{generic token: display name} for records that are a SINGLE ingredient.

    A record qualifies only when its identity tokens (salt and dosage-form words
    stripped by stage 06) reduce to exactly one token. `Sildenafil Citrate` →
    {sildenafil} qualifies; `Oxycodone And Acetaminophen` → {oxycodone,
    acetaminophen} does not.
    """
    out: dict = {}
    for rec in records:
        name = str(rec.get("name", "")).strip()
        if not name:
            continue
        tokens = retrieval.identity_tokens(name)
        if len(tokens) == 1:
            token = next(iter(tokens))
            # Prefer the shortest display name for a token (the plainest form).
            if token not in out or len(name) < len(out[token]):
                out[token] = name
    return out


def build() -> dict:
    records = documents_stage.load_drug_records()
    standalone = standalone_generics(records)
    all_tokens = set()
    for rec in records:
        all_tokens |= retrieval.identity_tokens(str(rec.get("name", "")))

    resolve: dict = {}
    refuse_only: dict = {}

    for brand, generic in sorted(BRANDS.items()):
        head = generic.split()[0]           # "acetylsalicylic acid" → token match
        if head in standalone:
            resolve[brand] = {"generic": head, "record": standalone[head]}
        elif head in all_tokens:
            refuse_only[brand] = {
                "generic": generic,
                "reason": "present only within combination products",
            }
        else:
            refuse_only[brand] = {"generic": generic,
                                  "reason": "not in corpus"}

    return {
        "_comment": (
            "Generated by build_brand_map.py — do not hand-edit. "
            "`resolve` maps a brand to a generic that exists as a STANDALONE "
            "record and is safe to ground. `refuse_only` names the generic so a "
            "refusal can explain itself, but never grounds the query: resolving "
            "to a combination product would answer about a different medicine."),
        "corpus_records": len(records),
        "resolve": resolve,
        "refuse_only": refuse_only,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the committed map matches the current corpus")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    fresh = build()

    if args.check:
        if not OUT_PATH.exists():
            print(f"❌ {OUT_PATH.name} missing — run `python build_brand_map.py`")
            return 1
        current = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        same = (current.get("resolve") == fresh["resolve"]
                and current.get("refuse_only") == fresh["refuse_only"])
        print("✅ brand map matches the corpus" if same else
              "❌ brand map is STALE — regenerate it (the corpus changed)")
        return 0 if same else 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fresh, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    print(f"wrote {OUT_PATH.relative_to(ROOT)}\n")
    print(f"RESOLVE ({len(fresh['resolve'])}) — brand grounds to a standalone record:")
    for brand, info in fresh["resolve"].items():
        print(f"  {brand:14s} → {info['generic']:18s} ({info['record']})")

    combos = {b: i for b, i in fresh["refuse_only"].items()
              if "combination" in i["reason"]}
    absent = {b: i for b, i in fresh["refuse_only"].items()
              if "combination" not in i["reason"]}

    print(f"\nREFUSE-ONLY · combination products ({len(combos)}) — "
          f"⚠️ resolving these would answer about a DIFFERENT medicine:")
    for brand, info in combos.items():
        print(f"  {brand:14s} → {info['generic']}")

    print(f"\nREFUSE-ONLY · not in corpus ({len(absent)}) — "
          f"named so the refusal can explain itself:")
    for brand, info in absent.items():
        print(f"  {brand:14s} → {info['generic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

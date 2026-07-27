"""
interactions.py — structured drug-drug interaction lookup (DDInter 2.0).

An interaction is a fact about a *pair* of drugs, not a passage to retrieve — so
this is a deterministic dictionary lookup, never RAG. That means: no missed pairs,
no hallucinated ones, and a clean "no documented interaction" vs "drug unknown"
distinction. RAG (rag_pipeline) is used only to *explain* an interaction we found.

Data: DDInter 2.0 (https://ddinter2.scbdd.com), CC BY-NC-SA 4.0 — 8 CSVs under
data/ddinter/. Each row: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level.

    db = InteractionDB()
    db.check_pair("warfarin", "naproxen")        # -> {'status':'ok','level':'Major',...}
    db.check_list(["warfarin", "naproxen", "metformin"])
"""
from __future__ import annotations

import csv
import difflib
import glob
import re
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
DDINTER_DIR = HERE / "data" / "ddinter"

# Severity ranking (higher = more dangerous). "Unknown" = documented interaction,
# severity not classified by DDInter — still surfaced, ranked lowest.
LEVEL_RANK = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
LEVEL_EMOJI = {"Major": "🔴", "Moderate": "🟠", "Minor": "🟡", "Unknown": "⚪"}

DDINTER_CITATION = ("DDInter 2.0 (Nucleic Acids Research 2025) — https://ddinter2.scbdd.com, "
                    "CC BY-NC-SA 4.0")

# Salt / dosage-form / connector words carry no drug identity — mirrors rag_pipeline.
_SALT_WORDS = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "sulfate", "sulphate", "succinate", "fumarate", "tartrate", "maleate",
    "mesylate", "besylate", "citrate", "phosphate", "acetate", "dipropionate",
    "bromide", "chloride", "hydrobromide", "monohydrate", "dihydrate", "hydrate",
    "and", "with", "for", "in", "of", "tablets", "tablet", "capsules", "capsule",
    "injection", "oral", "topical", "extended", "release", "cream", "ointment",
    "solution", "suspension", "gel", "spray", "er", "xr", "sr", "the", "a",
}
_ALPHA_TOKEN = re.compile(r"[a-z][a-z\-]{2,}")

# Common brand / lay names → DDInter's generic (it uses INN like "acetylsalicylic
# acid", not "aspirin"). Small, high-frequency map; RxNorm covers the long tail.
_SYNONYMS = {
    "aspirin": "acetylsalicylic acid", "asa": "acetylsalicylic acid",
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen", "panadol": "acetaminophen",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nurofen": "ibuprofen",
    "aleve": "naproxen", "coumadin": "warfarin", "jantoven": "warfarin",
    "glucophage": "metformin", "lipitor": "atorvastatin", "zocor": "simvastatin",
    "prilosec": "omeprazole", "nexium": "esomeprazole", "zoloft": "sertraline",
    "prozac": "fluoxetine", "xanax": "alprazolam", "ventolin": "salbutamol",
    "albuterol": "salbutamol",
}


def identity_tokens(name: str) -> frozenset:
    """Drug-identity tokens (salt/form words removed)."""
    return frozenset(t for t in _ALPHA_TOKEN.findall(name.lower()) if t not in _SALT_WORDS)


class InteractionDB:
    """In-memory DDInter lookup with tolerant drug-name resolution."""

    def __init__(self, ddinter_dir: Optional[str] = None, rxnorm=None):
        self.dir = Path(ddinter_dir) if ddinter_dir else DDINTER_DIR
        self.rxnorm = rxnorm                       # optional RxNormResolver (runtime)
        self.pairs: dict = {}                      # frozenset({keyA,keyB}) -> level (most severe)
        self._canon: set = set()                   # lowercased canonical drug names
        self._display: dict = {}                   # lower -> original-case display name
        self._tokens_of: dict = {}                 # lower -> frozenset identity tokens
        self._by_token: dict = {}                  # token -> set(lower names)
        self._resolve_cache: dict = {}
        self.n_rows = 0
        self._load()

    # ---- ingest ------------------------------------------------------------
    def _load(self):
        files = sorted(glob.glob(str(self.dir / "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No DDInter CSVs in {self.dir}. Download the 8 files from "
                f"https://ddinter2.scbdd.com/download/ into data/ddinter/.")
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    a, b, lvl = row.get("Drug_A", ""), row.get("Drug_B", ""), row.get("Level", "")
                    if not a or not b or a == b:
                        continue
                    self.n_rows += 1
                    ka, kb = a.lower().strip(), b.lower().strip()
                    self._register(ka, a)
                    self._register(kb, b)
                    key = frozenset((ka, kb))
                    # keep the most-severe level if a pair recurs across files
                    if LEVEL_RANK.get(lvl, 0) > LEVEL_RANK.get(self.pairs.get(key, "Unknown"), -1) \
                            or key not in self.pairs:
                        self.pairs[key] = lvl if lvl in LEVEL_RANK else "Unknown"

    def _register(self, key: str, display: str):
        if key in self._canon:
            return
        self._canon.add(key)
        self._display[key] = display
        toks = identity_tokens(display)
        self._tokens_of[key] = toks
        for t in toks:
            self._by_token.setdefault(t, set()).add(key)

    # ---- name resolution ---------------------------------------------------
    def resolve(self, name: str) -> Optional[str]:
        """Map an arbitrary drug name to a canonical DDInter key, or None."""
        if not name or not name.strip():
            return None
        raw = name.lower().strip()
        if raw in self._resolve_cache:
            return self._resolve_cache[raw]

        hit = self._resolve_offline(raw)
        # Optional runtime normalization (RxNorm) if offline matching failed.
        if hit is None and self.rxnorm is not None:
            std = self.rxnorm.standard_name(name)
            if std:
                hit = self._resolve_offline(std.lower().strip())
        self._resolve_cache[raw] = hit
        return hit

    def _resolve_offline(self, raw: str) -> Optional[str]:
        raw = _SYNONYMS.get(raw, raw)          # brand/lay name → generic
        if raw in self._canon:
            return raw
        toks = identity_tokens(raw)
        if not toks:
            return None
        # candidate names that contain ALL of the input's identity tokens
        cand = None
        for t in toks:
            s = self._by_token.get(t)
            if not s:
                cand = set()
                break
            cand = set(s) if cand is None else (cand & s)
        cand = cand or set()
        exact = [c for c in cand if self._tokens_of[c] == toks]
        if exact:
            return self._pick(exact)
        if cand:
            return self._pick(cand)
        # last resort: fuzzy on the full string (typos)
        m = difflib.get_close_matches(raw, self._canon, n=1, cutoff=0.9)
        return m[0] if m else None

    def _pick(self, keys) -> str:
        """Prefer the most generic (fewest tokens, then shortest) canonical name."""
        return sorted(keys, key=lambda k: (len(self._tokens_of[k]), len(k)))[0]

    # ---- lookups -----------------------------------------------------------
    def check_pair(self, a: str, b: str) -> dict:
        ca, cb = self.resolve(a), self.resolve(b)
        unresolved = [n for n, c in ((a, ca), (b, cb)) if c is None]
        if unresolved:
            return {"status": "unknown_drug", "unresolved": unresolved}
        if ca == cb:
            return {"status": "same_drug", "a": self._display[ca], "b": self._display[cb]}
        lvl = self.pairs.get(frozenset((ca, cb)))
        if lvl is None:
            return {"status": "no_interaction", "a": self._display[ca], "b": self._display[cb]}
        return {"status": "ok", "a": self._display[ca], "b": self._display[cb],
                "level": lvl, "rank": LEVEL_RANK[lvl], "emoji": LEVEL_EMOJI[lvl]}

    def check_list(self, names: list) -> dict:
        """Pairwise-check a medication list. Returns interactions (severity-sorted),
        the count of clean pairs, and any names we couldn't resolve."""
        clean_names = [n for n in (x.strip() for x in names) if n]
        resolved = {n: self.resolve(n) for n in clean_names}
        unresolved = sorted({n for n, c in resolved.items() if c is None})

        found, safe = [], 0
        known = [n for n in clean_names if resolved[n] is not None]
        for i in range(len(known)):
            for j in range(i + 1, len(known)):
                res = self.check_pair(known[i], known[j])
                if res["status"] == "ok":
                    res["input"] = (known[i], known[j])
                    found.append(res)
                elif res["status"] == "no_interaction":
                    safe += 1
        found.sort(key=lambda r: -r["rank"])
        summary = {lvl: sum(1 for r in found if r["level"] == lvl) for lvl in LEVEL_RANK}
        return {"interactions": found, "safe_pairs": safe, "unresolved": unresolved,
                "resolved": {n: self._display[c] for n, c in resolved.items() if c},
                "summary": summary, "n_drugs": len(clean_names)}


# ── CLI smoke test:  python interactions.py warfarin naproxen metformin ──────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    db = InteractionDB()
    print(f"Loaded {db.n_rows:,} rows, {len(db._canon):,} drugs, "
          f"{len(db.pairs):,} unique pairs.\n")
    drugs = sys.argv[1:] or ["warfarin", "naproxen", "metformin"]
    out = db.check_list(drugs)
    print("Drugs:", ", ".join(drugs))
    if out["unresolved"]:
        print("Unresolved:", out["unresolved"])
    print(f"\n{len(out['interactions'])} interaction(s), {out['safe_pairs']} clean pair(s):\n")
    for r in out["interactions"]:
        print(f"  {r['emoji']} {r['level']:8s} {r['a']} + {r['b']}")
    print(f"\nSource: {DDINTER_CITATION}")

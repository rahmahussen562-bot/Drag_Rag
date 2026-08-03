"""
retrieval/query_understanding.py — what the user meant, before anything is retrieved.

Runs BEFORE `Retriever.retrieve_context()`. Deterministic and cheap: no LLM call on
the hot path. Produces a `QueryPlan` describing what was understood, so the UI can
show *"I understood this as: …"* rather than silently rewriting the question.

# WHY must a rewrite never be silent?
# Because a rewritten query that goes wrong is indistinguishable, from the user's
# side, from a retrieval failure. If "and its side effects?" resolves to the wrong
# drug, the answer is confidently about a medicine they never asked about. Showing
# the resolved query makes that failure visible in one glance instead of invisible.

Phase 0 ships the data structure and the normalisation that is provably safe.
The pieces that would CHANGE retrieval results — lay-term expansion, coreference,
multi-drug decomposition, intent distribution — are declared here and implemented
in Phase 1, where each one is measured against the ground-truth set before it is
allowed to affect a ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

#: Query intents the field router (see field_router.py) maps onto label fields.
INTENTS = (
    "dosage", "side_effects", "interactions", "contraindications", "mechanism",
    "indications", "pregnancy_lactation", "storage_handling", "overdose",
    "pharmacokinetics", "symptom_triage", "out_of_scope",
)

# Clinical shorthand a practitioner types and a corpus never contains.
# # WHY a table rather than an LLM call? Determinism. This runs on every query, and
# an expansion that varies run to run makes an evaluation set meaningless.
ABBREVIATIONS = {
    "sob": "shortness of breath", "htn": "hypertension", "t2dm": "type 2 diabetes",
    "t1dm": "type 1 diabetes", "ckd": "chronic kidney disease", "gi": "gastrointestinal",
    "bid": "twice daily", "tid": "three times daily", "qid": "four times daily",
    "qd": "once daily", "prn": "as needed", "po": "by mouth", "iv": "intravenous",
    "hf": "heart failure", "mi": "myocardial infarction", "afib": "atrial fibrillation",
    "copd": "chronic obstructive pulmonary disease", "uti": "urinary tract infection",
}

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z][a-z\-]*")


@dataclass
class QueryPlan:
    """What the system understood, and why — the object the UI explains from."""
    raw: str
    normalised: str
    entities: set = field(default_factory=set)      # grounded corpus entities
    unknown: list = field(default_factory=list)     # named but not in the corpus
    intents: dict = field(default_factory=dict)     # intent → confidence (a distribution)
    expansions: list = field(default_factory=list)  # BM25-only extra terms
    rewrite_reason: str = ""                        # shown as "I understood this as…"
    is_multi_drug: bool = False

    @property
    def rewritten(self) -> str:
        """The query actually issued. Identical to `normalised` unless rewritten."""
        return self.normalised

    @property
    def top_intent(self) -> Optional[str]:
        return max(self.intents, key=self.intents.get) if self.intents else None


def normalise(query: str) -> str:
    """Lowercase, collapse whitespace, expand clinical abbreviations.

    Deliberately conservative: it does NOT strip punctuation or numbers. A dose is
    punctuation-and-digits ("500 mg/day"), and stage 02 already documents why
    destroying those is a safety bug rather than a tuning choice.
    """
    text = _WS.sub(" ", str(query or "")).strip().lower()
    if not text:
        return ""
    out = []
    for token in text.split(" "):
        bare = token.strip(".,;:?!()")
        expansion = ABBREVIATIONS.get(bare)
        out.append(token.replace(bare, expansion) if expansion else token)
    return " ".join(out)


def plan(query: str, retriever=None) -> QueryPlan:
    """Build a QueryPlan. Grounding is delegated to stage 06 — never reimplemented.

    Phase 0 behaviour: normalise, then ask the retriever what it grounds. That is
    exactly what `retrieve_context` already does internally, so a plan built here
    describes the retrieval that is about to happen without altering it.
    """
    normalised = normalise(query)
    entities: set = set()
    unknown: list = []

    if retriever is not None:
        # Ground against the RAW query: stage 06 owns entity resolution, and its
        # vocabulary was built from the corpus. Passing the normalised string could
        # only lose information here, since abbreviations are never drug names.
        ground = retriever.ground(query)
        entities = set(ground["matched"])
        unknown = list(ground["unknown"])

    return QueryPlan(
        raw=query,
        normalised=normalised,
        entities=entities,
        unknown=unknown,
        is_multi_drug=len(entities) > 1,
    )

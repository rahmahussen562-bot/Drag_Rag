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


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
#
# # WHY rule-based and not a model?
# It runs on every query, so it must be deterministic: an intent that varies run
# to run makes an evaluation set meaningless and makes a ranking irreproducible.
# It is also trivially auditable — a buyer can read why a query was routed, which
# is not true of a classifier head.
#
# # WHY a DISTRIBUTION rather than a label?
# "can I take this at night?" is arguably dosage, storage or side_effects. Forcing
# a single label bets the ranking on that being resolved correctly. A distribution
# lets the field router pull on several sections proportionally, so a
# misclassification costs ranking position instead of making the right chunk
# unreachable.
#
# Patterns are ordered most-specific first within each intent. Weights are
# relative, not probabilities — they are normalised at the end.
# ─────────────────────────────────────────────────────────────────────────────
_INTENT_PATTERNS: list = [
    # mechanism — note "what does X do" lands HERE, not on indications. That is
    # what the ground-truth set says a clinician means by it, and the classifier
    # follows the data rather than the phrasing's surface reading.
    ("mechanism", 3.0, r"\bmechanism(?:\s+of\s+action)?\b|\bmoa\b"),
    ("mechanism", 2.5, r"\bhow\s+(?:does|do|is)\b.*\b(?:work|works|act|acts|lower|lowers|reduce|reduces)\b"),
    ("mechanism", 2.0, r"\bwhat\s+does\b.*\bdo\b"),
    ("mechanism", 1.5, r"\bpharmacodynamic|\bacts?\s+on\b|\bmode\s+of\s+action\b"),

    ("indications", 3.0, r"\bindication(?:s|ed)?\b"),
    ("indications", 2.5, r"\bused?\s+(?:for|to\s+treat|in)\b|\bprescribed\s+for\b"),
    ("indications", 2.5, r"\buses\b|\bwhat\s+is\b.*\bfor\b"),
    ("indications", 1.5, r"\btreats?\b|\btreatment\s+of\b"),

    ("dosage", 3.0, r"\bdosage\b|\bdosing\b|\bdose[sd]?\b"),
    ("dosage", 2.0, r"\bhow\s+(?:much|many|often)\b|\btitrat\w+"),
    ("dosage", 1.5, r"\b\d+\s*(?:mg|mcg|g|ml|units?)\b|\bbid\b|\btid\b|\bqd\b"),

    ("side_effects", 3.0, r"\bside[-\s]?effects?\b|\badverse\b"),
    ("side_effects", 2.0, r"\breactions?\b|\btoxicit\w+"),
    ("side_effects", 1.5, r"\bcauses?\b.*\b(?:nausea|dizziness|rash|headache)\b"),

    ("interactions", 3.0, r"\binteract\w*\b"),
    ("interactions", 2.0, r"\btake\s+(?:with|together)\b|\bcombin\w+|\bconcurrent\w*"),
    ("interactions", 1.5, r"\balongside\b|\bco-?administ\w+"),

    ("contraindications", 3.0, r"\bcontraindicat\w+"),
    ("contraindications", 2.0, r"\bshould\s+not\s+(?:take|be)\b|\bwho\s+(?:cannot|can't|should not)\b"),
    ("contraindications", 1.5, r"\bavoid\b|\bnot\s+recommended\b"),

    ("pregnancy_lactation", 3.0, r"\bpregnan\w+|\bbreast[-\s]?feed\w*|\blactation\b|\bnursing\b"),
    ("storage_handling", 3.0, r"\bstor(?:e|age|ing)\b|\brefrigerat\w+|\bshelf\s+life\b"),
    ("overdose", 3.0, r"\boverdose\b|\btoo\s+much\b|\bexceed\w*\b"),
    ("pharmacokinetics", 3.0, r"\bhalf[-\s]?life\b|\bpharmacokinetic\w*|\bcmax\b|\bclearance\b"),
    ("pharmacokinetics", 2.0, r"\babsorption\b|\bmetaboli[sz]\w+|\bexcret\w+"),
    ("symptom_triage", 2.5, r"\bi\s+(?:have|feel|am\s+having|got)\b|\bmy\s+\w+\s+(?:hurts|aches)\b"),
]

_COMPILED = [(intent, weight, re.compile(pattern, re.IGNORECASE))
             for intent, weight, pattern in _INTENT_PATTERNS]


def classify_intents(query: str) -> dict:
    """Score a query against every intent. Returns a normalised distribution.

    An empty dict means nothing matched — the caller should apply no field
    preference at all rather than guessing, because a wrong boost is worse than
    none: it actively demotes the section that would have answered.
    """
    scores: dict = {}
    text = str(query or "")
    for intent, weight, pattern in _COMPILED:
        if pattern.search(text):
            scores[intent] = scores.get(intent, 0.0) + weight
    total = sum(scores.values())
    if not total:
        return {}
    return {k: round(v / total, 4) for k, v in
            sorted(scores.items(), key=lambda kv: -kv[1])}


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
        intents=classify_intents(query),
        is_multi_drug=len(entities) > 1,
    )

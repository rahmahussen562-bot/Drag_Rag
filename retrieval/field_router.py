"""
retrieval/field_router.py — steer retrieval toward the right LABEL SECTION.

FDA labels are already field-structured. Today stage 06 restricts by DRUG and then
hopes the ranker surfaces the right FIELD; measured field recall@5 is 91% against
drug recall@5 of 100%. Every one of those misses is the right medicine, wrong
section — so the field is the remaining accuracy gap.

# WHY a soft BOOST and not a hard filter?
# A filter is a bet that the intent classifier is right. It is not always right:
# "can I take this at night?" is arguably dosage, storage or side_effects. A
# multiplicative boost lets an unusual phrasing still surface another field when
# the evidence is strong, so a misclassification costs ranking position rather
# than making the correct chunk unreachable.
#
# Applied AFTER the drug restriction and BEFORE the relevance floor: boosting must
# never resurrect a chunk the floor would have rejected, or the boost would quietly
# weaken abstention — the one behaviour this project refuses to trade away.

FIELD_BOOST is 0.0 in Phase 0, which makes this module a NO-OP by construction.
Phase 1 sweeps it in eval/tune_field_boost.py exactly as tune_alpha.py sweeps
alpha, and adopts a non-zero value only if field recall improves while drug recall
stays at 100%. If the sweep says it hurts, it stays 0.0 and the negative result
goes in the README.
"""
from __future__ import annotations

#: Phase 0: no-op. Phase 1 sets this from a measured sweep.
FIELD_BOOST = 0.0

#: intent -> [(label field, relative weight)], most relevant first.
INTENT_TO_FIELDS = {
    "dosage":              [("dosage", 1.0), ("indications", 0.3)],
    "side_effects":        [("side_effects", 1.0), ("contraindications", 0.4)],
    "interactions":        [("interactions", 1.0), ("contraindications", 0.3)],
    "contraindications":   [("contraindications", 1.0), ("side_effects", 0.4)],
    "mechanism":           [("mechanism", 1.0), ("description", 0.4)],
    "indications":         [("indications", 1.0), ("description", 0.3)],
    "pregnancy_lactation": [("contraindications", 1.0), ("side_effects", 0.4)],
    "storage_handling":    [("description", 1.0)],
    "overdose":            [("side_effects", 1.0), ("dosage", 0.5)],
    "pharmacokinetics":    [("mechanism", 1.0), ("description", 0.5)],
    "symptom_triage":      [("side_effects", 1.0), ("indications", 0.4)],
    "out_of_scope":        [],
}


def field_weights(intents: dict) -> dict:
    """Blend per-intent field preferences into one {field: weight} map.

    `intents` is a distribution, not a label, so a query that is 60% dosage and
    40% interactions pulls on both sections proportionally instead of committing
    to whichever scored marginally higher.
    """
    weights: dict = {}
    for intent, confidence in (intents or {}).items():
        for fld, w in INTENT_TO_FIELDS.get(intent, []):
            weights[fld] = weights.get(fld, 0.0) + confidence * w
    return weights


def boost_multipliers(fields, intents: dict, field_boost: float = FIELD_BOOST):
    """Per-candidate multiplier for a sequence of field names.

    Returns 1.0 everywhere when `field_boost` is 0.0, which is why Phase 0 can ship
    this module without changing a single ranking.
    """
    fields = list(fields)
    if not field_boost or not intents:
        return [1.0] * len(fields)
    weights = field_weights(intents)
    top = max(weights.values()) if weights else 0.0
    if top <= 0:
        return [1.0] * len(fields)
    return [1.0 + field_boost * (weights.get(f, 0.0) / top) for f in fields]

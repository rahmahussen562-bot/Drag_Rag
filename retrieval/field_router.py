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

# ═══════════════════════════════════════════════════════════════════════════
# MEASURED, AND THEN CHECKED FOR SELF-DECEPTION
# ═══════════════════════════════════════════════════════════════════════════
# `eval/tune_field_boost.py` swept FIELD_BOOST over the ground-truth set:
#
#     BOOST   drug R@5   field R@5   field MRR
#      0.00      100%        91%       0.822
#      0.20      100%       100%       1.000     ← "perfect"
#
# That result is NOT the one to quote, and the giveaway is the MRR of exactly
# 1.000. The intent classifier in query_understanding.py was written while looking
# at eval_set.json, so scoring it there measures memorisation as much as skill.
#
# `eval/run_heldout.py` re-runs it over 19 alternative phrasings of the same
# information needs — the honest number:
#
#                    drug R@5   field R@5   field MRR
#      no routing       100%        84%       0.684
#      boost 0.20       100%        95%       0.789
#
# So routing DOES generalise (+11 points of field recall on unseen phrasings) but
# it is worth far less than the tuning set claimed. 0.20 is adopted on the
# strength of the held-out number, not the flattering one.
#
# # THE REAL LIMITATION, recorded rather than hidden:
# intent accuracy on held-out phrasings is only 6/19 (32%) — 13 of them match no
# pattern at all. Routing still helps because the classifier FAILS SAFE: no match
# produces an empty distribution, which produces a multiplier of exactly 1.0, so an
# unrecognised query is ranked as though this module did not exist. Low coverage
# therefore costs opportunity, never correctness. Widening the patterns (or moving
# to an embedding-based classifier) is the highest-value next step, and it should
# be measured on run_heldout.py, not on the set it was tuned against.
"""
from __future__ import annotations

#: Adopted from eval/run_heldout.py (+11 pts field recall on unseen phrasings at
#: unchanged 100% drug recall), NOT from the tuning set's flattering 100%.
#: Re-derive with: python eval/tune_field_boost.py && python eval/run_heldout.py
FIELD_BOOST = 0.20

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

"""
agents/patient.py — the lay persona.

Phase 2 target: k=4, smaller budget, mmr_lambda=0.6 (patients ask broad questions),
lay→clinical expansion into the BM25 query only, neighbour expansion on, safety
fields packed first, and a red-flag triage screen that BYPASSES RAG entirely for
emergency symptoms and returns a fixed, non-generated urgent-care message.

# WHY block pharmacokinetics / clinical_pharmacology / mechanism?
# Not because they are secret — they are public label text. Because they produce
# answers a layperson will misread. "Cmax is reached in 2.5 hours" invites someone
# to reason about their own timing and dose, which is precisely the decision this
# product refuses to help anyone make.
#
# # WHY does red-flag triage bypass RAG instead of prompting carefully?
# Because an LLM asked to handle chest pain will improvise, and the failure mode is
# somebody not calling an ambulance. A fixed string cannot improvise. This is the
# one path in the system where generation is not merely constrained but removed.

Phase 0 registers the persona with today's behaviour so nothing changes yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stages import load  # noqa: E402

from agents.base import AgentConfig, register  # noqa: E402

prompting = load("07_prompting")

#: Symptoms that must never reach the RAG path. Phase 2 wires the screen in.
RED_FLAGS = (
    "chest pain", "difficulty breathing", "trouble breathing", "can't breathe",
    "swelling of the face", "swelling of my face", "swollen lips", "swollen tongue",
    "suicidal", "kill myself", "severe bleeding", "won't stop bleeding",
    "seizure", "convulsion", "unconscious", "passed out", "anaphylaxis",
    "stevens-johnson", "blistering rash",
)

ESCALATION_NOTICE = (
    "If you are worried about your own symptoms right now, contact your doctor, "
    "your pharmacist, or your local emergency number."
)

CONFIG = AgentConfig(
    id="patient",
    label="Patient",
    description="Plain language, safety-first, never advises a dose change.",
    prompt_path="patient.md",

    # Phase 0 mirrors today's numbers; Phase 2 drops k to 4 and justifies it.
    k=5,
    context_token_budget=1800,
    alpha=0.20,
    use_reranker=False,
    allowed_fields=frozenset({
        "indications", "side_effects", "contraindications", "description",
    }),
    blocked_fields=frozenset({"mechanism"}),
    field_boost=0.0,
    mmr_lambda=0.6,                   # broader questions → more diversity
    expand_neighbours=False,          # Phase 1 turns this on, once measured
    lay_expansion=False,              # Phase 1 adds the synonym table

    temperature=0.0,
    reading_level="plain",
    max_answer_words=180,
    require_escalation_notice=True,
    output_schema=(),
)


def is_red_flag(query: str) -> bool:
    """True when the query describes an emergency symptom."""
    q = str(query or "").lower()
    return any(flag in q for flag in RED_FLAGS)


class PatientAgent:
    """Phase 0: a thin, behaviour-preserving delegate to stage 07."""

    config = CONFIG

    def answer(self, query: str, retriever, k: int = None, use_llm: bool = True,
               prefer: str = None, on_stage=None):
        return prompting.answer_question(
            query, retriever, k=k or self.config.k, use_llm=use_llm,
            prefer=prefer, temperature=self.config.temperature, on_stage=on_stage)


register(CONFIG, PatientAgent)

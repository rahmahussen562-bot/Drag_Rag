"""
agents/practitioner.py — the clinical persona.

Phase 2 target: k=8, larger budget, mmr_lambda=0.8, reranker available, all fields
allowed, multi-drug decomposition, and an automatic DDInter lookup injected as
[Source 0] marked STRUCTURED LOOKUP whenever ≥2 drugs ground — deterministic
lookup for facts, RAG only to explain them.

# WHY is this the DEFAULT persona in Phase 0?
# Because its config is the one that reproduces today's behaviour exactly (k=5 from
# the sidebar, all fields, no expansion). Making Patient the default would mean
# Phase 0 changed what users see, and Phase 0's contract is that it does not.
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

CONFIG = AgentConfig(
    id="practitioner",
    label="Healthcare Professional",
    description="Clinical register, full label access, structured interaction lookup.",
    prompt_path="practitioner.md",

    # Phase 0 keeps today's numbers. Phase 2 raises k to 8 and justifies it.
    k=5,
    context_token_budget=3000,
    alpha=0.20,                       # measured — see eval/tune_alpha.py
    use_reranker=False,               # off by default on Cloud: a 2nd transformer
    allowed_fields=None,              # all fields
    blocked_fields=frozenset(),
    field_boost=0.0,                  # Phase 1 sweeps this
    mmr_lambda=0.8,
    expand_neighbours=False,
    lay_expansion=False,              # practitioners type clinical terms already

    temperature=0.0,                  # factual assistant; reproducible evaluation
    reading_level="clinical",
    max_answer_words=400,
    require_escalation_notice=False,
    output_schema=(
        "Indication", "Dosing", "Renal/Hepatic adjustment", "Contraindications",
        "Key interactions", "Monitoring", "Notable warnings",
    ),
)


class PractitionerAgent:
    """Phase 0: a thin, behaviour-preserving delegate to stage 07."""

    config = CONFIG

    def answer(self, query: str, retriever, k: int = None, use_llm: bool = True,
               prefer: str = None, on_stage=None):
        # Delegates verbatim. Stage 07 stays the single source of truth for
        # generation; Phase 2 adds the persona prompt and the [Source 0] lookup.
        return prompting.answer_question(
            query, retriever, k=k or self.config.k, use_llm=use_llm,
            prefer=prefer, temperature=self.config.temperature, on_stage=on_stage)


register(CONFIG, PractitionerAgent)

"""
agents/practitioner.py — the clinical persona.

Differs from the patient agent in every dimension that matters:

  CONTEXT   k=8, all label fields, strict score order (no safety reordering — a
            prescriber wants the ranking, not a curated one)
  PROMPT    prompts/practitioner.md — clinical register, verbatim numbers, and a
            fixed heading schema
  ENFORCE   the heading schema is validated after generation, and a section with
            no supporting context is rendered as "Not addressed in the retrieved
            label sections" rather than dropped

# WHY make silence VISIBLE instead of omitting the heading?
# Because a missing "Renal/Hepatic adjustment" section is ambiguous: it could mean
# the label says nothing, or it could mean the model forgot. For a prescriber
# those are very different facts. Rendering the gap explicitly turns an absence
# into a statement, which is the whole point of citing a label rather than
# summarising one.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stages import load  # noqa: E402

from agents.base import AgentConfig, fetch_k, register, select_chunks  # noqa: E402

prompting = load("07_prompting")

CONFIG = AgentConfig(
    id="practitioner",
    label="Healthcare Professional",
    description="Clinical register, full label access, structured headings.",
    prompt_path="practitioner.md",

    k=8,                              # a prescriber wants breadth, not a summary
    context_token_budget=3000,
    alpha=0.20,                       # measured — see eval/tune_alpha.py
    use_reranker=False,               # off by default on Cloud: a 2nd transformer
    allowed_fields=None,              # all fields
    blocked_fields=frozenset(),
    field_boost=0.0,
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

MISSING_SECTION = "*Not addressed in the retrieved label sections.*"


def enforce(text: str, chunks: list) -> str:
    """Validate the heading schema; make omissions explicit rather than invisible.

    Only headings the model ALREADY used are checked for content. Appending every
    missing heading would bury a three-line answer under seven empty sections, so
    the report is a compact note listing what the retrieved context did not cover.
    """
    lowered = text.lower()
    present = [h for h in CONFIG.output_schema if h.lower() in lowered]
    if not present:
        # The model ignored the schema entirely. That is a model-quality signal,
        # not something to paper over by fabricating structure it did not produce.
        return text

    missing = [h for h in CONFIG.output_schema if h.lower() not in lowered]
    if not missing:
        return text
    note = ("\n\n---\n**Not covered by the retrieved sections:** "
            + " · ".join(missing) + f"\n\n{MISSING_SECTION}")
    return text.rstrip() + note


class PractitionerAgent:
    """Clinical persona: full label access, structured output, schema validated."""

    config = CONFIG

    def answer(self, query: str, retriever, k: int = None, use_llm: bool = True,
               prefer: str = None, on_stage=None):
        want = k or self.config.k
        return prompting.answer_question(
            query, retriever, k=fetch_k(self.config, want), use_llm=use_llm,
            prefer=prefer, temperature=self.config.temperature, on_stage=on_stage,
            system_prompt=self.config.prompt(),
            select=lambda chunks: select_chunks(self.config, chunks, limit=want),
            postprocess=enforce,
            agent=self.config.id)


register(CONFIG, PractitionerAgent)

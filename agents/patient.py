"""
agents/patient.py — the lay persona.

Genuinely different from the practitioner in three ways, in order of how much
they matter:

  1. CONTEXT BUILDING  a narrower field allowlist and safety-first packing order,
     so the reader physically cannot be shown a pharmacokinetics paragraph.
  2. PROMPT            prompts/patient.md — eight numbered imperatives, including
     the two that do not exist for a practitioner: never direct a medication
     change, never interpret personal symptoms.
  3. ENFORCEMENT       the prompt is a request; `postprocess` is a check.

# WHY block mechanism / pharmacokinetics for a patient?
# Not because the text is secret — it is public label text. Because it produces
# answers a layperson will misread. "Cmax is reached in 2.5 hours" invites someone
# to reason about their own timing and dose, which is exactly the decision this
# product refuses to help anyone make.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stages import load  # noqa: E402

from agents.base import AgentConfig, fetch_k, register, select_chunks  # noqa: E402

prompting = load("07_prompting")

# ─────────────────────────────────────────────────────────────────────────────
# Red-flag triage
#
# # WHY does this bypass RAG ENTIRELY rather than prompting carefully?
# Because an LLM asked to handle chest pain will improvise, and the failure mode
# is somebody not calling an ambulance. A fixed string cannot improvise. This is
# the one path in the system where generation is not merely constrained but
# REMOVED — no retrieval, no model, no variation between runs.
# ─────────────────────────────────────────────────────────────────────────────
RED_FLAGS = (
    "chest pain", "chest pains", "difficulty breathing", "trouble breathing",
    "can't breathe", "cannot breathe", "struggling to breathe",
    "swelling of the face", "swelling of my face", "face is swelling",
    "swollen lips", "swollen tongue", "throat closing",
    "suicidal", "kill myself", "end my life",
    "severe bleeding", "won't stop bleeding", "wont stop bleeding",
    "seizure", "convulsion", "fitting",
    "unconscious", "passed out", "collapsed",
    "anaphylaxis", "anaphylactic", "stevens-johnson", "blistering rash",
)

URGENT_MESSAGE = (
    "### 🚑 This may be a medical emergency\n\n"
    "The symptoms you have described can be serious and need to be assessed by a "
    "person, not by software.\n\n"
    "**Call your local emergency number now, or go to your nearest emergency "
    "department.** If you are not sure, call your doctor or pharmacist "
    "straight away.\n\n"
    "I have not searched the drug database for this question, because a symptom "
    "like this should not wait for a search result."
)

ESCALATION_NOTICE = (
    "**Who to ask:** your doctor or pharmacist knows your full medical history "
    "and can tell you what this means for you personally."
)

# ─────────────────────────────────────────────────────────────────────────────
# Medication-change screen
#
# # WHY refuse these outright rather than answer them carefully?
# "Should I increase my dose to 1000 mg?" is a request for a DECISION, not for
# information. Answering it from label text — even hedged, even cited — produces
# something a worried person can read as permission. Prompt rule 5 forbids
# directing a change; this is that rule enforced before a model ever sees the
# question, because a rule the model must remember is weaker than a branch.
#
# # WHY is this a SEPARATE screen from the field allowlist?
# An earlier version of this agent appeared to refuse titration questions, but only
# because the field filter had starved its context — remove the bug and the refusal
# vanished. A safety property that depends on another component under-performing is
# not a safety property. This screen refuses on INTENT, so it holds regardless of
# what retrieval returns.
# ─────────────────────────────────────────────────────────────────────────────
_MEDICATION_CHANGE = re.compile(
    r"\b(?:should|shall|can|could|may|ought)\s+(?:i|we|my|he|she|they)\b"
    r"[^.?!]*\b(?:take|takes|taking|start|starting|stop|stopping|increase|"
    r"increasing|decrease|decreasing|reduce|reducing|double|doubling|halve|"
    r"halving|switch|switching|combine|combining|skip|skipping|add)\b"
    r"|\b(?:is it (?:ok|okay|safe|alright))\b[^.?!]*\b(?:to )?(?:take|stop|start|"
    r"increase|double|combine|skip|mix)\b"
    r"|\b(?:can|should) i (?:just )?(?:stop|quit|come off)\b",
    re.IGNORECASE)

MEDICATION_CHANGE_MESSAGE = (
    "I can't help you decide whether to change how you take a medicine — that "
    "decision depends on your medical history, your other medicines and your test "
    "results, none of which I can see.\n\n"
    "**Please ask your doctor or pharmacist.** They can answer this properly, and "
    "it is exactly the kind of question they expect.\n\n"
    "I *can* tell you what the official label says about this medicine — its uses, "
    "its side effects, and its warnings. Ask me about any of those and I'll quote "
    "the label and cite it."
)


def is_medication_change(query: str) -> bool:
    """True when the query asks whether to start/stop/change/combine a medicine."""
    return bool(_MEDICATION_CHANGE.search(str(query or "")))

# Imperative dosing verbs aimed at the reader. Matching these does not prove the
# answer is unsafe — it proves it is PHRASED as an instruction, which for this
# persona is the thing to catch regardless of intent.
_DOSING_IMPERATIVE = re.compile(
    r"\b(?:you should|you must|you can|you may)\s+"
    r"(?:take|start|stop|increase|decrease|double|halve|switch|combine)\b"
    r"|^\s*(?:take|start|stop|increase|decrease|double|halve)\b",
    re.IGNORECASE | re.MULTILINE)

ADVISORY_FRAMING = (
    "\n\n> ⚠️ **This is what the label says — not advice for you.** Any change to "
    "how you take a medicine has to be decided with your doctor or pharmacist."
)

CONFIG = AgentConfig(
    id="patient",
    label="Patient",
    description="Plain language, safety-first. Never advises a dose change.",
    prompt_path="patient.md",

    k=4,                              # patients ask broad questions; fewer, clearer
    context_token_budget=1800,
    alpha=0.20,
    use_reranker=False,
    # The label sections a layperson can read without being misled.
    allowed_fields=frozenset({
        "indications", "side_effects", "contraindications", "description",
    }),
    # Hard deny, applied before the allowlist. Explicit rather than merely absent,
    # so the intent survives someone widening allowed_fields later.
    blocked_fields=frozenset({"mechanism", "pharmacokinetics",
                              "clinical_pharmacology"}),
    field_boost=0.0,
    mmr_lambda=0.6,
    expand_neighbours=False,
    lay_expansion=False,

    temperature=0.0,
    reading_level="plain",
    max_answer_words=180,
    require_escalation_notice=True,
    output_schema=(),
)


def is_red_flag(query: str) -> bool:
    """True when the query describes an emergency symptom."""
    q = " " + re.sub(r"\s+", " ", str(query or "").lower()) + " "
    return any(flag in q for flag in RED_FLAGS)


def enforce(text: str, chunks: list) -> str:
    """Enforcement in code, mirroring how `ensure_disclaimer` already works.

    A prompt rule is a request. These are the two that must not be optional for a
    lay reader, so they are applied to the returned text regardless of whether the
    model complied.
    """
    # An answer phrased as an instruction gets re-framed rather than suppressed —
    # the label content is still useful, the imperative framing is what is wrong.
    if _DOSING_IMPERATIVE.search(text):
        text = text.rstrip() + ADVISORY_FRAMING

    if CONFIG.require_escalation_notice and "doctor or pharmacist" not in text.lower():
        text = text.rstrip() + "\n\n" + ESCALATION_NOTICE
    return text


class PatientAgent:
    """Lay persona: narrow context, plain prompt, enforced safety framing."""

    config = CONFIG

    def answer(self, query: str, retriever, k: int = None, use_llm: bool = True,
               prefer: str = None, on_stage=None):
        # Triage first — before retrieval, before any model call.
        if is_red_flag(query):
            if on_stage:
                on_stage("refused", "red-flag triage")
            return prompting.Answer(
                text=URGENT_MESSAGE, chunks=[], mode="refused",
                status="red_flag", agent=self.config.id)

        # Then the medication-change screen. Also pre-retrieval: there is no
        # context that would make "should I double my dose?" answerable here.
        if is_medication_change(query):
            if on_stage:
                on_stage("refused", "medication-change request")
            return prompting.Answer(
                text=MEDICATION_CHANGE_MESSAGE, chunks=[], mode="refused",
                status="medication_change", agent=self.config.id)

        # Over-retrieve, then let the field policy trim back to `want`. Asking for
        # exactly `want` and filtering afterwards starves the context (see
        # agents.base.OVERFETCH).
        want = k or self.config.k
        return prompting.answer_question(
            query, retriever, k=fetch_k(self.config, want), use_llm=use_llm,
            prefer=prefer, temperature=self.config.temperature, on_stage=on_stage,
            system_prompt=self.config.prompt(),
            select=lambda chunks: select_chunks(self.config, chunks, limit=want),
            postprocess=enforce,
            agent=self.config.id)


register(CONFIG, PatientAgent)

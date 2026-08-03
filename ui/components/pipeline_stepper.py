"""
ui/components/pipeline_stepper.py — live per-stage progress while an answer runs.

# WHY show the stages at all instead of one spinner?
# Retrieval and generation have very different latencies (~0.1 s vs 3-20 s).
# Naming the slow one is what makes the wait feel accounted for rather than
# broken, and it doubles as a demonstration: the buyer watches the pipeline
# actually execute, stage by stage.
#
# # WHY is this driven by a CALLBACK from stage 07 rather than by the UI?
# Because splitting `answer_question` apart in the UI would move orchestration —
# the abstention rule, the packing order, the fallback policy — into the
# presentation layer, where it could drift from what the CLI and the eval harness
# run. `on_stage` lets the UI *observe* the pipeline without owning any part of it,
# so every caller still executes byte-identical logic.
"""
from __future__ import annotations

#: Pipeline stage name → what the user sees. Stage names are emitted by
#: `07_prompting.answer_question` via its `on_stage` callback.
STAGE_LABELS = {
    "retrieving": "🔍 Retrieving relevant sources…",
    "building":   "🧩 Building the grounded prompt…",
    "generating": "✍️ Generating answer…",
    "verifying":  "🔎 Verifying citations…",
    "refused":    "🛑 No supporting sources found",
}

FIRST_LABEL = STAGE_LABELS["retrieving"]


def make_on_stage(box):
    """Return an `on_stage(name, detail)` callback bound to a `st.status` box."""
    def on_stage(name: str, detail: str = "") -> None:
        label = STAGE_LABELS.get(name, name)
        if detail:
            label = f"{label}  ·  {detail}"
        box.update(label=label)
    return on_stage


def finalise(box, result) -> None:
    """Set the terminal label on the status box once an answer exists.

    A refusal is NOT an error state: it is the system working correctly, so it
    completes rather than showing a red box. An LLM failure IS an error, because a
    provider was configured and did not work.
    """
    if result.refused:
        box.update(label="🛑 No supporting sources — answer withheld",
                   state="complete", expanded=False)
    elif result.llm_failed:
        box.update(label="⚠️ LLM unavailable — showing retrieved sources",
                   state="error", expanded=False)
    else:
        box.update(label="✅ Answer complete", state="complete", expanded=False)

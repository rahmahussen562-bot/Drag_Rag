"""
ui/components/trust_panel.py — everything that lets a user check the answer.

Two levels of disclosure, deliberately separated:

  `answer_badges`    — always on. Provider, citations, groundedness at a glance.
  `render_evaluation`— 🔬 evaluation mode. Every intermediate the pipeline produced.

# WHY does a product ship a debug panel?
# Because the project's central claim is "the answer comes from the retrieved
# context". This panel is the evidence: the exact chunks, their component scores,
# the verbatim prompt, and the mechanical verification of the answer. Without it,
# groundedness is a promise; with it, it is auditable.
"""
from __future__ import annotations

import streamlit as st

from ui.theme import pill, pills


def answer_badges(result) -> str:
    """The meta strip under an answer — provenance at a glance.

    Returns an HTML string (caller decides where to place it), or "" when the
    answer carries nothing worth badging.
    """
    # The persona is provenance too — the same question answered as Patient and as
    # Practitioner is two different products, so the badge says which one ran.
    lead = [pill(f"👤 {result.agent}", dot=True)] if result.agent else []

    # A field the persona's policy removed is a deliberate omission, and the user
    # is entitled to know a section was withheld rather than simply absent.
    if result.dropped_fields:
        lead.append(pill(f"🚫 withheld: {', '.join(result.dropped_fields)}", "warn"))

    if result.mode == "llm":
        badges = [pill(f"⚡ {result.provider}", "ok", dot=True), pill(result.model)]
        v = result.verification
        if v:
            badges.append(pill(f"📎 {len(v.citations)}/{v.n_sources} cited",
                               "ok" if v.citations else "warn"))
            badges.append(pill(f"🎯 {v.overlap:.0%} grounded",
                               "ok" if v.overlap >= 0.3 else "warn"))
        return pills(*lead, *badges)

    if result.status == "red_flag":
        return pills(*lead, pill("🚑 Emergency triage — RAG bypassed", "err"))
    if result.status == "medication_change":
        return pills(*lead, pill("🛑 Medication-change request — declined", "warn"))
    if result.status == "persona_field_policy":
        return pills(*lead, pill("🛑 Withheld — outside this persona's sections", "warn"))
    if result.refused:
        return pills(*lead, pill("🛑 Withheld — not in corpus", "warn"))
    if result.mode == "extractive":
        return pills(*lead, pill("📄 Retrieved text (no LLM)", "warn"))
    return pills(*lead) if lead else ""


def render_evaluation(result) -> None:
    """Evaluation mode — expose every intermediate the pipeline produced."""
    st.divider()
    st.markdown("### 🔬 Evaluation mode")

    verification = result.verification
    if verification:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sources cited", f"{len(verification.citations)}/{verification.n_sources}")
        c2.metric("Context overlap", f"{verification.overlap:.0%}")
        c3.metric("Disclaimer", "yes" if verification.has_disclaimer else "NO")
        c4.metric("Grounded", "yes" if verification.grounded else "no")
        if verification.invalid_citations:
            st.error(f"⚠️ The answer cited sources that do not exist: "
                     f"{verification.invalid_citations}. That is a fabricated citation.")
        elif result.mode == "llm" and not verification.citations:
            st.warning("The answer cited no sources. For a refusal that is correct; "
                       "for a factual claim it means the answer is not traceable.")

    st.markdown(f"**Generation:** `{result.mode}`"
                + (f" · provider `{result.provider}` · model `{result.model}`"
                   if result.provider else ""))
    if result.llm_failed:
        st.error(f"Provider error: `{result.llm_error}`", icon="🚨")
    if result.attempts:
        st.caption("Provider cascade: " +
                   " → ".join(f"{p} ({o})" for p, o in result.attempts))

    st.markdown("**Retrieved chunks (what the LLM was allowed to see):**")
    if not result.chunks:
        st.info("No chunks retrieved — the system abstained instead of guessing.")
    for rank, chunk in enumerate(result.chunks, 1):
        st.markdown(f"`[Source {rank}]` **{chunk.title}** · *{chunk.field}* · "
                    f"final={chunk.score:.3f} · bm25={chunk.bm25:.2f} · dense={chunk.dense:.3f}")
        st.code(chunk.text, language=None)

    if result.prompt:
        with st.expander("🧾 The exact prompt sent to the LLM"):
            st.code(result.prompt, language=None)

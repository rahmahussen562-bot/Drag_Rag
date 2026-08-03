"""
ui/pages/evaluation.py — run the eval suite from inside the app. (Phase 3)

Planned: run `eval/run_agent_eval.py` (cached), chart per-persona recall / MRR /
refusal accuracy, and expose the full prompt + context inspector for any single
query — today's evaluation mode, upgraded to a page.

# WHY surface evaluation in the UI rather than leaving it in CI?
# A buyer cannot read your CI. Numbers they can regenerate in front of you are
# worth more than numbers in a README, especially the refusal metrics, which are
# the ones that matter clinically and the ones a demo never shows by accident.

Not in the sidebar yet — Phase 0 is a zero-behaviour-change refactor.
"""
from __future__ import annotations

import streamlit as st


def render(ctx) -> None:
    st.title("📊 Evaluation")
    st.info("Planned for Phase 3. Reproducible today with "
            "`python eval/run_eval.py --embeddings` and `python eval/run_e2e.py`.",
            icon="🚧")

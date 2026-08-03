"""
ui/pages/retrieval_lab.py — the engineering-credibility page. (Phase 3)

Planned: sliders for alpha, k, the relevance floors, field_boost and mmr_lambda;
run the SAME query under two configurations side by side and diff the retrieved
sets; export the comparison as JSON.

# WHY does a product ship a parameter playground?
# Because "alpha = 0.20 was measured, not guessed" is a claim. This page lets a
# technical buyer re-run the measurement themselves, on their own query, and watch
# the ranking change. It converts a README table into something falsifiable.

Not in the sidebar yet — Phase 0 is a zero-behaviour-change refactor.
"""
from __future__ import annotations

import streamlit as st


def render(ctx) -> None:
    st.title("🔬 Retrieval lab")
    st.info("Planned for Phase 3. The alpha sweep it will expose is reproducible "
            "today with `python eval/tune_alpha.py`.", icon="🚧")

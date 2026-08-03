"""
ui/pages/audit.py — the append-only answer log. (Phase 3)

Planned: timestamp, persona, query, rewritten query, grounded entities, retrieval
status, source IDs, model and verification result for every answer; exportable to
CSV, plus PDF export of a single answer with its citations and disclaimer.

# WHY is this the feature that makes it sellable to a pharmacy?
# Because a pharmacist has to be able to FILE the advice they acted on. An answer
# that cannot be exported, dated and attributed is not usable in a regulated
# workplace, however good the retrieval is.
#
# Constraint carried forward from the brief: log no patient-identifying data. The
# audit log stores queries, so it needs a redaction pass before it is switched on.

Not in the sidebar yet — Phase 0 is a zero-behaviour-change refactor.
"""
from __future__ import annotations

import streamlit as st


def render(ctx) -> None:
    st.title("🧾 Audit log")
    st.info("Planned for Phase 3. Requires the query-redaction pass described in "
            "the brief before it stores anything.", icon="🚧")

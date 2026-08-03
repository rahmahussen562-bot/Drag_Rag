"""
ui.pages — one module per page, each exposing `render(ctx)`.

# WHY a registry dict instead of if/elif in the router?
# Adding a page becomes a data change, not a control-flow change: Phase 3 turns
# the four scaffolded pages on by moving them from PLANNED into PAGES. The router
# never grows a branch, and nothing else in the app knows how many pages exist.
"""
from __future__ import annotations

from . import architecture, ask, audit, evaluation, interactions, retrieval_lab, upload  # noqa: F401

#: label → module. Order is the sidebar order.
#
# Phase 0 ships exactly the three tools the app had before — this refactor is not
# allowed to change what the user sees.
PAGES = {
    "💬 Ask a question": ask,
    "⚠️ Interaction checker": interactions,
    "📄 Upload documents": upload,
}

#: Built, imported and testable — but not yet routed. Phase 3 merges these in.
PLANNED = {
    "🏗️ Architecture": architecture,
    "🔬 Retrieval lab": retrieval_lab,
    "📊 Evaluation": evaluation,
    "🧾 Audit log": audit,
}

__all__ = ["PAGES", "PLANNED"]

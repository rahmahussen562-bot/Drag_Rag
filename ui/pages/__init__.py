"""
ui.pages — one module per page, each exposing `render(ctx)`.

# WHY a registry dict instead of if/elif in the router?
# Adding a page is a data change, not a control-flow change. The router never
# grows a branch, and nothing else in the app needs to know how many pages exist.

Phase 3 turned on the four pages Phase 0 scaffolded. `evaluation` remains under
PLANNED: running the full suite from the UI needs the caching and per-persona
harness from Phase 1, and a page that ran it uncached would block the container
for minutes. Reproducible today with `python eval/run_e2e.py`.
"""
from __future__ import annotations

from . import (architecture, ask, audit, corpus, evaluation,  # noqa: F401
               interactions, retrieval_lab, upload)

#: label → module. Order is the sidebar order.
PAGES = {
    "💬 Ask a question": ask,
    "🏗️ Architecture": architecture,
    "📚 Sources & corpus": corpus,
    "🔬 Retrieval lab": retrieval_lab,
    "⚠️ Interaction checker": interactions,
    "📄 Upload documents": upload,
    "🧾 Audit log": audit,
}

#: Built and importable, not yet routed.
PLANNED = {
    "📊 Evaluation": evaluation,
}

__all__ = ["PAGES", "PLANNED"]

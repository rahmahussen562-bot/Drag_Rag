"""
ui/pages/architecture.py — the pipeline, explained inside the product. (Phase 3)

Planned: a rendered 8-stage diagram generated FROM A DATA STRUCTURE so it cannot
drift out of sync with the code, each stage clickable to show what it does, why it
exists and its live parameter values (imported, never hard-coded), plus a "run
this stage" button and the side-by-side view of where the two agents diverge.

# WHY does this page exist at all in Phase 0?
# So the router, the nav and the import graph are already shaped for it. Phase 0's
# contract is zero behaviour change, so it is NOT yet listed in the sidebar — the
# module exists, nothing renders it. Phase 3 adds one line to PAGES.
#
# The notebook (PhARMA_RAG_Final.ipynb) already generates exactly these figures
# from live constants; this page is that work moved in-product.
"""
from __future__ import annotations

import streamlit as st


def render(ctx) -> None:
    st.title("🏗️ Architecture")
    st.info("Planned for Phase 3. The generated diagrams already exist in "
            "`PhARMA_RAG_Final.ipynb` and `docs/figures/`.", icon="🚧")

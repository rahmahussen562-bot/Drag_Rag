"""
streamlit_app.py — STAGE 8 (final):  **Streamlit UI**

    Raw Documents → Preprocessing → Chunking → Vector Representation → Vector Store
    → Context Retrieval → Prompt Building → LLM Generation → >>> Streamlit UI <<<

# WHY is this file so short?
# It is the ROUTER, and nothing else: page config, session bootstrap, dispatch.
# It contains no pipeline logic and no markup.
#
#   · every RAG decision — how to chunk, how to score, when to refuse, what enters
#     the prompt — lives in the numbered stage modules and is testable from the
#     command line without Streamlit;
#   · every pixel lives in `ui/` — theme, components, one module per page;
#   · every cached resource lives in `ui/state.py`.
#
# That separation is what lets `python 06_retrieve_context.py` and the eval harness
# exercise byte-identical code to what the UI runs. It is also what makes a second
# persona a config change rather than a rewrite: pages read an `AppContext`, so
# swapping which agent builds the answer never touches page code.

Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="PhARMA RAG — Grounded Drug Q&A",
                   page_icon="💊", layout="centered",
                   initial_sidebar_state="expanded")

from ui import inject_css  # noqa: E402
from ui.pages import PAGES  # noqa: E402
from ui.sidebar import render as render_sidebar  # noqa: E402
from ui.state import bootstrap  # noqa: E402

inject_css()
bootstrap()

# The sidebar resolves every setting and builds the retriever, so it is also what
# produces the context each page consumes. Routing is then a dict lookup — adding
# a page is a data change in ui/pages/__init__.py, never a branch here.
page_label, ctx = render_sidebar()
PAGES[page_label].render(ctx)

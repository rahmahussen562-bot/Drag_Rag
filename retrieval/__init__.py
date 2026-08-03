"""
retrieval — query understanding, field routing, context building and tracing.

# WHY is this package separate from 06_retrieve_context.py?
# `06_retrieve_context.py` remains the SINGLE SOURCE OF TRUTH for retrieval — the
# grounding vocabulary, the hybrid blend, the relevance floors and the abstention
# path all live there and are exercised by `python 06_retrieve_context.py`, by the
# eval harness and by the app alike.
#
# What this package adds is everything that happens *around* that core: deciding
# what the query means before retrieval (query_understanding), which label section
# to favour (field_router), what enters the prompt afterwards (context_builder),
# and recording every decision so the UI can show it (trace).
#
# The rule for every module here: CALL INTO the stages, never fork them.

Phase 0 ships the seams with behaviour-preserving defaults — FIELD_BOOST is 0.0
and context_builder delegates to stage 07 — so this package can be imported
without changing a single retrieval result. Phases 1 and 2 fill them in, each
change measured against the ground-truth set before adoption.
"""
from __future__ import annotations

from .context_builder import build_context, context_stats  # noqa: F401
from .field_router import FIELD_BOOST, INTENT_TO_FIELDS, boost_multipliers, field_weights  # noqa: F401
from .query_understanding import ABBREVIATIONS, INTENTS, QueryPlan, normalise, plan  # noqa: F401
from .trace import RetrievalTrace, StageRecord  # noqa: F401

__all__ = [
    "QueryPlan", "plan", "normalise", "INTENTS", "ABBREVIATIONS",
    "INTENT_TO_FIELDS", "FIELD_BOOST", "field_weights", "boost_multipliers",
    "build_context", "context_stats",
    "RetrievalTrace", "StageRecord",
]

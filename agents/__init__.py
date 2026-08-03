"""
agents — the personas.

    from agents import get_agent, config_of, available
    agent = get_agent("patient")
    answer = agent.answer("what is metformin for?", retriever)

Importing this package registers both shipped personas. A third one is a new
module that calls `register(...)` — no change here, and no change in the UI.

# WHY does Phase 0 register agents that behave identically?
# So the seam is real and exercised before any behaviour rides on it. Both classes
# delegate to `07_prompting.answer_question`, so the eval suite proves the wiring
# is correct while the answers are provably unchanged. Phase 2 then changes ONE
# thing at a time — context building, then prompts — with a regression gate that
# was already green.

The configs already differ (see `config_of("patient").differences(...)`), they are
simply not consumed yet. That difference is what the Architecture page renders.
"""
from __future__ import annotations

from .base import (  # noqa: F401
    DEFAULT_AGENT,
    Agent,
    AgentConfig,
    available,
    config_of,
    get_agent,
    register,
)

# Importing the implementations is what populates the registry.
from .patient import RED_FLAGS, PatientAgent, is_red_flag  # noqa: F401
from .practitioner import PractitionerAgent  # noqa: F401

__all__ = [
    "AgentConfig", "Agent", "register", "get_agent", "config_of", "available",
    "DEFAULT_AGENT", "PatientAgent", "PractitionerAgent", "is_red_flag", "RED_FLAGS",
]

"""
agents/base.py — AgentConfig, the Agent protocol, and the registry.

# WHY is a persona a CONFIG object rather than a subclass?
# Because the selling point is that the two agents are genuinely different in
# their *retrieval*, not just their tone — and the honest way to prove that is to
# make the difference legible as data. Every knob that separates Patient from
# Practitioner is a field below. You can diff two AgentConfigs and see the whole
# product difference, which is exactly what the Architecture page will render.
#
# It also means a third persona ("Pharmacist", "Prescriber") is a config entry,
# not a code change.
#
# # WHY do both agents share ONE Retriever, ONE embedder, ONE store?
# Memory. Peak RSS is 868 MB against Streamlit Cloud's 1 GB limit, and a second
# sentence-transformer would add ~90 MB of weights plus a second embedding matrix.
# Config is the ONLY thing that differs at retrieval time — the index is shared.
# Any change that loads a second encoder per agent breaks the deployment.

Phase 0 defines the shape and registers both personas with configs that reproduce
today's single-path behaviour exactly. Phase 2 makes them diverge.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class AgentConfig:
    """Everything that distinguishes one persona from another.

    Frozen because a config is an identity, not a scratchpad: a page that mutated
    the shared Practitioner config would silently change every later request in
    the same container. Use `.variant(...)` for a one-off tweak.
    """
    id: str                             # "patient" | "practitioner"
    label: str                          # shown on the persona card
    description: str                    # one line, shown under the label
    prompt_path: str                    # prompts live as text files, not literals

    # ── retrieval ────────────────────────────────────────────────────────────
    k: int
    context_token_budget: int
    alpha: float                        # may differ per persona; justify with a sweep
    use_reranker: bool
    allowed_fields: Optional[frozenset]  # None = all fields
    blocked_fields: frozenset
    field_boost: float
    mmr_lambda: float
    expand_neighbours: bool
    lay_expansion: bool                 # lay→clinical terms, BM25 query only

    # ── generation ───────────────────────────────────────────────────────────
    temperature: float
    reading_level: str
    max_answer_words: int
    require_escalation_notice: bool
    output_schema: tuple                # section headings the answer must contain

    def prompt(self) -> str:
        """Load this persona's system prompt from disk.

        # WHY a file and not a string literal?
        # Prompts are edited by whoever is tuning behaviour, and a .md file can be
        # diffed, reviewed and versioned on its own. A 40-line triple-quoted string
        # buried in a module invites 'while I'm here' edits that never get read.
        """
        path = PROMPT_DIR / Path(self.prompt_path).name
        if not path.exists():
            raise FileNotFoundError(f"Prompt file missing for agent {self.id!r}: {path}")
        return path.read_text(encoding="utf-8").strip()

    def variant(self, **overrides) -> "AgentConfig":
        """A copy with fields overridden — for the Retrieval Lab's A/B comparison."""
        return replace(self, **overrides)

    def differences(self, other: "AgentConfig") -> dict:
        """{field: (mine, theirs)} for every field that differs.

        This is what the Architecture page's "the two agents diverge here" view
        renders. Computing it from the dataclass means the view can never claim a
        difference that does not exist, or miss one that does.
        """
        out = {}
        for f in self.__dataclass_fields__:
            if f in ("id", "label", "description", "prompt_path"):
                continue
            mine, theirs = getattr(self, f), getattr(other, f)
            if mine != theirs:
                out[f] = (mine, theirs)
        return out


@runtime_checkable
class Agent(Protocol):
    """What every persona must offer the UI."""

    config: AgentConfig

    def answer(self, query: str, retriever, **kwargs):
        """Return a stage-07 `Answer` for this persona."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Registry — a third persona is a registration, not a rewrite.
# ─────────────────────────────────────────────────────────────────────────────
_AGENTS: dict[str, Callable[[], Agent]] = {}
_CONFIGS: dict[str, AgentConfig] = {}

DEFAULT_AGENT = "practitioner"


def register(config: AgentConfig, factory: Callable[[], Agent]) -> None:
    _CONFIGS[config.id] = config
    _AGENTS[config.id] = factory


def available() -> list:
    """Registered persona ids, in registration order."""
    return list(_AGENTS)


def config_of(agent_id: str) -> AgentConfig:
    if agent_id not in _CONFIGS:
        raise KeyError(f"Unknown agent {agent_id!r}. Registered: {available()}")
    return _CONFIGS[agent_id]


def get_agent(agent_id: Optional[str] = None) -> Agent:
    agent_id = agent_id or DEFAULT_AGENT
    if agent_id not in _AGENTS:
        raise KeyError(f"Unknown agent {agent_id!r}. Registered: {available()}")
    return _AGENTS[agent_id]()

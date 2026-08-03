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
# Shared context building
#
# # WHY does field policy live here rather than in each agent?
# Because "which sections of a label is this reader allowed to see" is the same
# mechanism for every persona — only the field SET differs, and that is config.
# One implementation means one place for the allowlist bug to not be.
# ─────────────────────────────────────────────────────────────────────────────
#: How many extra candidates to retrieve when a persona filters by field.
#
# # WHY over-retrieve at all?
# The field policy runs AFTER stage 06 has already trimmed to k. Ask the retriever
# for 4 and filter 3 of them away and the patient gets ONE source — measured, not
# hypothetical: that is exactly what the first run of the persona tests produced.
# Fetching a wider pool and trimming afterwards is what makes "k=4" mean four
# readable sources rather than four candidates. Retrieval is ~2 ms, so the extra
# candidates are free; a starved context is not.
OVERFETCH = 5


def fetch_k(config: AgentConfig, k: Optional[int] = None) -> int:
    """How many candidates to ask stage 06 for, given this persona's policy."""
    effective = k or config.k
    if config.allowed_fields is None and not config.blocked_fields:
        return effective                      # no filtering → no need to over-fetch
    return min(effective * OVERFETCH, 60)     # cap: the pool is not free forever


def select_chunks(config: AgentConfig, chunks: list, limit: Optional[int] = None) -> list:
    """Apply a persona's field policy, packing order and k. Returns new chunks.

    Order of operations matters and is deliberate:
      1. blocked_fields   — a hard deny, applied first so nothing can re-admit it
      2. allowed_fields   — an allowlist, when the persona declares one
      3. packing order    — safety-first for lay readers, score order otherwise
      4. k                — trim last, so the trim sees the persona's ordering
    """
    out = [c for c in chunks if c.field not in config.blocked_fields]
    if config.allowed_fields is not None:
        out = [c for c in out if c.field in config.allowed_fields]

    if config.reading_level == "plain":
        # # WHY reorder for a lay reader?
        # Attention favours the EDGES of a long context ("lost in the middle"), so
        # whichever source lands first is the one most likely to shape the answer.
        # For a patient that must be the safety text, not the marketing-flavoured
        # indications paragraph that usually scores highest.
        priority = {"contraindications": 0, "side_effects": 1, "description": 2,
                    "indications": 3}
        out.sort(key=lambda c: (priority.get(c.field, 9), -c.score))

    return out[:(limit or config.k)]


def numbers_in(text: str) -> set:
    """Numeric tokens in a string — the raw material for the hallucination check."""
    import re
    return set(re.findall(r"\d+(?:\.\d+)?", str(text or "")))


def unsupported_numbers(answer: str, chunks: list) -> list:
    """Numbers in the answer that appear in NO source chunk.

    # WHY is this the highest-value check for a medical buyer?
    # Because the dangerous hallucination in this domain is not a wrong sentence,
    # it is a wrong NUMBER inside a right-looking sentence. A fabricated dose is
    # fluent, correctly cited, and clinically actionable. Citations cannot catch
    # it; word overlap cannot catch it. Comparing numeric tokens against the union
    # of the supplied sources can.
    """
    supported: set = set()
    for c in chunks:
        supported |= numbers_in(c.text)
    return sorted(n for n in numbers_in(answer) if n not in supported)


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

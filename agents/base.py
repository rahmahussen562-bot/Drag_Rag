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
      3. SELECT by score  — which chunks get in is a RELEVANCE question
      4. ORDER for packing— where they sit is an ATTENTION question

    # WHY are steps 3 and 4 separate, when one sort could do both?
    # Because they answer different questions, and merging them silently turns a
    # presentation preference into a retrieval filter. The first version of this
    # function sorted by (field_priority, -score) and THEN trimmed to k, so a
    # lower-priority field was dropped from the context entirely even when it was
    # the most relevant chunk retrieved — a patient asking "what is this for?"
    # could lose the `indications` chunk because `contraindications` sorted ahead
    # of it. Measured cost: 58% field recall over in-policy cases.
    # Select on relevance, then order the survivors. Never the other way round.
    """
    # # WHY are uploads exempt from the field policy?
    # Because the allowlist encodes "which sections of an FDA LABEL may a lay
    # reader see". An uploaded document's fields are its own headings — "page 3",
    # "Internal Clinic Protocol ZX-9", "row 5" — which match no label section, so
    # a naive allowlist silently deletes the user's ENTIRE document.
    #
    # Found by the counterfactual test once it was run per-persona: the patient
    # agent could not read the injected Zorbaxin protocol at all, because its
    # heading was not a label field. Stage 06 already exempts uploads from drug
    # grounding for the same reason — someone asking about their own report is
    # entitled to an answer from it — and the policy layer must agree.
    def in_policy(c) -> bool:
        if c.origin != "drug-corpus":
            return True
        if c.field in config.blocked_fields:
            return False
        return config.allowed_fields is None or c.field in config.allowed_fields

    out = [c for c in chunks if in_policy(c)]

    # 3 — SELECT: strictly by relevance. Retrieval already ranked these.
    out = sorted(out, key=lambda c: -c.score)[:(limit or config.k)]

    # 4 — ORDER: only among the chunks already selected.
    if config.reading_level == "plain":
        # # WHY reorder for a lay reader at all?
        # Attention favours the EDGES of a long context ("lost in the middle"), so
        # whichever source lands first is most likely to shape the answer. For a
        # patient that should be the safety text rather than the indications
        # paragraph. This changes emphasis WITHIN the context — it can no longer
        # change what the context contains.
        priority = {"contraindications": 0, "side_effects": 1, "description": 2,
                    "indications": 3}
        out.sort(key=lambda c: (priority.get(c.field, 9), -c.score))

    return out


def routing_for(config: AgentConfig, query: str) -> dict:
    """Field-routing weights for a query, honouring the persona's field policy.

    # WHY intersect the routing with the persona's allowlist?
    # Otherwise the router would boost a section the persona is then forbidden to
    # read — promoting `mechanism` for a patient asking "how does it work?", only
    # for select_chunks to delete every boosted chunk. The persona would end up
    # with a WORSE context than if routing were off, because the boost displaced
    # readable sections it was allowed to use.
    """
    from retrieval.field_router import field_weights
    from retrieval.query_understanding import classify_intents

    weights = field_weights(classify_intents(query))
    if not weights:
        return {}
    if config.allowed_fields is not None:
        weights = {f: w for f, w in weights.items() if f in config.allowed_fields}
    return {f: w for f, w in weights.items() if f not in config.blocked_fields}


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

"""
interactions/base.py — the interaction-source ADAPTER.

# WHY does this abstraction exist? (licensing, not architecture astronautics)
# DDInter 2.0 is CC BY-NC-SA 4.0 — **non-commercial**. The data is excellent and
# perfectly fine for teaching, evaluation and this demo, but it cannot be shipped
# inside a product that is sold. That is a licensing fact, not an engineering
# preference, and it is discovered late by projects that hard-code one source.
#
# So the interaction source sits behind a protocol. Swapping DDInter for a
# commercially-licensed feed later is then a CONFIG change (`provider="acme"`),
# not a refactor that reaches into the UI, the eval harness and the RAG loop.
#
# # WHY a Protocol rather than an ABC?
# `InteractionDB` already exists and already has the right shape. A Protocol lets
# it satisfy the contract WITHOUT inheriting from anything, so this file adds a
# seam without touching working code — which is the whole point of Phase 0.

The contract, deliberately small — three questions a pharmacist asks:

    resolve(name)        -> canonical name, or None if the source doesn't know it
    check_pair(a, b)     -> one pair
    check_list([names])  -> every pair in a medication list

`check_pair` returns a dict whose "status" is one of:

    "ok"             — a documented interaction; carries level/rank/emoji
    "no_interaction" — both drugs known, no documented pair  ← NOT the same thing
    "unknown_drug"   — at least one name could not be resolved
    "same_drug"      — the two names resolved to one drug

# WHY keep "no_interaction" and "unknown_drug" apart?
# Collapsing them is a clinical error. "No documented interaction" is evidence of
# absence; "I don't know this drug" is absence of evidence. A UI that renders both
# as a green tick is actively misleading, so the distinction is in the contract.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

# Severity ranking (higher = more dangerous). "Unknown" means DDInter documents
# the pair but did not classify its severity — still surfaced, ranked lowest.
LEVEL_RANK = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
LEVEL_EMOJI = {"Major": "🔴", "Moderate": "🟠", "Minor": "🟡", "Unknown": "⚪"}

# Every status a caller must be prepared to handle.
STATUSES = ("ok", "no_interaction", "unknown_drug", "same_drug")


@runtime_checkable
class InteractionProvider(Protocol):
    """What any interaction source must offer for the app to use it."""

    #: Human-facing attribution string, rendered under every result.
    citation: str
    #: Short id used in config, e.g. "ddinter".
    provider_id: str

    def resolve(self, name: str) -> Optional[str]:
        """Map an arbitrary drug name to this source's canonical key, or None."""
        ...

    def check_pair(self, a: str, b: str) -> dict:
        """Look up one pair. Returns a dict with a "status" key from STATUSES."""
        ...

    def check_list(self, names: list) -> dict:
        """Pairwise-check a medication list."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Registry
#
# # WHY a factory registry rather than importing the class directly?
# Loading DDInter reads 8 CSVs (~13 MB) and builds several indexes. Registering a
# FACTORY keeps that cost lazy, so a deployment configured for a different source
# never pays for one it does not use.
# ─────────────────────────────────────────────────────────────────────────────
_PROVIDERS: dict[str, Callable[..., InteractionProvider]] = {}

#: The source shipped by default. Change this (or pass `provider=` to
#: `get_provider`) to swap the interaction feed — see the licensing note above.
DEFAULT_PROVIDER = "ddinter"


def register(provider_id: str, factory: Callable[..., InteractionProvider]) -> None:
    """Register an interaction source under a short id."""
    _PROVIDERS[provider_id] = factory


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(provider_id: Optional[str] = None, **kwargs) -> InteractionProvider:
    """Instantiate an interaction source by id. Defaults to DEFAULT_PROVIDER."""
    provider_id = provider_id or DEFAULT_PROVIDER
    if provider_id not in _PROVIDERS:
        raise KeyError(
            f"Unknown interaction provider {provider_id!r}. "
            f"Registered: {available_providers() or '(none)'}")
    return _PROVIDERS[provider_id](**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Licensing metadata — surfaced in the UI, not buried in a comment.
#
# A buyer's first question is about liability and licensing. Keeping this as DATA
# means the app can render the actual terms of whatever source is configured,
# instead of a hard-coded sentence that goes stale the moment the source changes.
# ─────────────────────────────────────────────────────────────────────────────
LICENCES = {
    "ddinter": {
        "name": "DDInter 2.0",
        "url": "https://ddinter2.scbdd.com",
        "licence": "CC BY-NC-SA 4.0",
        "commercial_use": False,
        "note": ("Non-commercial licence. Fine for evaluation, teaching and this "
                 "demo. Before any commercial deployment: license it, replace it "
                 "with a commercially-licensed feed, or ship with the interaction "
                 "module disabled."),
    },
}


def licence_of(provider_id: Optional[str] = None) -> dict:
    """Licence metadata for a provider, for display in the UI."""
    return LICENCES.get(provider_id or DEFAULT_PROVIDER, {
        "name": provider_id or DEFAULT_PROVIDER, "url": "", "licence": "unknown",
        "commercial_use": False, "note": "No licence metadata registered.",
    })

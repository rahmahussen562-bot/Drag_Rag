"""
interactions — structured drug-drug interaction lookup, behind a swappable adapter.

# WHY is this a package now, when it used to be one file?
# Licensing. DDInter 2.0 is CC BY-NC-SA 4.0 (non-commercial), so the source has to
# be replaceable without a refactor. `base.py` defines the contract, `ddinter.py`
# implements it, and this file re-exports the names the rest of the app already
# imports — so the split cost zero call-site changes.

Public API (unchanged from the previous single-module version):

    from interactions import InteractionDB, DDINTER_CITATION

Adapter API (new — use this for anything that should survive a source swap):

    from interactions import get_provider, licence_of
    db = get_provider()               # the configured source, default "ddinter"
    licence_of().["commercial_use"]   # False for DDInter — surfaced in the UI
"""
from __future__ import annotations

from .base import (  # noqa: F401
    DEFAULT_PROVIDER,
    LEVEL_EMOJI,
    LEVEL_RANK,
    LICENCES,
    STATUSES,
    InteractionProvider,
    available_providers,
    get_provider,
    licence_of,
    register,
)

# Importing the implementation is what registers "ddinter" in the registry.
from .ddinter import (  # noqa: F401
    DDINTER_CITATION,
    DDINTER_DIR,
    InteractionDB,
    identity_tokens,
)

__all__ = [
    # original API — kept so existing imports keep working
    "InteractionDB", "DDINTER_CITATION", "DDINTER_DIR",
    "LEVEL_RANK", "LEVEL_EMOJI", "identity_tokens",
    # adapter API
    "InteractionProvider", "get_provider", "register", "available_providers",
    "licence_of", "LICENCES", "DEFAULT_PROVIDER", "STATUSES",
]

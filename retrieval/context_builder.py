"""
retrieval/context_builder.py — what actually ENTERS the prompt.

Retrieval returns chunks; context building decides which of them the model sees,
in what order, and where the budget runs out. They are different problems, and
this module keeps them separate.

Planned for Phase 1, each behind a flag and each measured before adoption:

  MMR diversity          five chunks from one paragraph is wasted budget
  neighbour expansion    a dose limit can straddle a chunk boundary ("...not to
                         exceed | 2000 mg per day..."), so pull ±1 and merge
  numeric-span guard     never end a packed source mid-number, mid-unit or
                         mid-negation; asserted against adversarial fixtures
  token budget           a real tokeniser count, not MAX_CONTEXT_CHARS
  deduplication          near-identical generic/brand chunks collapse to one
                         source carrying multiple provenance entries
  persona packing order  patient: safety fields first (attention favours edges)
                         practitioner: strict score order, components preserved

# WHY does Phase 0 delegate instead of implementing?
# Because `07_prompting.pack_context` is the single source of truth for packing,
# and a second implementation would drift from it the first time either changed.
# This module is the seam; Phase 1 moves the logic here and deletes nothing until
# the eval suite agrees the replacement is at least as good.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stages import load  # noqa: E402

prompting = load("07_prompting")

#: Phase 1 tunables, declared here so the Retrieval Lab can bind sliders to them.
MMR_LAMBDA = 0.7          # 1.0 = pure relevance, 0.0 = pure diversity
EXPAND_NEIGHBOURS = False
DEDUPLICATE = False


def build_context(chunks, max_chars: int = None, persona: str = None) -> str:
    """Pack retrieved chunks into the numbered [Source N] block for the prompt.

    Phase 0: delegates verbatim to stage 07. `persona` is accepted and ignored so
    both agents can already call this with their own identity, and Phase 1 can give
    it meaning without changing a single call site.
    """
    if max_chars is None:
        max_chars = prompting.MAX_CONTEXT_CHARS
    return prompting.pack_context(chunks, max_chars=max_chars)


def context_stats(context: str) -> dict:
    """Cheap size report — what the token-budget work in Phase 1 will replace."""
    return {
        "chars": len(context),
        "words": len(context.split()),
        "approx_tokens": int(len(context.split()) * 1.3),
        "sources": context.count("[Source "),
    }

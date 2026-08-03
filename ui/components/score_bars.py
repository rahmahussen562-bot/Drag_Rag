"""
ui/components/score_bars.py — stacked BM25 / dense / rerank contribution bar.

# WHY does a score need a picture?
# "score 0.973" is a number the buyer has to take on faith. The same result shown
# as 20% lexical + 80% semantic is an *explanation*: it says which retriever found
# this chunk and therefore why it is here. That is the difference between a
# relevance score and an auditable one.
#
# Status: built in Phase 0, rendered in Phase 3. It is deliberately NOT wired into
# `source_card` yet, because Phase 0's contract is zero visible change. Phase 3
# drops the call into the card header.

Contributions must be pre-normalised by the caller — this module draws, it does
not decide. Stage 06 owns the scoring maths and stays the single source of truth.
"""
from __future__ import annotations


def score_bar(lexical: float, dense: float, rerank: float = 0.0) -> str:
    """Render a stacked contribution bar as an HTML string.

    Each argument is that component's share of the final score. Shares are
    rescaled to sum to 1 so the bar always spans its container; a chunk with no
    score at all renders as an empty track rather than a divide-by-zero.
    """
    total = max(float(lexical) + float(dense) + float(rerank), 1e-9)
    lex_pct = 100.0 * float(lexical) / total
    den_pct = 100.0 * float(dense) / total
    rnk_pct = 100.0 * float(rerank) / total

    segments = [
        f'<span class="sb-lex" style="width:{lex_pct:.2f}%"></span>',
        f'<span class="sb-dense" style="width:{den_pct:.2f}%"></span>',
    ]
    if rnk_pct > 0:
        segments.append(f'<span class="sb-rank" style="width:{rnk_pct:.2f}%"></span>')

    key = [f"BM25 {lex_pct:.0f}%", f"dense {den_pct:.0f}%"]
    if rnk_pct > 0:
        key.append(f"rerank {rnk_pct:.0f}%")

    return (f'<div class="scorebar">{"".join(segments)}</div>'
            f'<div class="scorekey">{"".join(f"<span>{k}</span>" for k in key)}</div>')


def contributions(chunk, alpha: float, reranked: bool = False) -> tuple:
    """Split a RetrievedChunk's score into (lexical, dense, rerank) shares.

    Mirrors the blend in stage 06 — `alpha * minmax(bm25) + (1-alpha) * minmax(cos)`
    — using the raw component scores the chunk already carries. When the
    cross-encoder ran, its score REPLACED the hybrid score rather than adding to
    it, so the bar shows a single rerank segment instead of implying a blend that
    did not happen.
    """
    if reranked:
        return (0.0, 0.0, 1.0)
    return (alpha * float(chunk.bm25), (1.0 - alpha) * float(chunk.dense), 0.0)

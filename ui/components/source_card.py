"""
ui/components/source_card.py — the citation list.

# WHY render sources as cards instead of a bullet list?
# The citation list is the evidence for the answer, so it has to be scannable: the
# [Source N] marker must be findable at a glance to match it against the marker in
# the answer text. A card with the number badged on the left does that; a wall of
# bullets does not.
#
# # WHY show the SCORE BREAKDOWN rather than just the score?
# "score 0.973" is a number the reader has to take on faith. The same result shown
# as 20% lexical + 80% semantic is an explanation: it says WHICH retriever found
# this chunk, and therefore why it is here. That is the difference between a
# relevance score and an auditable one.
"""
from __future__ import annotations

import html
import re
from urllib.parse import quote

import streamlit as st

from ui.components.score_bars import contributions, score_bar

# openFDA has no stable per-record permalink for a generic name, so we link the
# label search. It lands the user on the authoritative source rather than on a
# 404 built from an id we do not have.
_OPENFDA_SEARCH = ("https://api.fda.gov/drug/label.json?search="
                   "openfda.generic_name:%22{}%22&limit=1")

_STOP = {"what", "which", "how", "does", "do", "is", "are", "the", "a", "an",
         "of", "for", "and", "or", "to", "in", "on", "with", "about", "side",
         "effect", "effects", "me", "my", "you", "tell", "can", "should"}


def _highlight(text: str, query: str) -> str:
    """Bold the query's content words inside the chunk body.

    # WHY highlight at all?
    # It answers "why was this retrieved?" without the reader parsing the whole
    # paragraph. Escaping happens FIRST and the markup is inserted afterwards, so
    # label text containing angle brackets cannot inject anything.
    """
    escaped = html.escape(text)
    terms = {t for t in re.findall(r"[a-z][a-z\-]{3,}", str(query or "").lower())
             if t not in _STOP}
    for term in sorted(terms, key=len, reverse=True):
        escaped = re.sub(rf"(?<![\w>])({re.escape(term)}\w*)",
                         r'<mark class="hl">\1</mark>', escaped, flags=re.IGNORECASE)
    return escaped


def source_card(rank: int, chunk, snippet_chars: int = 320, query: str = "",
                alpha: float = 0.20, reranked: bool = False) -> str:
    body = chunk.text
    # Strip the "Name — field: " prefix stage 03 added; the header already shows it.
    marker = f"{chunk.title} — {chunk.field}: "
    if body.startswith(marker):
        body = body[len(marker):]
    if len(body) > snippet_chars:
        body = body[:snippet_chars].rstrip() + "…"

    is_upload = chunk.origin != "drug-corpus"
    if is_upload:
        origin = (f'<span class="badge up">📄 '
                  f'{html.escape(chunk.origin.replace("upload:", ""))}</span>')
        link = ""
    else:
        origin = '<span class="badge fda">openFDA</span>'
        link = (f'<a class="srclink" target="_blank" rel="noopener" '
                f'href="{_OPENFDA_SEARCH.format(quote(str(chunk.drug)))}">'
                f'view label ↗</a>')

    lex, dense, rank_c = contributions(chunk, alpha=alpha, reranked=reranked)

    return (
        f'<div class="srccard" id="src-{rank}">'
        f'<div class="hd">'
        f'<span class="n">Source {rank}</span>'
        f'<span class="t">{html.escape(chunk.title)}</span>'
        f'<span class="f">{html.escape(chunk.field.replace("_", " "))}</span>'
        f'{origin}'
        f'<span class="s">score {chunk.score:.3f}</span>'
        f'</div>'
        f'{score_bar(lex, dense, rank_c)}'
        f'<div class="b">{_highlight(body, query)}</div>'
        f'{link}'
        f'</div>'
    )


def render_sources(chunks, expanded: bool = False, query: str = "",
                   alpha: float = 0.20, reranked: bool = False) -> None:
    """The citation list. Numbering matches [Source N] in the answer text."""
    if not chunks:
        return
    with st.expander(f"📚 Sources used ({len(chunks)})", expanded=expanded):
        st.markdown(
            "".join(source_card(i, c, query=query, alpha=alpha, reranked=reranked)
                    for i, c in enumerate(chunks, 1)),
            unsafe_allow_html=True)

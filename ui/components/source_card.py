"""
ui/components/source_card.py — the citation list.

# WHY render sources as cards instead of a bullet list?
# The citation list is the evidence for the answer, so it has to be scannable: the
# [Source N] marker must be findable at a glance to match it against the marker in
# the answer text. A card with the number badged on the left does that; a wall of
# bullets does not.
#
# # WHY is it shown by default rather than behind evaluation mode?
# "The answer cites its sources" is only meaningful if the user can READ those
# sources. This expander is what turns a citation from a claim into something
# verifiable in one click.
"""
from __future__ import annotations

import html

import streamlit as st


def source_card(rank: int, chunk, snippet_chars: int = 320) -> str:
    body = chunk.text
    # Strip the "Name — field: " prefix stage 03 added; the header already shows it.
    marker = f"{chunk.title} — {chunk.field}: "
    if body.startswith(marker):
        body = body[len(marker):]
    if len(body) > snippet_chars:
        body = body[:snippet_chars].rstrip() + "…"

    origin = ""
    if chunk.origin != "drug-corpus":
        origin = f' <span class="f">· {html.escape(chunk.origin.replace("upload:", "📄 "))}</span>'

    return (
        f'<div class="srccard">'
        f'<div class="hd">'
        f'<span class="n">Source {rank}</span>'
        f'<span class="t">{html.escape(chunk.title)}</span>'
        f'<span class="f">{html.escape(chunk.field.replace("_", " "))}</span>{origin}'
        f'<span class="s">score {chunk.score:.3f}</span>'
        f'</div>'
        f'<div class="b">{html.escape(body)}</div>'
        f'</div>'
    )


def render_sources(chunks, expanded: bool = False) -> None:
    """The citation list. Numbering matches [Source N] in the answer text."""
    if not chunks:
        return
    label = f"📚 Sources used ({len(chunks)})"
    with st.expander(label, expanded=expanded):
        st.markdown("".join(source_card(i, c) for i, c in enumerate(chunks, 1)),
                    unsafe_allow_html=True)

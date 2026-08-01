"""
ui.py — presentation layer for the Streamlit app.

# WHY a separate module?
# `streamlit_app.py` owns FLOW: which view is active, what the user asked, which
# pipeline call runs. This file owns APPEARANCE: CSS, badges, cards, status
# formatting. Keeping them apart means the chat logic stays readable instead of
# being buried under markup, and a styling change can never alter behaviour.
#
# # WHY hand-written CSS rather than a component library?
# Streamlit Cloud enforces a strict CSP and every extra component is another
# dependency on a ~1 GB container. Everything here is a few hundred bytes of
# inline CSS keyed off Streamlit's own theme variables, so it inherits the
# palette from .streamlit/config.toml and adapts to light AND dark automatically
# instead of hard-coding colours that break in one of them.
"""
from __future__ import annotations

import html

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Theme CSS
#
# Every colour below is expressed with Streamlit's CSS custom properties
# (--primary-color, --background-color, …) which Streamlit injects from
# config.toml. That is what makes one stylesheet correct in both themes.
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Rhythm ─────────────────────────────────────────────────────────────── */
.block-container { padding-top: 2.2rem; padding-bottom: 5rem; max-width: 52rem; }
h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: .2rem; }
h2, h3 { font-weight: 650; letter-spacing: -0.01em; }

/* ── Status badges (sidebar + chat meta) ────────────────────────────────── */
.pill {
  display: inline-flex; align-items: center; gap: .38em;
  padding: .2em .7em; border-radius: 999px;
  font-size: .76rem; font-weight: 600; line-height: 1.5;
  border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
  background: color-mix(in srgb, var(--text-color) 5%, transparent);
  white-space: nowrap;
}
.pill-ok   { border-color: color-mix(in srgb, #0F9D8E 45%, transparent);
             background: color-mix(in srgb, #0F9D8E 13%, transparent); }
.pill-warn { border-color: color-mix(in srgb, #D97706 45%, transparent);
             background: color-mix(in srgb, #D97706 14%, transparent); }
.pill-err  { border-color: color-mix(in srgb, #DC2626 45%, transparent);
             background: color-mix(in srgb, #DC2626 14%, transparent); }
.pill-dot  { width: .5em; height: .5em; border-radius: 50%; background: currentColor;
             opacity: .85; }

/* ── Sidebar status rows ────────────────────────────────────────────────── */
.statrow {
  display: flex; justify-content: space-between; align-items: center;
  gap: .5rem; padding: .3rem 0;
  border-bottom: 1px dashed color-mix(in srgb, var(--text-color) 10%, transparent);
  font-size: .84rem;
}
.statrow:last-child { border-bottom: none; }
.statrow .k { opacity: .72; }
.statrow .v { font-weight: 650; font-variant-numeric: tabular-nums; }

/* ── Source cards ───────────────────────────────────────────────────────── */
.srccard {
  border: 1px solid var(--border-color, color-mix(in srgb, var(--text-color) 12%, transparent));
  border-left: 3px solid var(--primary-color);
  border-radius: .6rem; padding: .7rem .85rem; margin: .5rem 0;
  background: color-mix(in srgb, var(--text-color) 3%, transparent);
}
.srccard .hd {
  display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
  margin-bottom: .35rem;
}
.srccard .n {
  font-size: .72rem; font-weight: 700; color: var(--primary-color);
  border: 1px solid color-mix(in srgb, var(--primary-color) 40%, transparent);
  border-radius: .35rem; padding: .05em .45em;
}
.srccard .t { font-weight: 650; font-size: .9rem; }
.srccard .f { opacity: .6; font-size: .78rem; }
.srccard .s { margin-left: auto; opacity: .55; font-size: .72rem;
              font-variant-numeric: tabular-nums; }
.srccard .b { font-size: .84rem; line-height: 1.5; opacity: .88; }

/* ── Answer block ───────────────────────────────────────────────────────── */
.answer { font-size: .97rem; line-height: 1.65; }
.answer p { margin-bottom: .6rem; }

/* ── Meta strip under an answer ─────────────────────────────────────────── */
.metastrip { display: flex; gap: .4rem; flex-wrap: wrap; margin: .55rem 0 .2rem; }

/* ── Example prompt buttons ─────────────────────────────────────────────── */
div[data-testid="stButton"] > button { font-weight: 550; }

/* ── Chat bubbles: give the assistant turn a little presence ────────────── */
div[data-testid="stChatMessage"] { padding: .3rem .1rem; }
</style>
"""


def inject_css() -> None:
    """Install the stylesheet once per rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Badges
# ─────────────────────────────────────────────────────────────────────────────
def pill(label: str, tone: str = "", dot: bool = False) -> str:
    """Return a badge as an HTML string. tone: '', 'ok', 'warn', 'err'."""
    cls = f"pill pill-{tone}" if tone else "pill"
    marker = '<span class="pill-dot"></span>' if dot else ""
    return f'<span class="{cls}">{marker}{html.escape(str(label))}</span>'


def pills(*items: str) -> str:
    """Wrap several badges in a spaced row."""
    return f'<div class="metastrip">{"".join(items)}</div>'


def stat_row(key: str, value: str, tone: str = "") -> str:
    """One key/value line for the sidebar status panel."""
    val = f'<span class="v">{value}</span>' if not tone else \
        f'<span class="v">{pill(value, tone, dot=True)}</span>'
    return f'<div class="statrow"><span class="k">{html.escape(key)}</span>{val}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Source rendering
#
# # WHY render sources as cards instead of a bullet list?
# The citation list is the evidence for the answer, so it has to be scannable:
# the [Source N] marker must be findable at a glance to match it against the
# marker in the answer text. A card with the number badged on the left does that;
# a wall of bullets does not.
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Errors
#
# # WHY a dedicated helper instead of st.exception()?
# st.exception renders a raw traceback. That is developer output leaking into a
# clinical tool: it exposes file paths, is unreadable to the user, and buries the
# one line that says what to actually do. Every failure path in the app routes
# through here, so the user always gets a plain sentence plus a next action, and
# the technical detail stays folded away.
# ─────────────────────────────────────────────────────────────────────────────
def error_box(title: str, detail: str = "", action: str = "", icon: str = "🚨") -> None:
    st.error(f"**{title}**" + (f"\n\n{action}" if action else ""), icon=icon)
    if detail:
        with st.expander("Technical detail"):
            st.code(detail.strip(), language=None)

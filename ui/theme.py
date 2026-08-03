"""
ui/theme.py — the single stylesheet, plus the badge primitives it defines.

# WHY hand-written CSS rather than a component library?
# Streamlit Cloud enforces a strict CSP and every extra component is another
# dependency on a ~1 GB container. Everything here is a few hundred bytes of
# inline CSS keyed off Streamlit's own theme variables, so it inherits the palette
# from .streamlit/config.toml and adapts to light AND dark automatically instead
# of hard-coding colours that break in one of them.
#
# # WHY do pill()/stat_row() live beside the CSS instead of in components/?
# They are not components, they are the *vocabulary* the CSS defines — each one is
# a thin function over a class in the stylesheet below. Splitting them from the
# rules they depend on is how a class gets renamed in one file and silently
# stops applying in another.
"""
from __future__ import annotations

import html

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Theme CSS
#
# Every colour is expressed with Streamlit's CSS custom properties
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

/* ── Score bars (stacked BM25 / dense / rerank contribution) ─────────────── */
.scorebar { display: flex; height: .34rem; border-radius: 999px; overflow: hidden;
            margin: .4rem 0 .15rem; background:
            color-mix(in srgb, var(--text-color) 8%, transparent); }
.scorebar span { display: block; height: 100%; }
.sb-lex  { background: #EA580C; }
.sb-dense{ background: #2563EB; }
.sb-rank { background: #059669; }
.scorekey { display: flex; gap: .6rem; font-size: .68rem; opacity: .6;
            font-variant-numeric: tabular-nums; }

/* ── Source card extras: origin badges, highlight, label link ───────────── */
.srccard .badge {
  font-size: .66rem; font-weight: 700; letter-spacing: .02em;
  padding: .1em .45em; border-radius: .3rem; text-transform: uppercase;
}
.srccard .badge.fda { color: #2563EB;
  background: color-mix(in srgb, #2563EB 14%, transparent); }
.srccard .badge.up  { color: #7C3AED;
  background: color-mix(in srgb, #7C3AED 16%, transparent); text-transform: none; }
mark.hl { background: color-mix(in srgb, #FACC15 42%, transparent);
          color: inherit; border-radius: .18em; padding: 0 .1em; }
a.srclink { font-size: .72rem; opacity: .65; text-decoration: none;
            display: inline-block; margin-top: .35rem; }
a.srclink:hover { opacity: 1; text-decoration: underline; }

/* ── Persona cards (the Ask page's two-product switch) ──────────────────── */
.persona {
  display: flex; gap: .75rem; margin: .2rem 0 1rem;
}
.pcard {
  flex: 1; border-radius: .7rem; padding: .8rem .9rem;
  border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
  background: color-mix(in srgb, var(--text-color) 3%, transparent);
}
.pcard.on { border-color: var(--primary-color); border-width: 2px;
            background: color-mix(in srgb, var(--primary-color) 10%, transparent); }
.pcard .pl { font-weight: 700; font-size: .95rem; display: flex;
             align-items: center; gap: .4rem; }
.pcard .pd { font-size: .78rem; opacity: .7; margin-top: .2rem; line-height: 1.45; }
.pcard .pm { font-size: .7rem; opacity: .55; margin-top: .45rem;
             font-variant-numeric: tabular-nums; }

/* ── Pipeline stepper (Architecture page + live trace) ──────────────────── */
.step { display: flex; align-items: baseline; gap: .6rem; padding: .28rem 0;
        border-bottom: 1px dashed color-mix(in srgb, var(--text-color) 9%, transparent); }
.step:last-child { border-bottom: none; }
.step .sn { font-size: .68rem; font-weight: 700; opacity: .5; min-width: 1.6rem; }
.step .st { font-weight: 600; font-size: .84rem; min-width: 9rem; }
.step .sd { font-size: .78rem; opacity: .68; }
.step .sms { margin-left: auto; font-size: .7rem; opacity: .5;
             font-variant-numeric: tabular-nums; }

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
# Badges — the vocabulary the CSS above defines
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

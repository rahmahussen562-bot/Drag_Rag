"""
ui/pages/interactions.py — the drug interaction checker.

# WHY is this a separate tool from Ask, when Ask can answer interaction questions?
# Because they answer different questions with different guarantees. Ask retrieves
# label *narrative* about a drug. This page runs a deterministic pairwise lookup
# over every combination in a medication list — it cannot miss a pair and cannot
# invent one. Structured lookup for facts, RAG for narrative.
"""
from __future__ import annotations

import re

import streamlit as st

from interactions import licence_of
from ui.components import DISCLAIMER
from ui.state import prompting


def render_interaction_block(res: dict) -> str:
    """Format a `check_list` result as markdown. Shared with the Ask page."""
    ints = res["interactions"]
    lines = []
    if ints:
        counts = ", ".join(f"{res['summary'][l]} {l}"
                           for l in ("Major", "Moderate", "Minor", "Unknown")
                           if res["summary"][l])
        lines.append(f"**{len(ints)} documented interaction(s)** ({counts}):\n")
        for r in ints:
            lines.append(f"- {r['emoji']} **{r['level']}** — **{r['a']}** + **{r['b']}**")
    elif not res["unresolved"]:
        lines.append(f"✅ **No documented interactions** among these drugs "
                     f"({res['safe_pairs']} pair(s) checked in DDInter).")
    if res["unresolved"]:
        # "Not in the database" is NOT "no interaction" — never render it as a
        # green tick. Absence of evidence, not evidence of absence.
        lines.append(f"\n⚠️ Not in the interaction database (skipped): "
                     f"*{', '.join(res['unresolved'])}*.")
    lines.append(f"\n<small>Source: {licence_of()['name']} — "
                 f"{licence_of()['licence']}</small>")
    return "\n".join(lines)


def render(ctx) -> None:
    st.title("⚠️ Drug interaction checker")
    st.caption("Enter the drugs on a prescription — one per line or comma-separated. "
               "Every pair is checked against DDInter 2.0. This is a deterministic "
               "lookup, not retrieval: it cannot miss a pair or invent one.")

    if not ctx.interactions_available:
        st.error(f"Interaction database not available: {ctx.idb}\n\nDownload the 8 DDInter "
                 "CSVs into `data/ddinter/` (see README).")
        return

    default = st.session_state.get("rx_text", "warfarin\nnaproxen\nmetformin\nomeprazole")
    text = st.text_area("Drugs / prescription", value=default, height=140,
                        placeholder="warfarin\nnaproxen\nmetformin")

    if st.button("🔎 Check interactions", type="primary"):
        names = [t for t in re.split(r"[\n,;]+", text) if t.strip()]
        if len(names) < 2:
            st.warning("Enter at least two drugs to check for interactions.")
        else:
            res = ctx.idb.check_list(names)
            summary = res["summary"]
            if summary["Major"]:
                st.error(f"🔴 {summary['Major']} MAJOR interaction(s) found — "
                         f"review before dispensing.")
            elif summary["Moderate"]:
                st.warning(f"🟠 {summary['Moderate']} moderate interaction(s) found.")
            elif res["interactions"]:
                st.info("🟡 Only minor / unclassified interactions found.")
            elif not res["unresolved"]:
                st.success("✅ No documented interactions among these drugs.")

            st.markdown(render_interaction_block(res), unsafe_allow_html=True)

            # Explain the top interaction with RAG narrative from the labels.
            top = res["interactions"][0] if res["interactions"] else None
            if top:
                with st.expander(f"📖 Label context for {top['a']} + {top['b']}"):
                    result = prompting.answer_question(
                        f"{top['a']} interaction with {top['b']}",
                        ctx.retriever, k=ctx.top_k, use_llm=False)
                    st.markdown(result.text)

    st.divider()
    st.caption(DISCLAIMER)

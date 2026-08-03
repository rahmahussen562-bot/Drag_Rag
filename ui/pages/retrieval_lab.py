"""
ui/pages/retrieval_lab.py — run the same query under two configurations, side by side.

# WHY does a product ship a parameter playground?
# Because "α = 0.20 was measured, not guessed" is a claim, and a claim a buyer
# cannot test is marketing. This page lets them re-run the comparison on their own
# query and watch the ranking change — including the case that argues AGAINST the
# shipped setting, which is the one worth showing.
#
# # WHY does it never write back to the retriever?
# Because a page that mutated shared state would change what every other session
# gets. Each panel builds a ranking from the same store at a different alpha, reads
# the result, and throws the config away. Nothing here can affect the app.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from ui.state import retrieval


def _rank(retriever, query: str, alpha: float, k: int, floor_on: bool) -> list:
    """Rank chunks at a given alpha WITHOUT mutating the shared retriever.

    Mirrors stage 06's blend exactly — min-max both score vectors, weight, apply
    the drug restriction, then the floor. It calls the retriever's own scoring
    helpers rather than reimplementing them, so the lab cannot drift from the app.
    """
    ground = retriever.ground(query)
    matched = ground["matched"]
    raw_bm25, raw_dense = retriever._raw_scores(query)

    if raw_dense is None:
        combined = raw_bm25
    else:
        combined = (alpha * retriever._minmax(raw_bm25)
                    + (1 - alpha) * retriever._minmax(raw_dense))

    n = len(combined)
    candidates = np.arange(n)
    if matched:
        mask = np.fromiter(
            (bool(retriever.chunk_tokens[i] & matched) for i in range(n)),
            dtype=bool, count=n)
        if mask.any():
            candidates = np.where(mask)[0]

    order = candidates[np.argsort(combined[candidates])[::-1]]

    def clears(i: int) -> bool:
        if not floor_on:
            return True
        if raw_dense is not None:
            return bool(raw_dense[i] >= retrieval.EMB_FLOOR
                        or raw_bm25[i] >= retrieval.BM25_FLOOR)
        return bool(raw_bm25[i] >= retrieval.BM25_FLOOR)

    out = []
    for i in order:
        i = int(i)
        if clears(i):
            row = retriever.df.loc[i]
            out.append({
                "rank": len(out) + 1,
                "drug": row["drug"],
                "field": row["field"],
                "score": round(float(combined[i]), 4),
                "bm25": round(float(raw_bm25[i]), 2),
                "dense": round(float(raw_dense[i]), 3) if raw_dense is not None else 0.0,
                "chunk": i,
            })
            if len(out) == k:
                break
    return out


def _panel(col, label: str, retriever, query: str, key: str, default_alpha: float):
    with col:
        st.markdown(f"**{label}**")
        alpha = st.slider("α — BM25 weight", 0.0, 1.0, default_alpha, 0.05,
                          key=f"a_{key}",
                          help="0 = pure semantic · 1 = pure keyword")
        k = st.slider("k", 1, 15, 5, key=f"k_{key}")
        floor_on = st.toggle("Relevance floor", value=True, key=f"f_{key}",
                             help="Off = never abstain. Shows what the floor is "
                                  "actually preventing.")
        rows = _rank(retriever, query, alpha, k, floor_on) if query else []
        if query and not rows:
            st.warning("Nothing cleared the floor — this configuration abstains.")
        elif rows:
            st.dataframe(pd.DataFrame(rows)[["rank", "drug", "field", "score",
                                             "bm25", "dense"]],
                         hide_index=True, use_container_width=True, height=320)
        return {"alpha": alpha, "k": k, "floor": floor_on, "results": rows}


def render(ctx) -> None:
    st.title("🔬 Retrieval lab")
    st.caption("The same query, two configurations, side by side. Nothing here "
               "writes back to the running app.")

    query = st.text_input("Query", value="how does metformin work?",
                          placeholder="how does metformin work?")

    left, right = st.columns(2)
    a = _panel(left, "Configuration A", ctx.retriever, query, "a", retrieval.ALPHA)
    b = _panel(right, "Configuration B", ctx.retriever, query, "b", 0.60)

    if not query or not (a["results"] or b["results"]):
        return

    # ── The diff: what actually changed between the two rankings ────────────
    st.divider()
    st.markdown("### What changed")
    set_a = {(r["drug"], r["field"]) for r in a["results"]}
    set_b = {(r["drug"], r["field"]) for r in b["results"]}
    only_a, only_b, both = set_a - set_b, set_b - set_a, set_a & set_b

    c1, c2, c3 = st.columns(3)
    c1.metric("In both", len(both))
    c2.metric("Only in A", len(only_a))
    c3.metric("Only in B", len(only_b))

    if only_a or only_b:
        st.markdown("".join(
            [f'<div class="step"><span class="sn">A</span>'
             f'<span class="st">{d}</span><span class="sd">{f}</span></div>'
             for d, f in sorted(only_a)] +
            [f'<div class="step"><span class="sn">B</span>'
             f'<span class="st">{d}</span><span class="sd">{f}</span></div>'
             for d, f in sorted(only_b)]), unsafe_allow_html=True)
    else:
        st.success("Identical result sets — α made no difference for this query.")

    st.download_button(
        "⬇ Export comparison as JSON",
        data=json.dumps({"query": query, "A": a, "B": b}, indent=2),
        file_name="retrieval_comparison.json", mime="application/json")

    st.info(
        "**The measured result, for reference.** Sweeping α over the ground-truth "
        f"set gives α={retrieval.ALPHA} as the choice — but note *why*: pure-dense "
        "(α=0.00) scores the **best** field recall and is still **rejected**, "
        "because it drops drug recall below 100%. With no lexical weight an exact "
        "drug-name match earns no score of its own. Missing the right *field* "
        "answers the wrong section of the right drug; missing the right *drug* "
        "answers from a different medicine. Reproduce with "
        "`python eval/tune_alpha.py`.", icon="📐")

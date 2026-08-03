"""
ui/pages/corpus.py — what is actually in the index, and where it came from.

# WHY does a buyer need this page?
# Because "506 FDA drug labels" is a claim about coverage, and coverage is the
# first thing a pharmacist will test — by searching for a drug they dispense. A
# searchable list turns "does it know X?" from a guess into a lookup, and makes
# the corpus boundary (the thing that drives every refusal) visible rather than
# something discovered by being refused.
#
# Every number here is computed from the live chunk table, never hard-coded.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from interactions import licence_of


def render(ctx) -> None:
    df = ctx.retriever.df
    st.title("📚 Sources & corpus")
    st.caption("Everything the system is allowed to answer from. Anything outside "
               "this list is refused, by design.")

    drug_corpus = df[df["origin"] == "drug-corpus"]
    uploads = df[df["origin"] != "drug-corpus"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drugs", f"{drug_corpus['drug'].nunique():,}")
    c2.metric("Chunks", f"{len(df):,}")
    c3.metric("Label sections", f"{drug_corpus['field'].nunique():,}")
    c4.metric("Your uploads", f"{len(uploads):,}")

    # ── Field coverage ──────────────────────────────────────────────────────
    st.markdown("### Coverage by label section")
    by_field = (drug_corpus.groupby("field")
                .agg(chunks=("text", "size"), drugs=("drug", "nunique"))
                .sort_values("chunks", ascending=False))
    by_field["% of drugs"] = (100 * by_field["drugs"]
                              / drug_corpus["drug"].nunique()).round(1)
    st.dataframe(by_field, use_container_width=True)
    st.caption("A section missing for a drug is why an otherwise-known drug can "
               "still produce 'I found it, but nothing relevant enough'.")

    # ── Searchable drug list ────────────────────────────────────────────────
    st.markdown("### Is a drug covered?")
    needle = st.text_input("Search the indexed drugs",
                           placeholder="metformin, warfarin, naproxen…")
    names = sorted(drug_corpus["title"].unique())
    if needle:
        hits = [n for n in names if needle.strip().lower() in n.lower()]
        if hits:
            st.success(f"{len(hits)} match(es) — these are answerable.")
            sub = drug_corpus[drug_corpus["title"].isin(hits)]
            st.dataframe(
                sub.groupby("title").agg(
                    sections=("field", lambda s: ", ".join(sorted(set(s)))),
                    chunks=("text", "size")),
                use_container_width=True)
        else:
            st.warning(
                f"**'{needle}' is not in the corpus** — the system will refuse "
                f"questions about it rather than answer from a similar drug. "
                f"That refusal is the product working, not failing.")
    else:
        st.caption(f"{len(names):,} drugs indexed. Showing the first 40.")
        st.dataframe(pd.DataFrame({"drug": names[:40]}), hide_index=True,
                     use_container_width=True, height=280)

    # ── Provenance and licensing ────────────────────────────────────────────
    st.markdown("### Data provenance & licensing")
    lic = licence_of()
    st.dataframe(pd.DataFrame([
        {"source": "openFDA drug labels", "licence": "Public domain",
         "commercial use": "Yes", "used for": "the 506-drug corpus"},
        {"source": lic["name"], "licence": lic["licence"],
         "commercial use": "Yes" if lic["commercial_use"] else "NO",
         "used for": "drug-drug interaction lookup"},
        {"source": "RxNorm (NLM)", "licence": "NLM terms",
         "commercial use": "See NLM terms",
         "used for": "optional brand→generic normalisation"},
    ]), hide_index=True, use_container_width=True)

    if not lic["commercial_use"]:
        st.warning(f"**{lic['name']} is {lic['licence']} — non-commercial.** "
                   f"{lic['note']}", icon="⚖️")

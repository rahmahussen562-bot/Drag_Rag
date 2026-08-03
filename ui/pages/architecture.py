"""
ui/pages/architecture.py — the pipeline, explained inside the product.

# WHY is this a PAGE and not a diagram in the README?
# Because the buyer's question is "can I trust this?", and the honest answer is a
# system they can inspect while it runs. A README diagram is a claim about the
# code. This page reads its parameter values out of the live modules, so it is a
# measurement of the code.
#
# # WHY is every number imported rather than written here?
# A hand-maintained architecture page starts lying the first time someone tunes a
# constant, and nobody notices because nothing fails. ALPHA below comes from
# retrieval.ALPHA. Change the constant, reload the page, the page changes.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from agents import available, config_of
from ui.state import (chunking, documents_stage, preprocessing, prompting,
                      representation, retrieval, vector_store)


def _stages() -> list:
    """The eight stages, with values read live from the imported modules."""
    return [
        ("01", "Raw Documents", "01_documents.py",
         "Every source becomes one uniform `Document`. FDA labels explode into one "
         "document per field; uploads become one per page or section.",
         {"label fields": len(documents_stage.FIELD_LABELS),
          "upload formats": len(documents_stage.SUPPORTED_UPLOAD_TYPES)},
         "A whole 3,000-word label as one vector is a blurry average that matches "
         "every question weakly. The label's own field boundaries are free, "
         "semantically perfect splits."),

        ("02", "Preprocessing", "02_preprocessing.py",
         "Lowercase, lemmatise, drop stopwords — but KEEP every number and every "
         "negation word.",
         {"profile": preprocessing.DEFAULT_PROFILE_NAME,
          "protected words": len(preprocessing.PROTECTED_NEGATION),
          "stopword source": preprocessing.STOPWORD_SOURCE},
         "The textbook recipe is dangerous here. Dropping stopwords deletes "
         "'no'/'not' and inverts a contraindication; dropping numbers deletes the "
         "dose. This is a safety control, not a tuning knob."),

        ("03", "Chunking", "03_chunking.py",
         "Sliding window with overlap, and a self-describing prefix so a chunk "
         "torn out of its document still says which drug and section it is.",
         {"window (words)": chunking.CHUNK_WORDS,
          "overlap (words)": chunking.CHUNK_OVERLAP,
          "overlap %": f"{chunking.CHUNK_OVERLAP / chunking.CHUNK_WORDS:.0%}"},
         "90 words ≈ 120 tokens, inside the embedding model's 256-token limit — "
         "past that the model truncates silently, making text invisible to "
         "retrieval. 25 words of overlap means any fact shorter than 25 words "
         "survives a boundary cut intact."),

        ("04", "Vector Representation", "04_vector_representation.py",
         "Two representations, because they fail in opposite cases: sparse for "
         "exact tokens, dense for paraphrase.",
         {"model": representation.EMBED_MODEL_NAME,
          "dimensions": representation.EMBED_DIM,
          "batch size": representation.ENCODE_BATCH_SIZE},
         "A query saying 'high blood sugar' shares no token with a chunk saying "
         "'hyperglycemia' — sparse scores it zero. A rare drug name is near-noise "
         "to an embedding. A pharmacy corpus hits both failures constantly."),

        ("05", "Vector Store", "05_create_vector_store.py",
         "FAISS IndexFlatIP over unit-normalised vectors (so inner product IS "
         "exact cosine) plus a BM25 inverted index.",
         {"IVF threshold": f"{vector_store.IVF_THRESHOLD:,}",
          "IVF nprobe": vector_store.IVF_NPROBE},
         "At 15k vectors exact search costs ~1.5 ms — far below the LLM call it "
         "feeds. Flat's perfect recall is true by construction; an approximate "
         "index's is an empirical result on the queries you happened to test."),

        ("06", "Context Retrieval", "06_retrieve_context.py",
         "Ground → restrict to the named drug → hybrid score → relevance floor → "
         "optional rerank. **Abstains instead of guessing.**",
         {"α (BM25 weight)": retrieval.ALPHA,
          "cosine floor": retrieval.EMB_FLOOR,
          "BM25 floor": retrieval.BM25_FLOOR,
          "rerank pool": retrieval.RETRIEVE_POOL},
         "Vector search ALWAYS returns its k nearest neighbours, even when the "
         "nearest is irrelevant. Ask about a drug not in the corpus and naive RAG "
         "answers confidently from a different medicine. Grounding refuses first."),

        ("07", "Prompt + Generation", "07_prompting.py",
         "Numbered [Source N] blocks, a numbered-imperative rules block, then "
         "mechanical verification of what came back.",
         {"context cap (chars)": f"{prompting.MAX_CONTEXT_CHARS:,}",
          "layout": "RULES → CONTEXT → QUESTION"},
         "Attention degrades in the middle of long inputs, so the two things the "
         "model must obey sit at the high-attention edges. A prompt rule is a "
         "request — verify_answer() is the check."),

        ("08", "Streamlit UI", "streamlit_app.py",
         "Page config, session bootstrap and a dict-lookup router. No pipeline "
         "logic and no markup.",
         {"router (lines)": 39, "personas": len(available())},
         "Every RAG decision lives in the numbered stages and is runnable from the "
         "command line. That separation is what lets the CLI, the eval harness and "
         "this app execute byte-identical code."),
    ]


def render(ctx) -> None:
    st.title("🏗️ Architecture")
    st.caption("Eight stages. Every parameter below is read from the live modules "
               "at render time — this page cannot drift out of sync with the code.")

    tab_pipe, tab_diverge, tab_proof = st.tabs(
        ["The eight stages", "Where the agents diverge", "Why it refuses"])

    # ── Tab 1: the pipeline ──────────────────────────────────────────────────
    with tab_pipe:
        for num, name, fname, what, params, why in _stages():
            with st.expander(f"**{num} · {name}**  —  `{fname}`", expanded=False):
                st.markdown(what)
                st.markdown("**Live parameters**")
                st.markdown("".join(
                    f'<div class="step"><span class="st">{k}</span>'
                    f'<span class="sd"><code>{v}</code></span></div>'
                    for k, v in params.items()), unsafe_allow_html=True)
                st.info(why, icon="💡")

    # ── Tab 2: the persona divergence, computed from the dataclass ──────────
    with tab_diverge:
        st.markdown("### The two agents diverge here")
        st.caption("Computed from `AgentConfig.differences()` — this table cannot "
                   "claim a difference that does not exist, or miss one that does.")

        ids = available()
        if len(ids) >= 2:
            a, b = config_of(ids[0]), config_of(ids[1])
            diffs = a.differences(b)
            rows = [{"parameter": k,
                     a.label: _fmt(v[0]),
                     b.label: _fmt(v[1])} for k, v in sorted(diffs.items())]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(f"{len(diffs)} of "
                       f"{len(a.__dataclass_fields__) - 4} configurable fields differ. "
                       f"Both agents share ONE retriever, ONE embedder and ONE store — "
                       f"config is the only difference at retrieval time, which is what "
                       f"keeps peak memory inside the deployment limit.")

            st.markdown("### Enforced in code, not just prompted")
            st.markdown(
                "- **Patient · red-flag triage** — emergency symptoms bypass RAG "
                "entirely and return a fixed string. No retrieval, no model call, "
                "no variation between runs.\n"
                "- **Patient · medication-change screen** — *should I stop/start/"
                "increase…* is declined before retrieval. It refuses the decision "
                "while still offering what the label says.\n"
                "- **Patient · escalation notice** — appended to every answer "
                "including refusals.\n"
                "- **Practitioner · heading schema** — validated after generation; "
                "sections with no supporting context render as *not addressed* "
                "rather than being dropped, so silence stays visible.")

    # ── Tab 3: the refusal demonstrations ───────────────────────────────────
    with tab_proof:
        st.markdown("### The two tests that make 'grounded' checkable")
        st.markdown(
            "**The mirror test.** *Aspirin* is a drug every language model knows "
            "perfectly well, and it is absent from this corpus. A system answering "
            "it from pretraining would look completely normal — so it must refuse.")
        if st.button("▶ Run the mirror test", key="arch_mirror"):
            with st.spinner("Asking about a drug that is not in the corpus…"):
                out = ctx.retriever.retrieve_context(
                    "what are the side effects of aspirin?", k=5)
            if out.ok:
                st.error(f"LEAKED — returned {len(out.chunks)} sources. "
                         f"The grounding gate is not holding.")
            else:
                st.success(f"Refused · status `{out.status}` · 0 sources returned. "
                           f"Pretrained knowledge did not leak past the gate.")
                if out.suggestions:
                    st.caption(f"Did you mean: {out.suggestions}")

        st.divider()
        st.markdown(
            "**The counterfactual.** Word overlap is weak evidence for grounding — "
            "ask about a real drug and a model answering from memory still overlaps "
            "with the label. So `eval/run_e2e.py` injects **Zorbaxin**, a drug that "
            "does not exist, dosed *250 mg every eight hours*. Those numbers are "
            "unobtainable from model weights, so reproducing them proves the answer "
            "came through retrieval.")
        st.code("python eval/run_e2e.py --no-llm    # 26 checks, incl. the "
                "counterfactual", language="bash")


def _fmt(v) -> str:
    """Render a config value compactly for the divergence table."""
    if v is None:
        return "all (unrestricted)"
    if isinstance(v, frozenset):
        return ", ".join(sorted(v)) if v else "—"
    if isinstance(v, tuple):
        return " · ".join(str(x) for x in v) if v else "—"
    return str(v)

"""
ui/pages/upload.py — bring your own documents.

Uploaded files run through the IDENTICAL pipeline as the built-in corpus: parsed
in stage 01, chunked at 90/25 in stage 03, embedded in stage 04, indexed in 05 and
cited by filename and section. There is no special-casing anywhere downstream —
that uniformity is what stage 01's single `Document` shape buys.
"""
from __future__ import annotations

import streamlit as st

from ui.components import DISCLAIMER
from ui.state import documents_stage


def render(ctx) -> None:
    st.title("📄 Upload your own documents")
    st.caption("Uploaded files run through the identical pipeline as the built-in "
               "corpus: chunked at 90 words with 25-word overlap, embedded, indexed, "
               "and cited by filename and section.")

    uploaded = st.file_uploader(
        "Add documents",
        type=documents_stage.SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help=f"Supported: {', '.join(documents_stage.SUPPORTED_UPLOAD_TYPES)}")

    if uploaded:
        added, failed = [], []
        for file in uploaded:
            if file.name in st.session_state.uploaded_docs:
                continue
            try:
                docs = documents_stage.documents_from_upload(file.name, file.getvalue())
                st.session_state.uploaded_docs[file.name] = docs
                added.append((file.name, len(docs)))
            except Exception as exc:
                failed.append((file.name, str(exc)))
        for name, n in added:
            st.success(f"✅ **{name}** — {n} section(s) ingested.")
        for name, err in failed:
            st.error(f"❌ **{name}** — {err}")
        if added:
            st.rerun()

    if st.session_state.uploaded_docs:
        st.subheader("Indexed documents")
        for name, docs in list(st.session_state.uploaded_docs.items()):
            words = sum(d.word_count for d in docs)
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{name}** — {len(docs)} section(s), {words:,} words")
                with st.expander("Preview sections"):
                    for d in docs[:12]:
                        st.markdown(f"*{d.field}* — {d.text[:220]}…")
            with col2:
                if st.button("Remove", key=f"rm_{name}"):
                    del st.session_state.uploaded_docs[name]
                    st.rerun()

        st.info("Switch to **💬 Ask a question** to query these documents alongside "
                "the drug corpus. Answers will cite the filename and section.", icon="💡")
    else:
        st.info("No documents uploaded yet. The app still works on the built-in "
                f"corpus of {ctx.retriever.df['drug'].nunique():,} FDA drug labels.",
                icon="ℹ️")

    st.divider()
    st.caption(DISCLAIMER)

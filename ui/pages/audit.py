"""
ui/pages/audit.py — the append-only answer log.

# WHY is this the feature that makes it sellable to a pharmacy?
# Because a pharmacist has to be able to FILE the advice they acted on. An answer
# that cannot be exported, dated and attributed is not usable in a regulated
# workplace, however good the retrieval is.
#
# # WHY is redaction applied on WRITE rather than on export?
# Because a log that stores identifiers and strips them at export time has already
# stored them. Anything the redactor would remove must never reach the record in
# the first place — that is the difference between "we don't show PII" and "we
# don't hold PII", and only the second one survives a data-protection review.
#
# Scope, stated honestly: this log is SESSION-scoped. It lives in st.session_state
# and disappears when the session ends. A pharmacy deployment needs a durable,
# tamper-evident store (append-only file or database) — that is a deployment
# decision, not a UI one, and it is called out in the page rather than implied.
"""
from __future__ import annotations

import csv
import io
import re

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Redaction — applied before anything is recorded.
#
# Deliberately conservative and pattern-based. It is NOT a guarantee of
# de-identification; it is a floor that stops the obvious identifiers being
# written down by accident. Free text always needs a human policy on top.
# ─────────────────────────────────────────────────────────────────────────────
_REDACTIONS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
    (re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"), "[phone]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[id]"),
    (re.compile(r"\b(?:mrn|nhs|patient\s*(?:id|no|number))\s*[:#]?\s*\w+\b",
                re.IGNORECASE), "[record-id]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[date]"),
]


def redact(text: str) -> str:
    """Strip obvious identifiers. Applied on write, never on read."""
    out = str(text or "")
    for pattern, replacement in _REDACTIONS:
        out = pattern.sub(replacement, out)
    return out


def record(result, query: str, persona: str, when: str) -> None:
    """Append one answer to the session log. Call after every generation."""
    v = getattr(result, "verification", None)
    st.session_state.setdefault("audit", []).append({
        "time": when,
        "persona": persona,
        "query": redact(query),
        "status": result.status,
        "mode": result.mode,
        "sources": len(result.chunks),
        "source_ids": "; ".join(f"{c.title}·{c.field}" for c in result.chunks),
        "provider": result.provider or "—",
        "model": result.model or "—",
        "cited": f"{len(v.citations)}/{v.n_sources}" if v else "—",
        "overlap": f"{v.overlap:.0%}" if v else "—",
        "grounded": (("yes" if v.grounded else "no") if v else "—"),
        "withheld_fields": ", ".join(result.dropped_fields) or "—",
    })


def render(ctx) -> None:
    st.title("🧾 Audit log")
    st.caption("Every answer this session, with its sources, provider and "
               "verification result. Queries are redacted before they are stored.")

    rows = st.session_state.get("audit", [])
    if not rows:
        st.info("No answers recorded yet this session. Ask something on the "
                "**Ask** page and it will appear here.", icon="📭")
        _footnotes()
        return

    df = pd.DataFrame(rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Answers", len(df))
    c2.metric("Refused", int((df["status"] != "ok").sum()))
    c3.metric("Grounded", int((df["grounded"] == "yes").sum()))
    c4.metric("Personas used", df["persona"].nunique())

    st.dataframe(df, hide_index=True, use_container_width=True)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(df.columns))
    writer.writeheader()
    writer.writerows(rows)
    st.download_button("⬇ Export as CSV", data=buf.getvalue(),
                       file_name="pharma_rag_audit.csv", mime="text/csv")

    if st.button("🗑 Clear the log"):
        st.session_state["audit"] = []
        st.rerun()

    _footnotes()


def _footnotes() -> None:
    st.divider()
    st.warning(
        "**This log is session-scoped.** It is held in memory and is gone when the "
        "session ends. A pharmacy deployment needs a durable, tamper-evident store — "
        "that is a deployment decision, not a UI one.", icon="⚠️")
    with st.expander("What redaction does, and does not, guarantee"):
        st.markdown(
            "Applied **on write**, so identifiers never reach the record — a log "
            "that stores them and strips them at export has already stored them.\n\n"
            "Patterns removed: email addresses, phone numbers, "
            "SSN-shaped digits, MRN / NHS / patient-number labels, and dates.\n\n"
            "**This is a floor, not de-identification.** Free text can always carry "
            "identifying detail that no pattern catches. A regulated deployment "
            "needs a human policy on top of this, and should say so to its users.")

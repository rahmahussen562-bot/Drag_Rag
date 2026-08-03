"""
ui/components/notices.py — error and warning surfaces.

# WHY a dedicated helper instead of st.exception()?
# st.exception renders a raw traceback. That is developer output leaking into a
# clinical tool: it exposes file paths, is unreadable to the user, and buries the
# one line that says what to actually do. Every failure path in the app routes
# through here, so the user always gets a plain sentence plus a next action, and
# the technical detail stays folded away.
"""
from __future__ import annotations

import streamlit as st

DISCLAIMER = ("⚠️ **Educational tool — not medical advice.** Interaction and drug data can be "
              "incomplete. Always confirm with a pharmacist or prescriber.")


def error_box(title: str, detail: str = "", action: str = "", icon: str = "🚨") -> None:
    st.error(f"**{title}**" + (f"\n\n{action}" if action else ""), icon=icon)
    if detail:
        with st.expander("Technical detail"):
            st.code(detail.strip(), language=None)


def fallback_warning(result) -> None:
    """Name the provider substitution when a lower-priority LLM answered.

    # WHY surface this at all — the answer worked?
    # Because during development a local Ollama will happily cover for a
    # rate-limited OpenRouter, so the app looks healthy while the ACTUAL
    # deployment target — Cloud, where no local model exists — is broken. This
    # banner is what stops that being discovered in production.
    """
    if not result.fell_back_from:
        return
    st.warning(
        f"**Answered by fallback provider `{result.provider}`** — "
        f"the primary provider failed.\n\n`{result.fell_back_from}`\n\n"
        f"On Streamlit Cloud there is no local model, so this question would "
        f"return an error until the primary provider recovers.", icon="⚠️")

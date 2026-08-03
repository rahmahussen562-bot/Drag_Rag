"""
ui — the presentation layer.

`streamlit_app.py` owns FLOW: which page is active and what the user asked. This
package owns APPEARANCE: CSS, badges, cards, status formatting, and one module per
page. Keeping them apart means page logic stays readable instead of being buried
under markup, and a styling change can never alter behaviour.

    ui/theme.py         the stylesheet + the badge vocabulary it defines
    ui/state.py         cached resource construction, session bootstrap, AppContext
    ui/sidebar.py       navigation, settings, live status panel
    ui/components/      source_card · trust_panel · pipeline_stepper · score_bars
    ui/pages/           one module per page, each exposing render(ctx)

# WHY re-export the old flat names?
# This package replaced a single `ui.py`. Re-exporting what that module exposed
# means the split cost zero call-site changes and can be verified by the existing
# suite rather than by reading every caller.
"""
from __future__ import annotations

from .components import (  # noqa: F401
    DISCLAIMER,
    answer_badges,
    error_box,
    fallback_warning,
    render_evaluation,
    render_sources,
    source_card,
)
from .theme import inject_css, pill, pills, stat_row  # noqa: F401

__all__ = [
    "inject_css", "pill", "pills", "stat_row",
    "source_card", "render_sources", "answer_badges", "render_evaluation",
    "error_box", "fallback_warning", "DISCLAIMER",
]

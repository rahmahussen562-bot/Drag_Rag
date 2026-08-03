"""ui.components — presentational building blocks, each independently testable."""
from __future__ import annotations

from .notices import DISCLAIMER, error_box, fallback_warning  # noqa: F401
from .pipeline_stepper import FIRST_LABEL, STAGE_LABELS, finalise, make_on_stage  # noqa: F401
from .score_bars import contributions, score_bar  # noqa: F401
from .source_card import render_sources, source_card  # noqa: F401
from .trust_panel import answer_badges, render_evaluation  # noqa: F401

__all__ = [
    "source_card", "render_sources",
    "answer_badges", "render_evaluation",
    "STAGE_LABELS", "FIRST_LABEL", "make_on_stage", "finalise",
    "score_bar", "contributions",
    "error_box", "fallback_warning", "DISCLAIMER",
]

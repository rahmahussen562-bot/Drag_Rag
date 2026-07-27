"""
app.py — backwards-compatible entry point.

# WHY does this file still exist?
# The application now lives in `streamlit_app.py`, because that is the filename
# Streamlit Community Cloud selects by default and the one the project spec
# requires. Anyone (or any bookmark, or any deployment config) still running
# `streamlit run app.py` should not break, so this delegates rather than
# duplicating a second copy of the UI.

Prefer:  streamlit run streamlit_app.py
"""
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "streamlit_app.py"

# Execute the real app in THIS module's namespace. Streamlit re-runs the script
# top-to-bottom on every interaction, and exec keeps that model intact — whereas
# `import streamlit_app` would be cached after the first run and the UI would
# freeze on its initial render.
exec(compile(_TARGET.read_text(encoding="utf-8"), str(_TARGET), "exec"), globals())

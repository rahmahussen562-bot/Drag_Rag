"""
stages.py — the tiny loader that lets the numbered pipeline files import each other.

# WHY does this file exist?
# Python module names may not begin with a digit:  `import 01_documents`  is a
# SyntaxError. But the course pipeline is easiest to READ when the folder listing
# mirrors the diagram (01_documents → 02_preprocessing → … → 07_prompting), and we
# also refuse to copy-paste code between stages.
#
# Those two goals collide. This loader resolves the collision: it imports a
# numbered file *by path* and caches the resulting module object, so a stage can
# do the moral equivalent of a normal import:
#
#     from stages import load
#     chunking = load("03_chunking")
#     df = chunking.build_chunks(documents)
#
# That is how every stage below reuses the previous ones instead of duplicating
# them, while each file still runs standalone (`python 03_chunking.py`).

Nothing else lives here — this module is deliberately ~40 lines of plumbing.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Project root = the folder this file sits in. Every stage is resolved relative
# to it, so the pipeline works no matter what the current working directory is
# (Streamlit Cloud, `python eval/run_eval.py`, a notebook, …).
ROOT = Path(__file__).resolve().parent

# Put the project root on sys.path so stage modules can `import llm_providers`,
# `import interactions`, etc. no matter which directory the process was started
# from (`python eval/run_eval.py`, Streamlit Cloud, a notebook in a subfolder).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cache: stage name -> imported module. A module must only ever execute ONCE per
# process, otherwise stage 04 would rebuild its embedding model every time a
# later stage asked for it.
_LOADED: dict[str, ModuleType] = {}


def load(stage: str) -> ModuleType:
    """Import a numbered pipeline file (e.g. "03_chunking") and return the module.

    Repeated calls return the same cached module object, exactly like `import`.
    """
    if stage in _LOADED:
        return _LOADED[stage]

    path = ROOT / f"{stage}.py"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline stage not found: {path}")

    # A module name starting with a digit is illegal in `import` syntax but is
    # perfectly legal as a key in sys.modules, so we register it under a safe
    # alias ("stage_03_chunking") to keep tracebacks and pickling sane.
    alias = f"stage_{stage}"
    if alias in sys.modules:
        _LOADED[stage] = sys.modules[alias]
        return _LOADED[stage]

    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so that circular-ish imports and dataclasses that look
    # themselves up in sys.modules resolve correctly.
    sys.modules[alias] = module
    spec.loader.exec_module(module)

    _LOADED[stage] = module
    return module


# Convenience aliases so callers can write `stages.documents` instead of
# `load("01_documents")` when they want the whole pipeline at once.
STAGE_NAMES = {
    "documents": "01_documents",
    "preprocessing": "02_preprocessing",
    "chunking": "03_chunking",
    "vector_representation": "04_vector_representation",
    "vector_store": "05_create_vector_store",
    "retrieval": "06_retrieve_context",
    "prompting": "07_prompting",
}


def __getattr__(name: str) -> ModuleType:
    """`stages.chunking` → load("03_chunking"). Lazy so nothing loads until used."""
    if name in STAGE_NAMES:
        return load(STAGE_NAMES[name])
    raise AttributeError(f"module 'stages' has no attribute {name!r}")


if __name__ == "__main__":
    print("Pipeline stages discoverable from", ROOT)
    for friendly, stage in STAGE_NAMES.items():
        exists = "OK " if (ROOT / f"{stage}.py").exists() else "MISSING"
        print(f"  [{exists}] {stage:28s} stages.{friendly}")

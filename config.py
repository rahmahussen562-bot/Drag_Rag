"""
config.py — deployment configuration, with defaults that match the shipped code.

# WHY does a config layer exist at all?
# So a customer can rebrand, retune and redeploy without touching Python. That is
# the difference between a demo and something a pharmacy IT department will accept.
#
# # WHY does every key have a compiled-in default?
# Because the eval harness must measure the SAME numbers the app runs, and a
# missing config file must never change behaviour. Delete config.yaml and every
# value below is exactly what the code used before this file existed — so the
# 26-check gate is unaffected by whether the file is present.
#
# # WHY is the disclaimer special-cased?
# Because a buyer removing it is a liability transfer, and the brief is explicit
# that it must not be possible to do that SILENTLY. Blanking it does not disable
# it: `load()` restores the mandatory text and records a warning that the UI
# surfaces. The app keeps running (bricking a deploy over a config typo helps
# nobody) but it cannot quietly ship without one.

Usage:
    from config import CONFIG
    CONFIG.retrieval.alpha        # 0.20
    CONFIG.product.disclaimer     # never empty
    CONFIG.warnings               # list[str] — surfaced in the UI
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PHARMA_CONFIG", ROOT / "config.yaml"))

#: The text that cannot be removed. Restored if a config blanks it.
MANDATORY_DISCLAIMER = (
    "Educational tool — not medical advice. Drug data can be incomplete or "
    "outdated. Always confirm with a pharmacist or prescriber."
)


@dataclass
class Product:
    name: str = "PhARMA RAG"
    tagline: str = "Grounded drug Q&A with cited sources"
    icon: str = "💊"
    accent: str = "#0F9D8E"
    disclaimer: str = MANDATORY_DISCLAIMER


@dataclass
class Retrieval:
    alpha: float = 0.20
    top_k: int = 5
    emb_floor: float = 0.25
    bm25_floor: float = 1.0
    retrieve_pool: int = 40
    field_boost: float = 0.20
    use_reranker: bool = False


@dataclass
class Generation:
    temperature: float = 0.0
    max_context_chars: int = 12_000
    providers: list = field(default_factory=list)


@dataclass
class Agents:
    default: str = "practitioner"


@dataclass
class Interactions:
    provider: str = "ddinter"
    enabled: bool = True


@dataclass
class Audit:
    enabled: bool = True
    redact: bool = True


@dataclass
class Config:
    product: Product = field(default_factory=Product)
    retrieval: Retrieval = field(default_factory=Retrieval)
    generation: Generation = field(default_factory=Generation)
    agents: Agents = field(default_factory=Agents)
    interactions: Interactions = field(default_factory=Interactions)
    audit: Audit = field(default_factory=Audit)

    #: Non-fatal problems found while loading. The UI renders these.
    warnings: list = field(default_factory=list)
    #: Where the values came from, for the health check.
    source: str = "defaults"


def _apply(section: Any, values: dict, warnings: list, prefix: str) -> None:
    """Overlay a dict onto a dataclass, ignoring unknown keys with a warning.

    # WHY warn rather than raise on an unknown key?
    # A typo'd key that silently does nothing is the worst outcome — the operator
    # believes they changed something. Warning surfaces it; raising would brick a
    # deploy over a stray line.
    """
    if not isinstance(values, dict):
        warnings.append(f"config: '{prefix}' should be a mapping — ignored.")
        return
    known = {f.name: f for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            warnings.append(f"config: unknown key '{prefix}.{key}' — ignored.")
            continue
        expected = known[key].type
        # Light coercion only: YAML gives str/int/float/bool/list already.
        try:
            if expected in ("float", float) and value is not None:
                value = float(value)
            elif expected in ("int", int) and value is not None:
                value = int(value)
            elif expected in ("bool", bool) and value is not None:
                value = bool(value)
        except (TypeError, ValueError):
            warnings.append(
                f"config: '{prefix}.{key}' = {value!r} is not a valid "
                f"{expected} — keeping the default.")
            continue
        setattr(section, key, value)


def load(path: Path = CONFIG_PATH) -> Config:
    """Read config.yaml over the compiled-in defaults. Never raises."""
    cfg = Config()

    if not path.exists():
        cfg.source = "defaults (no config.yaml)"
        return _enforce(cfg)

    try:
        import yaml
    except Exception:
        # Consistent with the rest of the project: an optional dependency that is
        # missing degrades to the documented default rather than taking the app
        # down. The defaults ARE the shipped configuration.
        cfg.warnings.append(
            "config: PyYAML not installed — using built-in defaults. "
            "`pip install pyyaml` to enable config.yaml.")
        cfg.source = "defaults (PyYAML missing)"
        return _enforce(cfg)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        cfg.warnings.append(f"config: {path.name} could not be parsed "
                            f"({type(exc).__name__}) — using defaults.")
        cfg.source = "defaults (parse error)"
        return _enforce(cfg)

    for name in ("product", "retrieval", "generation", "agents",
                 "interactions", "audit"):
        if name in raw:
            _apply(getattr(cfg, name), raw[name], cfg.warnings, name)

    cfg.source = str(path)
    return _enforce(cfg)


def _enforce(cfg: Config) -> Config:
    """Apply the rules a customer is not permitted to configure away."""
    # 1. The disclaimer is mandatory. Restore it, loudly.
    if not str(cfg.product.disclaimer or "").strip():
        cfg.product.disclaimer = MANDATORY_DISCLAIMER
        cfg.warnings.append(
            "config: product.disclaimer was blank and has been RESTORED. This "
            "field cannot be removed — a tool that informs clinical decisions "
            "ships with a disclaimer or it does not ship.")

    # 2. Floors and weights must stay in range, or abstention stops meaning
    #    anything. Out-of-range values are clamped, not silently accepted.
    r = cfg.retrieval
    for name, lo, hi in (("alpha", 0.0, 1.0), ("field_boost", 0.0, 5.0),
                         ("emb_floor", -1.0, 1.0)):
        value = getattr(r, name)
        if not (lo <= value <= hi):
            setattr(r, name, min(max(value, lo), hi))
            cfg.warnings.append(
                f"config: retrieval.{name} = {value} is outside [{lo}, {hi}] — "
                f"clamped to {getattr(r, name)}.")
    if r.top_k < 1:
        r.top_k = 1
        cfg.warnings.append("config: retrieval.top_k must be ≥ 1 — set to 1.")

    return cfg


#: The loaded configuration. Import this, not `load()`.
CONFIG = load()


def as_dict(section=None) -> dict:
    """Serialise the config (for the health check and the Architecture page)."""
    target = section or CONFIG
    out = {}
    for f in fields(target):
        value = getattr(target, f.name)
        out[f.name] = as_dict(value) if is_dataclass(value) else value
    return out


if __name__ == "__main__":
    import json
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"source: {CONFIG.source}\n")
    print(json.dumps(as_dict(), indent=2, ensure_ascii=False))
    if CONFIG.warnings:
        print("\nwarnings:")
        for w in CONFIG.warnings:
            print(f"  ⚠️  {w}")

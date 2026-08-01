"""
llm_providers.py — the LLM generation layer used by stage 07.

# WHY is this a separate module and not part of 07_prompting.py?
# Stage 07's job is to build a grounded prompt and verify the answer that comes
# back. WHICH backend produced that answer is an orthogonal, deployment-time
# concern. Keeping them apart means the prompt contract is testable without a
# network call, and adding a provider never touches prompt logic.
#
# ════════════════════════════════════════════════════════════════════════════
# PROVIDER PRIORITY (each falls through to the next on missing key OR failure)
#     1. OpenRouter   — one key, hundreds of models, has a genuinely free tier
#     2. Gemini       — Google AI Studio, generous free tier
#     3. Groq         — fastest inference available for free
#     4. Ollama       — local, private, no key; used in development
#     5. extractive   — no LLM at all: return the retrieved sources verbatim
# ════════════════════════════════════════════════════════════════════════════
#
# # WHY automatic fallback rather than one configured provider?
# Free tiers rate-limit, free model slugs get retired, and a Streamlit Cloud demo
# has to survive both without a redeploy. Cascading means a 429 from OpenRouter
# silently becomes a Gemini answer instead of an error page. Step 5 matters most:
# even with zero credentials the app still answers, using retrieved text directly,
# so the pipeline is never "broken" — only less fluent.
#
# ════════════════════════════════════════════════════════════════════════════
# SECRETS — NO API KEY IS EVER STORED IN CODE
# ════════════════════════════════════════════════════════════════════════════
# Keys are resolved at runtime, in this order, first hit wins:
#     1. st.secrets[...]        — Streamlit Cloud (Settings → Secrets)
#     2. os.environ[...]        — CI, Docker, shell exports
#     3. .env file              — local development (git-ignored)
# There is deliberately no fourth source and no default value: a key can only
# ever enter this process from the environment it runs in. `.env` and
# `.streamlit/secrets.toml` are both in .gitignore.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1
# Configuration resolution — the only place a secret is ever read.
# ─────────────────────────────────────────────────────────────────────────────
_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load .env into os.environ for local development. Never overwrites a real
    environment variable — an explicit export must always win over a file."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = HERE / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except Exception:
        # Minimal parser so python-dotenv stays an optional dependency.
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        except Exception:
            pass


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets → environment → .env."""
    # 1. Streamlit Cloud. Guarded because importing/streaming secrets raises when
    #    the module runs outside a Streamlit process (CLI, eval scripts, tests).
    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    # 2 + 3. Real environment, then .env.
    value = os.environ.get(name)
    if value:
        return value
    _load_dotenv_once()
    return os.environ.get(name, default)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2
# Provider definitions.
#
# WHY keep the default model slugs here rather than hard-coding them at the call
# site? Free slugs get retired without notice. Every one is overridable by secret
# (e.g. OPENROUTER_MODEL) so a dead model is a config change, not a code change.
# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter free-tier model chain, in measured preference order.
#
# # WHY A CHAIN AND NOT ONE SLUG?
# This is the failure that actually broke the app. The previous default,
# "meta-llama/llama-3.3-70b-instruct:free", was retired from the free tier and
# OpenRouter began returning:
#     HTTP 404 {"error":{"message":"This model is unavailable for free …"}}
# A single hard-coded slug therefore has a shelf life, and when it expires the
# provider looks broken even though the key is perfectly valid.
#
# Ordering was measured, not guessed (see the model benchmark in the commit that
# introduced this): each candidate was given the real RAG prompt and scored on
# whether it emitted [Source N] citations, kept the disclaimer, and grounded its
# answer in the supplied context.
#     nemotron-3-nano-30b   2 citations, 68% context overlap, 4.0s
#     gpt-oss-20b           2 citations, 66% overlap, 7.7s
#     ling-3.0-flash        1 citation,  84% overlap, 1.4s
#     gemma-4-26b           1 citation,  70% overlap, 12.7s
#     openrouter/free       1 citation,  61% overlap, 2.0s   (meta-slug, routes)
# Setting OPENROUTER_MODEL in secrets overrides the chain entirely.
OPENROUTER_MODEL_CHAIN = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
    "inclusionai/ling-3.0-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
]

DEFAULT_MODELS = {
    "openrouter": OPENROUTER_MODEL_CHAIN[0],
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2:latest",
}

# The cascade order. OpenRouter first per project requirement.
PROVIDER_ORDER = ["openrouter", "gemini", "groq", "ollama"]

# Deterministic by default: this is a factual assistant, not a creative writer.
# temperature=0 also makes the evaluation in Phase 7 reproducible.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 90


@dataclass
class LLMResponse:
    """What came back, and which provider actually produced it."""
    text: str
    provider: str
    model: str
    attempts: list          # [(provider, "ok" | "error: …"), …] — shown in eval mode

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


class ProviderError(RuntimeError):
    """A provider was configured but the call failed."""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3
# One calling convention per provider.
#
# Each returns plain text or raises. They are deliberately written against raw
# `requests` rather than four vendor SDKs: the SDKs would add ~80 MB to the
# deployment, disagree about async, and pin conflicting httpx versions — for
# three endpoints that are all just JSON over HTTPS.
# ─────────────────────────────────────────────────────────────────────────────
def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    import requests
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if response.status_code >= 400:
        # Surface the provider's own message — "insufficient credits" and
        # "model not found" need completely different fixes by the operator.
        detail = response.text[:300].replace("\n", " ")
        raise ProviderError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def _extract_chat_text(data: dict, provider: str) -> str:
    """Pull the assistant text out of an OpenAI-shaped response, or raise.

    # WHY not just data["choices"][0]["message"]["content"]?
    # Because that line raises KeyError/IndexError/TypeError on three shapes that
    # OpenRouter genuinely returns in production, and a bare KeyError tells the
    # operator nothing about how to fix it:
    #   • {"error": {...}} with HTTP 200      — upstream provider failed mid-request
    #   • {"choices": []}                     — request accepted, nothing generated
    #   • content: null                       — hit the length cap or a safety stop
    # Each is turned into a ProviderError carrying the real reason, so the model
    # chain below can react and the UI can show something actionable.
    """
    if isinstance(data.get("error"), dict):
        err = data["error"]
        raise ProviderError(f"{provider} error {err.get('code', '')}: "
                            f"{err.get('message', 'unknown')}".strip())

    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"{provider} returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None or not str(content).strip():
        reason = choices[0].get("finish_reason", "unknown")
        raise ProviderError(f"{provider} returned empty content (finish_reason={reason})")
    return str(content)


# Errors that mean "this MODEL is unusable right now" rather than "this PROVIDER
# is unusable". Only these justify advancing the chain; an auth failure must not,
# because retrying a bad key against five models is five pointless round trips.
_MODEL_LEVEL_SIGNS = (
    "404", "429", "400",
    "unavailable for free", "not a valid model", "no endpoints found",
    "rate limit", "temporarily rate-limited", "no instances available",
)


def _is_model_level_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "401" in text or "403" in text or "no auth credentials" in text:
        return False                      # bad/absent key — the chain cannot help
    return any(sign in text for sign in _MODEL_LEVEL_SIGNS)


def openrouter_models() -> list[str]:
    """Models to try, in order. An explicit OPENROUTER_MODEL secret wins outright."""
    configured = get_secret("OPENROUTER_MODEL")
    return [configured] if configured else list(OPENROUTER_MODEL_CHAIN)


def call_openrouter(prompt: str, model: str = "", temperature: float = DEFAULT_TEMPERATURE,
                    timeout: int = DEFAULT_TIMEOUT) -> str:
    """Call OpenRouter, walking the model chain past retired/rate-limited slugs.

    Staying inside OpenRouter for model-level failures is deliberate: OpenRouter
    is the designated primary provider, so a dead free slug must not be allowed
    to demote the whole request to a different provider.
    """
    key = get_secret("OPENROUTER_API_KEY")
    if not key:
        raise ProviderError("OPENROUTER_API_KEY not set")

    host = get_secret("OPENROUTER_HOST", "https://openrouter.ai/api/v1")
    headers = {
        "Authorization": f"Bearer {key}",
        # Optional attribution headers OpenRouter uses for its free-tier ranking.
        "HTTP-Referer": "https://github.com/Drag_Rag",
        "X-Title": "Drag_Rag",
    }

    candidates = [model] if model else openrouter_models()
    last_error: Optional[Exception] = None

    for candidate in candidates:
        try:
            data = _post_json(
                f"{host}/chat/completions",
                {"model": candidate,
                 "messages": [{"role": "user", "content": prompt}],
                 "temperature": temperature},
                headers, timeout)
            text = _extract_chat_text(data, "openrouter")
            _LAST_OPENROUTER_MODEL["value"] = candidate   # for accurate UI reporting
            return text
        except Exception as exc:
            last_error = exc
            if _is_model_level_error(exc) and len(candidates) > 1:
                continue                  # retired or rate-limited slug → next model
            raise

    raise ProviderError(f"all OpenRouter models failed; last error: {last_error}")


# Which chain entry actually answered, so the UI reports the real model rather
# than the first candidate we intended to use.
_LAST_OPENROUTER_MODEL = {"value": ""}


def call_gemini(prompt: str, model: str = "", temperature: float = DEFAULT_TEMPERATURE,
                timeout: int = DEFAULT_TIMEOUT) -> str:
    key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY not set")
    model = model or get_secret("GEMINI_MODEL", DEFAULT_MODELS["gemini"])
    # Gemini takes the key as a query parameter and uses its own request schema.
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    data = _post_json(
        url,
        {"contents": [{"parts": [{"text": prompt}]}],
         "generationConfig": {"temperature": temperature}},
        {"Content-Type": "application/json"},
        timeout)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        # Most often a safety block — the response has no candidates at all.
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
        raise ProviderError(f"Gemini returned no text ({reason})") from exc


def call_groq(prompt: str, model: str = "", temperature: float = DEFAULT_TEMPERATURE,
              timeout: int = DEFAULT_TIMEOUT) -> str:
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise ProviderError("GROQ_API_KEY not set")
    model = model or get_secret("GROQ_MODEL", DEFAULT_MODELS["groq"])
    # Groq exposes an OpenAI-compatible surface, so the payload matches OpenRouter's.
    data = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}],
         "temperature": temperature},
        {"Authorization": f"Bearer {key}"},
        timeout)
    return _extract_chat_text(data, "groq")


def call_ollama(prompt: str, model: str = "", temperature: float = DEFAULT_TEMPERATURE,
                timeout: int = 180) -> str:
    """Local generation. No key — privacy-preserving and free, for development."""
    host = get_secret("OLLAMA_HOST", "http://localhost:11434")
    model = model or resolve_ollama_model()
    data = _post_json(
        f"{host}/api/generate",
        {"model": model, "prompt": prompt, "stream": False,
         "options": {"temperature": temperature}},
        {}, timeout)
    return data["response"]


_CALLERS = {
    "openrouter": call_openrouter,
    "gemini": call_gemini,
    "groq": call_groq,
    "ollama": call_ollama,
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4
# Availability checks — used by the sidebar so the user can see what is wired up
# BEFORE asking a question.
# ─────────────────────────────────────────────────────────────────────────────
_KEY_NAMES = {
    "openrouter": ["OPENROUTER_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
}


def provider_configured(provider: str) -> bool:
    """True if this provider has credentials (or, for Ollama, is reachable)."""
    if provider == "ollama":
        return ollama_available()
    return any(get_secret(k) for k in _KEY_NAMES.get(provider, []))


# Health-probe cache: (timestamp, models). See ollama_available() for why.
_OLLAMA_PROBE: tuple[float, list[str]] = (0.0, [])
_OLLAMA_PROBE_TTL = 30.0        # seconds
_OLLAMA_PROBE_TIMEOUT = 6       # seconds


def ollama_models(force: bool = False) -> list[str]:
    """Model names available on the local Ollama host, cached for a few seconds.

    # WHY cache, and WHY a 6-second timeout?
    # Both were bugs found by measurement. A cold Ollama takes ~2.4s to answer
    # /api/tags, so the original 2-second timeout made local generation
    # NONDETERMINISTIC — the same question would generate an answer one run and
    # silently fall back to extractive the next. And because the probe is hit by
    # both llm_status() (sidebar) and generate() (every question), an uncached
    # probe added several seconds of latency per request. A short TTL keeps the
    # status fresh enough to notice Ollama starting or stopping.
    """
    global _OLLAMA_PROBE
    import time as _time

    age = _time.time() - _OLLAMA_PROBE[0]
    if not force and age < _OLLAMA_PROBE_TTL and _OLLAMA_PROBE[0] > 0:
        return _OLLAMA_PROBE[1]

    models: list[str] = []
    try:
        import requests
        host = get_secret("OLLAMA_HOST", "http://localhost:11434")
        response = requests.get(f"{host}/api/tags", timeout=_OLLAMA_PROBE_TIMEOUT)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
    except Exception:
        models = []

    _OLLAMA_PROBE = (_time.time(), models)
    return models


def ollama_available() -> bool:
    """True if a local Ollama server is reachable with at least one model."""
    return bool(ollama_models())


def resolve_ollama_model() -> str:
    """The configured local model, or the first available one as a fallback.

    WHY fall back? A machine with `llama3.2` but not the configured default should
    still generate rather than refuse — any local model beats no generation.
    """
    models = ollama_models()
    wanted = get_secret("OLLAMA_MODEL", DEFAULT_MODELS["ollama"])
    if not models:
        return wanted
    if wanted in models:
        return wanted
    stem = wanted.split(":")[0]
    for name in models:
        if name.split(":")[0] == stem:
            return name
    return models[0]


def model_for(provider: str) -> str:
    """The model slug this provider would actually use right now."""
    if provider == "ollama":
        return resolve_ollama_model()
    if provider == "openrouter":
        # Report the chain entry that last succeeded; before the first call, the
        # one we would try first. Reporting a slug that never answered would make
        # the sidebar lie about which model produced the text on screen.
        return _LAST_OPENROUTER_MODEL["value"] or openrouter_models()[0]
    override = {"gemini": "GEMINI_MODEL", "groq": "GROQ_MODEL"}.get(provider, "")
    return get_secret(override, DEFAULT_MODELS.get(provider, "")) if override else ""


def available_providers() -> list[str]:
    """Configured providers, in cascade order."""
    return [p for p in PROVIDER_ORDER if provider_configured(p)]


def llm_status() -> dict:
    """Everything the sidebar needs to render LLM state."""
    available = available_providers()
    active = available[0] if available else None
    return {
        "active": active,
        "model": model_for(active) if active else None,
        "available": available,
        "all": {p: provider_configured(p) for p in PROVIDER_ORDER},
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5
# The cascade — the single function stage 07 calls.
# ─────────────────────────────────────────────────────────────────────────────
def generate(prompt: str, prefer: Optional[str] = None,
             temperature: float = DEFAULT_TEMPERATURE,
             timeout: int = DEFAULT_TIMEOUT) -> LLMResponse:
    """Try each configured provider in order; return the first success.

    `prefer` pins a provider to the front (the UI's provider selector). It does
    not disable the others — a pinned provider that fails still falls through, so
    a user's preference can never take the app down.

    Never raises: if everything fails, returns an LLMResponse with empty text and
    the full attempt log, and the caller falls back to an extractive answer.
    """
    order = list(PROVIDER_ORDER)
    if prefer and prefer in order:
        order.remove(prefer)
        order.insert(0, prefer)

    attempts: list[tuple[str, str]] = []
    for provider in order:
        if not provider_configured(provider):
            attempts.append((provider, "not configured"))
            continue
        try:
            text = _CALLERS[provider](prompt, temperature=temperature, timeout=timeout)
            if text and text.strip():
                attempts.append((provider, "ok"))
                return LLMResponse(text=text, provider=provider,
                                   model=model_for(provider), attempts=attempts)
            attempts.append((provider, "empty response"))
        except Exception as exc:
            # Record and continue — this is the whole point of the cascade.
            attempts.append((provider, f"{type(exc).__name__}: {exc}"[:180]))

    return LLMResponse(text="", provider="", model="", attempts=attempts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI:  python llm_providers.py  ["your prompt"]
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("  LLM PROVIDER LAYER")
    print("=" * 78)

    status = llm_status()
    print("Provider status (priority order):")
    for provider in PROVIDER_ORDER:
        configured = status["all"][provider]
        mark = "🟢" if configured else "⚪"
        note = f"model={model_for(provider)}" if configured else "no credentials"
        print(f"  {mark} {provider:<11s} {note}")

    if not status["available"]:
        print("\nNo provider configured → the app answers extractively (retrieved")
        print("sources verbatim, still cited). Set a key in .env or Streamlit secrets:")
        print("  OPENROUTER_API_KEY / GEMINI_API_KEY / GROQ_API_KEY")
        raise SystemExit(0)

    print(f"\nActive: {status['active']} · {status['model']}")

    prompt = " ".join(sys.argv[1:]) or "Reply with exactly the word: OK"
    print(f"\nTest prompt: {prompt!r}")
    result = generate(prompt)
    print("\nAttempt log:")
    for provider, outcome in result.attempts:
        print(f"  {provider:<11s} {outcome}")
    if result.ok:
        print(f"\nAnswered by {result.provider} ({result.model}):")
        print(result.text[:600])
    else:
        print("\nAll providers failed — extractive fallback would be used.")

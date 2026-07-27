"""
02_preprocessing.py — STAGE 2 of the RAG pipeline:  **Preprocessing**

    Raw Documents  →  >>> Preprocessing <<<  →  Chunking  →  Vector Representation
    →  Vector Store  →  Context Retrieval  →  Prompt Building  →  LLM Generation  →  Streamlit UI

# WHY does this stage exist?
# Lexical retrieval (BM25 / TF-IDF in stages 04-06) matches *literal tokens*. Without
# normalisation, "Doses", "doses" and "dosing" are three unrelated words and a query
# for one will not find the others. Cleaning collapses those variants so the sparse
# retriever can actually do its job.
#
# # WHY IS THIS THE MOST DANGEROUS FILE IN THE PROJECT?
# Because the textbook "clean your text" recipe — lowercase, strip punctuation,
# drop stopwords, drop numbers, stem — is *actively harmful* on medical text:
#
#     "No known hepatotoxicity at therapeutic doses (500 mg). Not recommended
#      with warfarin."
#
#   • dropping stopwords deletes "No" and "Not"  → the sentence now says the
#     OPPOSITE of what the label says;
#   • dropping numbers deletes "500"             → the dose is gone;
#   • stemming mangles "hepatotoxicity"→"hepatotox".
#
# So this file treats preprocessing as a SAFETY decision, not a tuning knob. Two
# guarantees are hard-coded into the default profile:
#   1. PROTECTED_NEGATION words are never removed and never stemmed.
#   2. Numbers are never removed.
# The aggressive profile is kept only so the damage can be *demonstrated* (run this
# file standalone to see it) — it is never used by the pipeline.
#
# NOTE: only the *lexical* index consumes this cleaned text. The dense embedding
# model in stage 04 is deliberately fed the RAW text, because transformer models
# are trained on natural language and are hurt, not helped, by stopword removal.

Run standalone:
    python 02_preprocessing.py         # side-by-side profile comparison + safety assertions
"""
from __future__ import annotations

import re
import string

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1
# Load NLTK's tokenizer / stopwords / lemmatizer, with fallbacks at every step.
#
# WHY so much defensive code around a simple import?
# Streamlit Cloud containers are rebuilt from scratch and NLTK's corpora are
# downloaded at runtime, not installed by pip. If that download is blocked the
# naive version of this file raises LookupError and the whole app dies. Instead we
# fall back: NLTK tokenizer → str.split(); NLTK stopwords → scikit-learn's list.
# The pipeline gets slightly worse, and it keeps running.
# ─────────────────────────────────────────────────────────────────────────────
NLTK_AVAILABLE = True
try:
    import nltk

    for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass  # offline is fine — the fallbacks below cover it

    from nltk.stem import PorterStemmer, WordNetLemmatizer

    _STEMMER = PorterStemmer()
    _LEMMATIZER = WordNetLemmatizer()

    try:
        _STOP = set(nltk.corpus.stopwords.words("english"))
        STOPWORD_SOURCE = "nltk"
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _STOP = set(ENGLISH_STOP_WORDS)
        STOPWORD_SOURCE = "sklearn-fallback"

    def _tokenize(text: str) -> list[str]:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            return text.split()

    def _stem(token: str) -> str:
        try:
            return _STEMMER.stem(token)
        except Exception:
            return token

    def _lemmatize(token: str) -> str:
        try:
            return _LEMMATIZER.lemmatize(token)
        except Exception:
            return token

except Exception:  # NLTK not installed at all
    NLTK_AVAILABLE = False
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    _STOP = set(ENGLISH_STOP_WORDS)
    STOPWORD_SOURCE = "sklearn-fallback"

    def _tokenize(text: str) -> list[str]:
        return text.split()

    def _stem(token: str) -> str:
        return token

    def _lemmatize(token: str) -> str:
        return token


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2
# The protected-negation list — the single most important constant in the repo.
#
# WHY? A stopword list is built for topic search ("find documents about X"), where
# "not" carries no topic signal. Clinical text is the opposite: negation IS the
# content. "No known interactions" and "known interactions" are opposite facts and
# must never collapse to the same token sequence.
# ─────────────────────────────────────────────────────────────────────────────
PROTECTED_NEGATION = {
    "no", "not", "nor", "never", "none",
    "neither", "nobody", "nothing", "nowhere",
    # contraction forms NLTK's tokenizer produces from "don't" / "cannot"
    "n't", "cannot", "without",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3
# The modular cleaner. Every transformation is an independent switch so that
# "profiles" (step 4) are just different keyword arguments — which is what makes
# the safety comparison in __main__ possible.
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_url: bool = True,
    remove_punct: bool = False,
    remove_num: bool = False,
    normalize_space: bool = True,
    remove_stop_words: bool = False,
    preserve_negation: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
) -> str:
    """Normalise text for lexical retrieval.

    Defaults are the SAFE settings: punctuation and numbers are kept, negation is
    protected. Callers opt into anything more aggressive explicitly.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 3a — URLs first, BEFORE lowercasing, so the regex sees the original scheme.
    #      URLs are pure noise for retrieval: no drug question is answered by a link.
    if remove_url:
        text = re.sub(r"https?://\S+", "", text)

    # 3b — Case folding. "Metformin" and "metformin" must be one token.
    if lowercase:
        text = text.lower()

    # 3c — Punctuation. OFF by default.
    #      WHY? "500 mg/day" and "(500 mg)" lose their structure without it, and
    #      hyphenated drug names ("co-trimoxazole") become two tokens.
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))

    # 3d — Numbers. OFF by default, and it should stay that way.
    #      WHY? Deleting digits deletes every dose, frequency and threshold in the
    #      corpus. This switch exists only to demonstrate the failure.
    if remove_num:
        text = re.sub(r"\b\d+\b", "", text)

    if normalize_space:
        text = re.sub(r"\s+", " ", text).strip()

    # 3e — Token-level work. Only tokenize if some token operation was requested;
    #      tokenizing is the slow part and we run it over ~35k chunks.
    if remove_stop_words or use_stemming or use_lemmatization:
        tokens = _tokenize(text)

        if remove_stop_words:
            tokens = [
                t for t in tokens
                if t not in _STOP or (preserve_negation and t in PROTECTED_NEGATION)
            ]

        if use_stemming:
            # Protected words bypass the stemmer so "none" cannot become "none"→"non".
            tokens = [t if t in PROTECTED_NEGATION else _stem(t) for t in tokens]

        if use_lemmatization:
            tokens = [t if t in PROTECTED_NEGATION else _lemmatize(t) for t in tokens]

        text = " ".join(tokens)

    if normalize_space:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4
# Named profiles — bundles of the switches above, ordered gentle → destructive.
#
# WHY keep `aggressive_stemmed` at all if it is unsafe?
# Because "this is unsafe" is a claim, and a claim you can run is worth more than a
# comment. The standalone run below prints all four side by side and asserts that
# the default profile preserves negation and numbers while the aggressive one
# destroys them. That assertion is a regression test on the project's safety rule.
# ─────────────────────────────────────────────────────────────────────────────
PROFILES: dict[str, dict] = {
    "minimal_clean": dict(
        lowercase=True, remove_url=True, normalize_space=True,
    ),
    "stopword_reduced": dict(
        lowercase=True, remove_url=True, normalize_space=True,
        remove_stop_words=True, preserve_negation=True,
    ),
    "aggressive_stemmed": dict(   # ⚠️ UNSAFE — demonstration only, never used
        lowercase=True, remove_url=True, remove_punct=True, remove_num=True,
        normalize_space=True, remove_stop_words=True, preserve_negation=False,
        use_stemming=True,
    ),
    "readable_lemmatized": dict(  # ✅ the pipeline default
        lowercase=True, remove_url=True, normalize_space=True,
        remove_stop_words=True, preserve_negation=True, use_lemmatization=True,
    ),
}

# The profile the whole pipeline uses. Chosen because it is the most aggressive
# setting that still passes the safety assertions at the bottom of this file:
# it removes stopword noise and collapses plural/verb forms (which measurably
# helps BM25) while keeping every number and every negation intact.
DEFAULT_PROFILE_NAME = "readable_lemmatized"
DEFAULT_PROFILE = PROFILES[DEFAULT_PROFILE_NAME]


def clean(text: str, profile: str | dict = DEFAULT_PROFILE_NAME) -> str:
    """Apply a named profile (or a raw kwargs dict) — the pipeline's entry point."""
    kwargs = PROFILES[profile] if isinstance(profile, str) else profile
    return preprocess_text(text, **kwargs)


def clean_many(texts, profile: str | dict = DEFAULT_PROFILE_NAME) -> list[str]:
    """Vectorised convenience wrapper used by stages 04/05 over whole corpora."""
    kwargs = PROFILES[profile] if isinstance(profile, str) else profile
    return [preprocess_text(t, **kwargs) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5
# Executable proof that the safety rules hold. Imported by the test in stage 07
# and by eval/, so the guarantee is checked, not just asserted in a docstring.
# ─────────────────────────────────────────────────────────────────────────────
SAFETY_SENTENCE = ("Paracetamol — side_effects: No known hepatotoxicity at therapeutic "
                   "doses (500 mg). Not recommended with warfarin.")


def safety_report(sentence: str = SAFETY_SENTENCE) -> dict:
    """For each profile, report whether negation and numbers survived."""
    report = {}
    for name, kwargs in PROFILES.items():
        out = preprocess_text(sentence, **kwargs)
        report[name] = {
            "output": out,
            "keeps_negation": ("no" in out.split()) and ("not" in out.split()),
            "keeps_numbers": bool(re.search(r"\b500\b", out)),
        }
    return report


def assert_default_profile_is_safe() -> None:
    """Raise AssertionError if the default profile ever stops being medically safe."""
    r = safety_report()[DEFAULT_PROFILE_NAME]
    assert r["keeps_negation"], (
        f"SAFETY REGRESSION: profile '{DEFAULT_PROFILE_NAME}' dropped a negation word. "
        f"Got: {r['output']!r}")
    assert r["keeps_numbers"], (
        f"SAFETY REGRESSION: profile '{DEFAULT_PROFILE_NAME}' dropped a dosage number. "
        f"Got: {r['output']!r}")


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("  STAGE 2 — PREPROCESSING")
    print("=" * 78)
    print(f"NLTK available : {NLTK_AVAILABLE}")
    print(f"Stopword source: {STOPWORD_SOURCE}  ({len(_STOP)} words)")
    print(f"Protected      : {sorted(PROTECTED_NEGATION)}")
    print(f"Pipeline uses  : {DEFAULT_PROFILE_NAME}\n")

    print("Original:")
    print(" ", SAFETY_SENTENCE, "\n")

    report = safety_report()
    for name, r in report.items():
        flag = "✅ SAFE  " if (r["keeps_negation"] and r["keeps_numbers"]) else "⚠️  UNSAFE"
        print(f"{flag} {name}")
        print(f"          {r['output']}")
        print(f"          negation kept={r['keeps_negation']}  numbers kept={r['keeps_numbers']}\n")

    assert_default_profile_is_safe()
    print("Safety assertions passed: the default profile preserves negation AND numbers.")
    print("\nNote: stage 04 feeds RAW text to the embedding model — this cleaning is")
    print("applied only to the lexical (BM25 / TF-IDF) index.")

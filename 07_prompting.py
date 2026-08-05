"""
07_prompting.py — STAGE 7:  **Prompt Building  →  LLM Generation**

    Raw Documents  →  Preprocessing  →  Chunking  →  Vector Representation  →  Vector Store
    →  Context Retrieval  →  >>> Prompt Building → LLM Generation <<<  →  Streamlit UI

# WHY does this stage exist?
# Retrieval (stage 06) found trustworthy text. This stage is the CONTRACT that
# stops the model from wandering away from it. Without these rules an LLM handed
# context will still answer from its pretraining, blend the two silently, and
# produce something fluent, plausible and unverifiable.
#
# Three mechanisms, in order of how much they matter:
#
#   1. NUMBERED SOURCE BLOCKS. Context is packed as "[Source 1] (drug=…, field=…)"
#      rather than concatenated prose. # WHY? A citation needs something to point
#      AT. Numbering gives the model a symbol per chunk, and — crucially — gives
#      *us* a way to mechanically verify afterwards that every [Source N] it
#      emitted refers to a chunk we actually supplied. An uncited claim becomes
#      visibly uncited instead of blending into the paragraph.
#
#   2. THE SYSTEM PROMPT. Six rules, carried over unchanged from the notebooks
#      because they were built around this exact corpus. Rules 5 and 6 are the
#      medical-safety ones: preserve numbers and negation verbatim, and refuse
#      rather than guess.
#
#   3. VERIFICATION AFTER GENERATION. A prompt rule is a request, not a guarantee.
#      `verify_answer()` mechanically checks the returned text for citations, for
#      the disclaimer, and — most importantly — for citation numbers that point at
#      sources that do not exist. This is what makes "the answer uses retrieved
#      context" a checkable claim rather than a hope.
#
# # WHY DOES THE EXTRACTIVE FALLBACK EXIST?
# If no LLM is reachable, the honest behaviour is not an error page: it is to show
# the retrieved sources directly. The answer is then less fluent but MORE grounded
# (it is quotation, not generation), still cited, and still carries the
# disclaimer. The pipeline degrades in quality, never in trustworthiness.

Run standalone:
    python 07_prompting.py                              # full demo incl. refusals
    python 07_prompting.py "how does metformin work?"
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import load  # noqa: E402

import llm_providers  # noqa: E402

retrieval = load("06_retrieve_context")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1
# The system prompt — the behavioural contract.
#
# Preserved verbatim from the notebooks. Do not "tidy" this text: the numbered,
# imperative phrasing is what small models actually follow. Softening rule 1 to
# "try to use the context" measurably increases outside-knowledge leakage.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a pharmaceutical information assistant. Answer drug-related
questions using ONLY the provided context sources.

RULES:
1. Base your answer STRICTLY on the provided context. Do not use outside knowledge.
2. Cite sources inline using [Source N].
3. If the context does not contain enough information, say so explicitly.
4. NEVER provide medical advice. Always include: "This is not medical advice.
   Consult a healthcare professional for medical decisions."
5. Preserve exact numbers (dosages, frequencies) and negation words (no, not, never).
6. If the question cannot be answered from the context, refuse with an explanation."""

DISCLAIMER = ("This is not medical advice. Consult a healthcare professional for "
              "medical decisions.")

# How much retrieved text may enter one prompt. # WHY cap it? Every provider has a
# context limit, and a prompt that overflows is silently truncated FROM THE END —
# which would cut off the question itself and produce a confidently wrong answer.
MAX_CONTEXT_CHARS = 12_000


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2
# Pack retrieved chunks into numbered, citable source blocks.
# ─────────────────────────────────────────────────────────────────────────────
def pack_context(chunks, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format retrieved chunks as the numbered context block the prompt expects.

    Source numbering is 1-based and positional: [Source 1] is chunks[0]. The UI
    renders its citation list from the SAME list in the SAME order, which is what
    keeps what the model cites and what the user sees in agreement.
    """
    parts, used = [], 0
    for rank, chunk in enumerate(chunks, 1):
        block = (f"[Source {rank}] (drug={chunk.drug}, field={chunk.field}, "
                 f"score={chunk.score:.3f})\n{chunk.text}")
        if used + len(block) > max_chars and parts:
            break                       # keep whole sources; never cut one in half
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_prompt(query: str, context: str,
                 system_prompt: Optional[str] = None) -> str:
    """Assemble the full grounded prompt: rules + context + question.

    # WHY does the context come BEFORE the question?
    # Attention degrades in the middle of long inputs ("lost in the middle").
    # Putting the rules first and the question last places both of the things the
    # model must obey at the high-attention edges, with the bulk material between.

    `system_prompt` lets a persona (see agents/) swap the RULES block for its own
    while keeping this layout — the part that was tuned — identical. It defaults to
    SYSTEM_PROMPT, so every existing caller is unaffected.
    """
    return (f"{system_prompt or SYSTEM_PROMPT}\n\n"
            f"---\nCONTEXT:\n{context}\n---\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (include [Source N] citations and the medical disclaimer):")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3
# Post-processing.
# ─────────────────────────────────────────────────────────────────────────────
def extract_answer(raw: str) -> str:
    """Strip reasoning-model scratchpads (DeepSeek-R1 etc.) from the response.

    WHY? Those models emit <think>…</think> containing their internal reasoning,
    which is long, often contradicts the final answer, and must never be shown as
    if it were the answer. If stripping leaves nothing, we return the original
    rather than an empty string.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return cleaned or raw.strip()


def ensure_disclaimer(text: str) -> str:
    """Append the medical disclaimer if the model forgot rule 4.

    WHY enforce in code rather than trusting the prompt? Because it is a hard
    requirement, and small models drop it perhaps one time in five.
    """
    return text if "not medical advice" in text.lower() else f"{text}\n\n{DISCLAIMER}"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4
# Verification — turn "the answer is grounded" into a measurement.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Verification:
    citations: list           # source numbers the answer cited, e.g. [1, 3]
    n_sources: int            # how many were supplied
    has_disclaimer: bool
    invalid_citations: list   # cited numbers with no such source → fabrication
    overlap: float            # word overlap between answer and supplied context

    @property
    def grounded(self) -> bool:
        """Grounded = cites at least one REAL source and invents none."""
        return bool(self.citations) and not self.invalid_citations

    def summary(self) -> str:
        bits = [f"{len(self.citations)}/{self.n_sources} sources cited",
                f"disclaimer {'yes' if self.has_disclaimer else 'NO'}",
                f"context overlap {self.overlap:.0%}"]
        if self.invalid_citations:
            bits.append(f"INVALID citations {self.invalid_citations}")
        return " · ".join(bits)


_STOPWORDS_FOR_OVERLAP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "with",
    "this", "that", "it", "as", "be", "by", "on", "at", "from", "not", "no",
}


def verify_answer(answer: str, chunks) -> Verification:
    """Mechanically check the answer against the sources it was given.

    `overlap` is the fraction of the answer's content words that also appear in
    the supplied context. # WHY measure it? Citations can be decorative — a model
    can write "[Source 1]" after a sentence it invented. Low overlap alongside
    confident citations is the fingerprint of exactly that failure.
    """
    cited = sorted({int(n) for n in re.findall(r"\[Source\s+(\d+)\]", answer)})
    valid = set(range(1, len(chunks) + 1))
    invalid = [n for n in cited if n not in valid]

    context_words = set()
    for chunk in chunks:
        context_words |= set(re.findall(r"[a-z0-9]+", chunk.text.lower()))

    answer_words = [w for w in re.findall(r"[a-z0-9]+", answer.lower())
                    if w not in _STOPWORDS_FOR_OVERLAP and len(w) > 2]
    overlap = (sum(1 for w in answer_words if w in context_words) / len(answer_words)
               if answer_words else 0.0)

    return Verification(
        citations=[n for n in cited if n in valid],
        n_sources=len(chunks),
        has_disclaimer="not medical advice" in answer.lower(),
        invalid_citations=invalid,
        overlap=overlap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5
# Answers for the cases where we deliberately do NOT generate.
# ─────────────────────────────────────────────────────────────────────────────
def refusal_message(outcome, n_entities: int) -> str:
    """Turn a retrieval abstention into a specific, actionable refusal.

    # WHY not one generic "I don't know"?
    # A refusal that names WHAT it does not know, and suggests the nearest thing it
    # does, is a usable answer. "aspirin → did you mean acetylsalicylic?" resolves
    # the user's actual problem; "I cannot help" does not.
    """
    if outcome.status == "unknown_entity":
        names = ", ".join(f"“{t}”" for t in outcome.unknown)
        message = (f"I don't have information on {names} in my database "
                   f"({n_entities} sources), so I won't guess.")
        if outcome.suggestions:
            hint = "; ".join(f"“{t}” → did you mean **{s}**?" for t, s in outcome.suggestions)
            message += f"\n\nClosest match: {hint}"
        else:
            message += ("\n\nTip: I index **generic** drug names (e.g. *acetaminophen*, "
                        "not *Tylenol*). You can also upload your own documents.")
    elif outcome.status == "no_relevant_context":
        found = ", ".join(sorted(outcome.matched)) or "that source"
        message = (f"I found **{found}** in my database, but nothing relevant enough "
                   f"to that specific question to answer confidently. I won't guess.")
    else:  # no_entity
        message = ("Please name a drug in your question — e.g. *“What are the side "
                   "effects of metformin?”* I answer only from my indexed sources "
                   "and won't guess.")
    return f"{message}\n\n{DISCLAIMER}"


def extractive_answer(chunks) -> str:
    """No-LLM fallback: present the retrieved sources themselves.

    This is quotation, not generation — nothing can be invented here. It stays
    cited and disclaimed, so the product contract holds even with zero credentials.
    """
    lines = ["_(No LLM connected — showing the most relevant sourced facts directly "
             "from the retrieved context.)_\n"]
    for rank, chunk in enumerate(chunks, 1):
        # Strip the "Name — field: " prefix stage 03 added; it is redundant here
        # because the heading already shows both.
        body = re.sub(r"^.*? — .*?:\s*", "", chunk.text).strip()
        snippet = body[:400] + ("…" if len(body) > 400 else "")
        lines.append(f"**[Source {rank}] {chunk.title} — "
                     f"{chunk.field.replace('_', ' ')}**  \n{snippet}")
    lines.append(f"\n{DISCLAIMER}")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6
# The full answer object and the orchestration that produces it.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Answer:
    """Everything the UI, the evaluator and the debugger need about one answer."""
    text: str
    chunks: list                       # the retrieved sources, in citation order
    mode: str                          # "llm" | "extractive" | "llm_failed" | "refused"
    status: str                        # the retrieval status behind it
    agent: str = ""                    # persona that produced it ("" = no persona)
    dropped_fields: tuple = ()         # fields a persona's allowlist removed
    provider: str = ""
    model: str = ""
    prompt: str = ""                   # the exact prompt sent (evaluation mode)
    context: str = ""                  # the exact context packed into it
    attempts: list = dc_field(default_factory=list)
    verification: Optional[Verification] = None
    llm_error: str = ""                # populated ONLY when mode == "llm_failed"
    # Set when the intended primary provider failed and a lower-priority one
    # answered. Carries "provider: reason" so the UI can name what broke.
    fell_back_from: str = ""

    @property
    def refused(self) -> bool:
        return self.mode == "refused"

    @property
    def llm_failed(self) -> bool:
        """A provider WAS configured and every attempt failed.

        # WHY is this distinct from "extractive"?
        # "extractive" is a designed, documented mode: no credentials are present,
        # so the app answers from retrieved text instead. That is expected and fine.
        # "llm_failed" means the operator configured a provider and it did not
        # work — a bad key, a retired model slug, a rate limit. Collapsing the two
        # would hide a real outage behind a degraded-but-plausible answer, so this
        # mode exists to force the UI to surface the provider's own error.
        """
        return self.mode == "llm_failed"


def answer_question(query: str, retriever, k: int = 5, use_llm: bool = True,
                    prefer: Optional[str] = None, temperature: float = 0.0,
                    on_stage: Optional[callable] = None,
                    system_prompt: Optional[str] = None,
                    select: Optional[callable] = None,
                    postprocess: Optional[callable] = None,
                    field_weights: Optional[dict] = None,
                    field_boost: float = 0.0,
                    agent: str = "") -> Answer:
    """The complete RAG loop: retrieve → pack → prompt → generate → verify.

    `on_stage(stage, detail)` is an optional progress callback invoked as the
    pipeline advances: "retrieving" → "building" → "generating" → "verifying".

    # THE THREE PERSONA HOOKS
    # Two agents must differ in their CONTEXT BUILDING and their PROMPT without
    # either owning a second copy of this loop — a forked loop is how the CLI, the
    # eval harness and the app stop running the same code. So the loop stays here
    # and exposes exactly three seams:
    #
    #   select(chunks)        -> chunks   after retrieval, before packing.
    #                            Field allowlists, persona packing order, k trim.
    #   system_prompt         -> str      swaps the RULES block only.
    #   postprocess(text, ch) -> str      enforcement that must not be optional:
    #                            escalation notices, heading schemas, advisory
    #                            re-framing. A rule is a request; this is a check.
    #
    # All three default to None, reproducing the original behaviour exactly — which
    # is what let Phase 2 ship against an already-green regression gate.

    # WHY a callback instead of letting the UI drive the steps itself?
    # The UI needs to show honest per-stage progress ("Retrieving…", then
    # "Generating…"), but splitting this function apart in streamlit_app.py would
    # move orchestration — the abstention rule, the packing order, the fallback
    # policy — into the presentation layer, where it could drift from what the
    # CLI and the eval harness execute. The callback lets the UI *observe* the
    # pipeline without owning any part of it, so every caller still runs
    # byte-identical logic.
    """
    def stage(name: str, detail: str = "") -> None:
        if on_stage is not None:
            try:
                on_stage(name, detail)
            except Exception:
                # Progress reporting must never be able to fail an answer.
                pass

    def finish(text: str, used: list) -> str:
        """Run a persona's enforcement over ANY outgoing text.

        # WHY does this cover refusals too?
        # A refusal IS an answer, and for a lay reader it is arguably the most
        # important one — it is the moment they still need somewhere to go.
        # Skipping enforcement here produced a patient refusal that declined to
        # help and then failed to say who could, leaving rule 7 of the patient
        # prompt unenforced on exactly the path that needed it most.
        """
        return postprocess(text, used) if postprocess is not None else text

    # 6a — RETRIEVE. If retrieval abstained, we stop here. Nothing is sent to an
    #      LLM when there is no context, because a model given no context answers
    #      from memory — precisely the failure the whole system exists to prevent.
    stage("retrieving")
    # `field_weights` is computed by retrieval/field_router.py and passed through.
    # Stage 07 forwards it without interpreting it, for the same reason stage 06
    # does not import the router: packages call into stages, never the reverse.
    outcome = retriever.retrieve_context(query, k=k, field_weights=field_weights,
                                         field_boost=field_boost)
    if not outcome.ok:
        stage("refused", outcome.status)
        n_entities = len(retriever.entity_vocab)
        return Answer(text=finish(refusal_message(outcome, n_entities), []),
                      chunks=[], mode="refused", status=outcome.status, agent=agent)

    chunks = outcome.chunks
    dropped: tuple = ()

    # 6a-ii — PERSONA SELECTION. A persona may narrow what it is willing to read.
    #
    # # WHY can this cause a refusal?
    # Because a filter that silently returned nothing would hand the model an empty
    # context, and a model given no context answers from memory — the exact failure
    # the whole system exists to prevent. If a persona's policy empties the result
    # set, that is an abstention and is reported as one.
    if select is not None:
        before = {c.field for c in chunks}
        chunks = list(select(chunks))
        dropped = tuple(sorted(before - {c.field for c in chunks}))
        if not chunks:
            stage("refused", "persona_field_policy")
            n_entities = len(retriever.entity_vocab)
            return Answer(
                text=finish(refusal_message(outcome, n_entities), []), chunks=[],
                mode="refused", status="persona_field_policy", agent=agent,
                dropped_fields=dropped)

    # 6b — PACK + BUILD.
    stage("building", f"{len(chunks)} sources")
    context = pack_context(chunks)
    prompt = build_prompt(query, context, system_prompt=system_prompt)

    # 6c — GENERATE via the provider cascade.
    if not use_llm:
        text = finish(extractive_answer(chunks), chunks)
        return Answer(text=text, chunks=chunks, mode="extractive",
                      status=outcome.status, prompt=prompt, context=context,
                      attempts=[("(llm disabled)", "skipped")], agent=agent,
                      dropped_fields=dropped,
                      verification=verify_answer(text, chunks))

    stage("generating", prefer or "auto")
    response = llm_providers.generate(prompt, prefer=prefer, temperature=temperature)
    if response.ok:
        stage("verifying", f"{response.provider} · {response.model}")
        text = finish(ensure_disclaimer(extract_answer(response.text)), chunks)
        return Answer(text=text, chunks=chunks, mode="llm",
                      status=outcome.status, provider=response.provider,
                      model=response.model, prompt=prompt, context=context,
                      attempts=response.attempts, agent=agent,
                      dropped_fields=dropped,
                      fell_back_from=response.fell_back_from,
                      verification=verify_answer(text, chunks))

    # Generation did not produce text. Split the two very different reasons apart
    # instead of quietly degrading both to the same "extractive" answer.
    configured = llm_providers.available_providers()
    text = finish(extractive_answer(chunks), chunks)

    if configured:
        # A provider is configured and still failed → this is an incident, not a
        # mode. Keep the sourced text so the user is not left empty-handed, but
        # mark it so the UI raises the provider's own error message.
        errors = "; ".join(f"{name}: {outcome_}" for name, outcome_ in response.attempts
                           if outcome_ not in ("not configured",))
        return Answer(text=text, chunks=chunks, mode="llm_failed",
                      status=outcome.status, prompt=prompt, context=context,
                      attempts=response.attempts, llm_error=errors, agent=agent,
                      dropped_fields=dropped,
                      verification=verify_answer(text, chunks))

    # No credentials anywhere → extractive is the designed, documented behaviour.
    return Answer(text=text, chunks=chunks, mode="extractive",
                  status=outcome.status, prompt=prompt, context=context,
                  attempts=response.attempts, agent=agent, dropped_fields=dropped,
                  verification=verify_answer(text, chunks))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("  STAGE 7 — PROMPT BUILDING & LLM GENERATION")
    print("=" * 78)

    status = llm_providers.llm_status()
    print(f"LLM: {status['active'] or 'none (extractive mode)'}"
          f"{' · ' + status['model'] if status['model'] else ''}\n")

    retriever = retrieval.build_retriever()
    print(f"Retriever ready over {len(retriever.df):,} chunks.\n")

    queries = sys.argv[1:] or [
        "how does metformin work?",                  # normal grounded answer
        "what are the side effects of aspirin?",     # unknown drug → refusal
        "what is the retail price of naproxen?",     # unanswerable from labels
    ]

    for query in queries:
        print("=" * 78)
        print(f"Q: {query}")
        print("=" * 78)
        result = answer_question(query, retriever, k=5)

        print(f"mode={result.mode}  status={result.status}"
              f"{'  provider=' + result.provider if result.provider else ''}")

        if result.chunks:
            print("\nRetrieved sources:")
            for rank, chunk in enumerate(result.chunks, 1):
                print(f"  [Source {rank}] {chunk.score:.3f}  {chunk.citation()}")

        print(f"\nANSWER:\n{result.text[:1400]}")

        if result.verification:
            print(f"\nVERIFICATION: {result.verification.summary()}")
            print(f"  grounded = {result.verification.grounded}")
        print()

    # Show a full assembled prompt once — this is the actual contract text.
    demo = answer_question("how does metformin work?", retriever, k=3, use_llm=False)
    print("=" * 78)
    print("  THE ASSEMBLED PROMPT (first 1200 chars)")
    print("=" * 78)
    print(demo.prompt[:1200], "\n…")

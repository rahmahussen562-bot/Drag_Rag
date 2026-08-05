# 🎬 PhARMA RAG — the 10-minute demo

Eight queries, in order. Each one demonstrates something the previous one could
not, and the order matters: **the refusals land harder after the system has
already proved it can answer.**

```bash
streamlit run streamlit_app.py      # or: make run
```

> **Before you start.** Check the sidebar **Status** panel says `LLM: ollama` or
> `openrouter` — not *not configured*. With no provider the app answers
> extractively, which is correct behaviour but a weaker demo. And note the
> **persona** control: half of this script depends on it.

---

## 1 · It answers, and shows its work — 60s

**Persona: Healthcare Professional**

> `How does metformin work?`

Point at, in this order:

1. The **live pipeline stepper** — retrieving → building → generating → verifying.
   The buyer is watching the pipeline execute, not a spinner.
2. The **answer**, with inline `[Source N]` markers.
3. **📚 Sources used** — expand it. Each card shows the drug, the label section,
   and a **stacked bar splitting the score into BM25 vs dense contribution**.

> *"The score is not a number you have to trust. It says which retriever found
> this chunk and why."*

---

## 2 · It refuses — and that is the product — 60s

> `What are the side effects of aspirin?`

Aspirin is a drug **every** language model knows in detail. It is **not in this
corpus**. The system refuses and offers the nearest thing it does have.

> *"This is the whole system in one query. Naive RAG answers this from a different
> drug — fluently, with citations, and wrongly. Refusing requires deciding not to
> answer before anything is retrieved."*

---

## 3 · Same question, different reader — 90s

**Switch persona to Patient**, then ask:

> `Should I increase my metformin dose to 1000 mg twice daily?`

The patient agent **declines** — and says who to ask instead.

**Switch to Healthcare Professional. Ask it again.** It answers, with dosing.

> *"Same query, same corpus, one shared index. The difference is configuration:
> which label sections that reader may see, and what is enforced on the answer
> afterwards. A third persona is a config file, not a code change."*

---

## 4 · The screen that bypasses the model entirely — 45s

**Persona: Patient**

> `I have chest pain after taking metformin`

No retrieval. No model call. A fixed urgent-care message.

> *"An LLM asked to handle chest pain will improvise, and that failure mode is
> somebody not calling an ambulance. This path removes generation rather than
> constraining it."*

---

## 5 · Structured lookup for facts, RAG for narrative — 60s

**Sidebar → ⚠️ Interaction checker.** Enter:

```
warfarin
naproxen
metformin
omeprazole
```

A deterministic pairwise lookup over 160,235 DDInter pairs.

> *"An interaction is a fact about a pair of drugs — a dictionary lookup, not a
> retrieval problem. It cannot miss a documented pair or invent one. RAG is used
> only to explain what the lookup found."*

Then, as **Patient**, ask `does warfarin interact with naproxen?` — the patient
agent **refers you here** rather than summarising the interactions section.

---

## 6 · The architecture, generated from the code — 90s

**Sidebar → 🏗️ Architecture.**

- Expand any stage: the parameters are **read from the live modules at render
  time**. `α = 0.20` comes from `retrieval.ALPHA`.
- **Tab 2 — "Where the agents diverge"**: computed from the dataclass, so it
  cannot claim a difference that does not exist.
- **Tab 3** — press **▶ Run the mirror test** and watch it refuse, live.

> *"A hand-drawn architecture diagram starts lying the first time someone tunes a
> constant. This one cannot."*

---

## 7 · The number was measured, and then double-checked — 90s

**Sidebar → 🔬 Retrieval lab.** Query `how does metformin work?`,
**A: α = 0.20**, **B: α = 0.60**. Watch the result sets diverge.

> *"α was swept, not guessed. But the more useful story is what we rejected:
> pure-dense scores the **best** field recall and we do not ship it, because it
> drops drug recall below 100%. Missing the right section answers the wrong part
> of the right drug. Missing the right drug answers from a different medicine."*

If they are technical, add:

> *"Field routing scored 100% on the set we tuned it on. We did not believe it —
> a perfect MRR is a red flag — so we re-ran it on phrasings it had never seen.
> The real number is 95%. That is the one in the README."*

---

## 8 · A pharmacist can file it — 45s

**Sidebar → 🧾 Audit log.**

Every answer with its sources, provider, and verification result. **Export as CSV.**

> *"Queries are redacted **on write**, not at export — a log that stores
> identifiers and strips them later has already stored them."*

---

## If you are asked the hard questions

| Question | Answer |
|---|---|
| *"How do you know it isn't making things up?"* | A counterfactual. `eval/run_e2e.py` injects **Zorbaxin** — a drug that does not exist — dosed 250 mg every 8 hours. Those numbers are unobtainable from model weights, so reproducing them proves retrieval. The mirror test proves the reverse: aspirin, which every model knows, is refused. |
| *"What happens when it breaks?"* | It degrades in quality, never in trustworthiness. No torch → BM25-only. No LLM → extractive. `python eval/test_degraded.py` proves each mode still grounds, still cites and **still refuses**. |
| *"Can we run it inside our network?"* | `docker compose up`. The image bakes in the embedding model and fails the build if it cannot answer. |
| *"Can we change it without a developer?"* | `config.yaml`. The disclaimer is the one field you cannot remove. |
| *"Can we sell this?"* | **Not as it stands.** DDInter is CC BY-NC-SA 4.0 — non-commercial. The interaction source sits behind an adapter, so swapping to a licensed feed is a config change. Raise this before anyone signs anything. |

---

## What not to oversell

Say these before they are discovered:

- **Brand names do not resolve.** `Glucophage` refuses instead of grounding to
  metformin. `rxnorm.py` exists but is not in the query path.
- **Intent coverage is 32%** on unseen phrasings. Routing helps because the
  classifier fails safe, not because it is good yet.
- **Small local models cite inconsistently.** Roughly 1 source in 5 with
  `llama3.2:3b`. That is instruction-following quality, not a pipeline defect —
  the same prompt with a 70B model cites reliably.
- **The audit log is session-scoped.** A real deployment needs a durable,
  tamper-evident store.

> Volunteering the limitations is what makes the measured claims credible.

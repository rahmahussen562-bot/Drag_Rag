# 💊 PhARMA RAG — Grounded Drug Q&A with Cited Sources

A production Retrieval-Augmented Generation system over **506 real FDA drug labels**, built
from scratch (no LangChain, no LlamaIndex) so every step of the pipeline is visible and
independently runnable.

The system answers drug questions **only** from retrieved text, **cites every source**, and
**refuses** when the corpus cannot support an answer. You can also upload your own documents
(PDF / DOCX / TXT / MD / CSV / JSON) and query them through the identical pipeline.

> ⚠️ **Educational project. Not medical advice.** Every answer carries a disclaimer and cites
> the exact chunks it used.

📓 **New here? Start with [`PhARMA_RAG_Final.ipynb`](PhARMA_RAG_Final.ipynb)** — every stage, why
it is built that way, and the measurement that justifies it, with figures. It imports the
production modules rather than re-implementing them, so it cannot drift out of date.

---

## Table of contents

- [Architecture](#architecture)
- [The final notebook](#the-final-notebook)
- [Folder structure](#folder-structure)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Key parameters at a glance](#key-parameters-at-a-glance)
- [How retrieval works](#how-retrieval-works)
- [How citations work](#how-citations-work)
- [Prompt strategy](#prompt-strategy)
- [LLM providers & fallback](#llm-providers--fallback)
- [Run locally](#run-locally)
- [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud)
- [Secrets configuration](#secrets-configuration)
- [Evaluation & results](#evaluation--results)
- [Design decisions](#design-decisions)
- [Provenance: the three source notebooks](#provenance-the-three-source-notebooks)
- [Data sources](#data-sources)

---

## Architecture

```
                     ┌──────────────────────────────────────────────┐
   data/*.json ──────►  01_documents.py        Raw Documents        │
   your uploads ─────►  · FDA labels → 1 doc per field              │
                     │  · PDF/DOCX/TXT/MD/CSV/JSON → 1 doc/section  │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  02_preprocessing.py    Preprocessing        │
                     │  lowercase · lemmatise · drop stopwords      │
                     │  KEEP numbers · KEEP negation  ← safety      │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  03_chunking.py         Chunking             │
                     │  sliding window 90 words / 25 overlap        │
                     │  self-describing prefix + metadata           │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  04_vector_representation.py                 │
                     │  DENSE  all-MiniLM-L6-v2 → 384-d unit vecs   │
                     │  SPARSE TF-IDF (1-2 grams)                   │
                     │  cached to disk w/ content fingerprint       │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  05_create_vector_store.py   Vector Store    │
                     │  FAISS IndexFlatIP (exact cosine)            │
                     │  + BM25 inverted index                       │
                     │  IVF auto-enables past 50k vectors           │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  06_retrieve_context.py   Context Retrieval  │
                     │  ground → restrict → hybrid score → floor    │
                     │  → optional cross-encoder rerank             │
                     │  ABSTAINS instead of guessing                │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  07_prompting.py    Prompt + LLM Generation  │
                     │  numbered [Source N] blocks + 6 rules        │
                     │  → llm_providers.py cascade                  │
                     │  → verify citations / disclaimer / overlap   │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  streamlit_app.py          Streamlit UI      │
                     │  Ask · Interaction checker · Upload          │
                     │  + 🔬 evaluation mode                        │
                     └──────────────────────────────────────────────┘
```

Every box is a file you can run on its own:

```bash
python 03_chunking.py            # prints chunk statistics + the overlap guarantee
python 06_retrieve_context.py    # demonstrates retrieval AND abstention
```

---

## The final notebook

**[`PhARMA_RAG_Final.ipynb`](PhARMA_RAG_Final.ipynb)** is the single reference document for the
system: every stage, the reason it is built that way, and the measurement that justifies it —
with generated figures.

It is the one place to start if you want to understand the project rather than run it.

**It imports the production modules; it does not re-implement them.**

```python
from stages import load
retrieval = load("06_retrieve_context")   # the actual file the Streamlit app runs
```

That is the whole point. Every number and every figure in the notebook is produced by the same
code that serves users, so it cannot drift out of date and cannot flatter the system with results
the product does not actually achieve. Change `ALPHA` in `06_retrieve_context.py` and the notebook
reports the new value — and redraws the sweep — on the next run.

Each stage follows the same three beats: **why the stage exists** → **what was decided (and what
was rejected)** → **the evidence**, run live. Highlights:

| | |
|---|---|
| **Part 2** | why the textbook "clean your text" recipe is *dangerous* on clinical text |
| **Part 4** | the paraphrase query where TF-IDF scores **exactly 0.000** — the case for hybrid, in one chart |
| **Part 6** | the **α sweep**, run live, including *why the best-scoring setting was rejected* |
| **Part 7** | the **counterfactual grounding proof** and its mirror test |
| **Part 9** | a decision ledger: every parameter, and what would make it wrong |

Figures are written to [`docs/figures/`](docs/figures) on each run.

> The three original teaching notebooks (*Pipeline*, *Learning Edition*, *Scale Edition*) have
> been consolidated into this one. Each re-implemented the pipeline inline, which meant four
> copies of the logic drifting apart from production. Their measured results are preserved in
> [Provenance](#provenance-the-three-source-notebooks) below, and the files themselves remain in
> git history (`git log --diff-filter=D --name-only`).

---

## Folder structure

```
.
├── 01_documents.py               Stage 1 — load FDA labels + parse uploads
├── 02_preprocessing.py           Stage 2 — medically-safe text cleaning
├── 03_chunking.py                Stage 3 — sliding-window chunking
├── 04_vector_representation.py   Stage 4 — dense embeddings + TF-IDF
├── 05_create_vector_store.py     Stage 5 — FAISS + BM25 indexes
├── 06_retrieve_context.py        Stage 6 — grounded hybrid retrieval
├── 07_prompting.py               Stage 7 — prompt building + generation
├── streamlit_app.py              Stage 8 — the UI (Streamlit Cloud entry point)
│
├── stages.py                     loader: lets 0N_*.py files import each other
├── llm_providers.py              OpenRouter → Gemini → Groq → Ollama cascade
├── rag_pipeline.py               compatibility facade (delegates to stages)
├── rxnorm.py                     optional brand→generic normalisation
│
├── agents/                       the personas — config, not subclasses
│   ├── base.py                   AgentConfig · Agent protocol · registry
│   ├── patient.py                lay persona (+ red-flag triage list)
│   ├── practitioner.py           clinical persona
│   └── prompts/*.md              prompts live as text files, not string literals
│
├── retrieval/                    everything AROUND stage 06 — never forking it
│   ├── query_understanding.py    normalise → expand → classify → resolve
│   ├── field_router.py           query intent → target label fields
│   ├── context_builder.py        MMR · neighbour expansion · budget · packing
│   └── trace.py                  RetrievalTrace — every decision, for the UI
│
├── interactions/                 swappable interaction source (licensing, §8)
│   ├── base.py                   InteractionProvider protocol + registry
│   └── ddinter.py                DDInter 2.0 implementation
│
├── ui/                           the presentation layer
│   ├── theme.py                  one stylesheet + the badge vocabulary
│   ├── state.py                  cached resources · session bootstrap · AppContext
│   ├── sidebar.py                nav · settings · live status panel
│   ├── components/               source_card · trust_panel · pipeline_stepper · score_bars
│   └── pages/                    one module per page, each exposing render(ctx)
│
├── PhARMA_RAG_Final.ipynb        THE notebook — every stage, why, and the figures
├── docs/figures/                 charts generated by the notebook (regenerated on run)
│
├── requirements.txt
├── .env.example                  template for local secrets (copy to .env)
├── .gitignore                    excludes .env, secrets.toml, caches
│
├── data/
│   ├── drugs_large.json          506 openFDA drug labels (the corpus)
│   ├── drugs.json                12-drug teaching corpus
│   ├── doc_embeddings.npy        cached embeddings — COMMITTED on purpose (see boot cost)
│   └── ddinter/                  DDInter 2.0 interaction CSVs
│
├── eval/
│   ├── eval_set.json             ground truth: answerable / refuse / interactions
│   ├── run_eval.py               retrieval recall + abstention accuracy
│   ├── run_e2e.py                full-pipeline verification incl. grounding proof
│   └── tune_alpha.py             empirical sweep of the hybrid weight
│
└── .streamlit/                   theme + secrets template
```

---

## The pipeline, stage by stage

### Stage 1 — Raw documents (`01_documents.py`)
Everything becomes one uniform `Document` (`title`, `field`, `category`, `text`, `origin`).

- **FDA labels** are already structured, so each drug explodes into **one document per field**
  (`description`, `mechanism`, `indications`, `dosage`, `side_effects`, `contraindications`,
  `interactions`). Splitting on the label's own boundaries is free, semantically perfect chunking.
- **Uploads** become one document per natural section: PDF → per page, Markdown → per heading,
  CSV → per row, DOCX/TXT → body.

506 drug records → **3,472 documents** → 934,850 words.

### Stage 2 — Preprocessing (`02_preprocessing.py`)
This is a **safety** decision, not a tuning knob. The standard NLP recipe is *actively dangerous*
on clinical text:

| Input | `aggressive_stemmed` (unsafe) | `readable_lemmatized` (used) |
|---|---|---|
| `No known hepatotoxicity at 500 mg. Not recommended.` | `known hepatotox mg recommend` ❌ | `no known hepatotoxicity 500 mg . not recommended` ✅ |

Dropping stopwords deletes **"no"/"not"** — inverting the meaning. Dropping numbers deletes the
**dose**. The default profile therefore protects a negation list and never removes digits.
`python 02_preprocessing.py` asserts both guarantees.

> Only BM25/TF-IDF consume cleaned text. The **embedding model is fed raw text** — transformers are
> trained on natural language and are hurt by stopword removal.

### Stage 3 — Chunking (`03_chunking.py`)
Sliding window of **90 words with 25 words of overlap**.

- **Why 90?** ≈120 tokens, safely inside all-MiniLM-L6-v2's **256-token limit**. Longer text is
  silently truncated by the model — i.e. invisible to retrieval.
- **Why 25 overlap?** A fact landing on a boundary would otherwise be cut in half and be
  unretrievable. 25 words guarantees any fact shorter than 25 words survives intact in one window.
- Every chunk is **self-describing**: prefixed `"Metformin Hydrochloride — dosage: …"`, so a chunk
  torn out of its document still says which drug and which section it belongs to.

3,472 documents → **14,955 chunks** (mean 85.7 words).

### Stage 4 — Vector representation (`04_vector_representation.py`)
Two representations, because they fail in **opposite** cases:

| | Sparse (BM25/TF-IDF) | Dense (embeddings) |
|---|---|---|
| exact tokens (`naproxen`, `500 mg`, `CYP3A4`) | ✅ | ❌ |
| paraphrase (`high blood sugar` → `hyperglycemia`) | ❌ scores exactly 0 | ✅ |

Embeddings are **unit-normalised** so cosine similarity reduces to a dot product — which is what
makes FAISS's inner-product index give *exact* cosine ranking.

Encoding is cached to `data/doc_embeddings.npy`, keyed by a **SHA-256 content fingerprint** of the
corpus + model name. (Comparing only lengths — the notebooks' approach — silently loads stale
vectors when a corpus edit preserves the chunk count.)

### Stage 5 — Vector store (`05_create_vector_store.py`)
**FAISS `IndexFlatIP`** (exact) + a **BM25 inverted index**. See
[Design decisions](#design-decisions) for why FAISS was kept over Chroma.

### Stage 6 — Context retrieval (`06_retrieve_context.py`)
See [How retrieval works](#how-retrieval-works).

### Stage 7 — Prompt + generation (`07_prompting.py`)
See [Prompt strategy](#prompt-strategy) and [How citations work](#how-citations-work).

---

## Key parameters at a glance

| Parameter | Value | Why |
|---|---|---|
| **Embedding model** | `all-MiniLM-L6-v2` (384-d, 22 MB) | Fits Streamlit Cloud's 1 GB limit; ~2k chunks/s on CPU |
| **Chunk size** | **90 words** (~120 tokens) | Inside the model's 256-token limit |
| **Chunk overlap** | **25 words** (28%) | Facts straddling a boundary survive |
| **Vector database** | **FAISS `IndexFlatIP`** | Exact cosine at 15k vectors in ~0.9 ms |
| **Approximate index** | FAISS `IndexIVFFlat`, auto past 50k vectors | Speed/recall trade-off only when needed |
| **Sparse retriever** | BM25-Okapi (k1=1.5, b=0.75) | Saturation + length normalisation |
| **Hybrid weight α** | **0.20** BM25 / 0.80 dense | Measured, not guessed — see below |
| **Retrieval pool** | 40 candidates before reranking | Cheap recall, then precision |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | MRR 0.42 → 0.80 |
| **Relevance floor** | cosine ≥ 0.25 **or** BM25 ≥ 1.0 | Below it → abstain |
| **Top-k** | 5 (3–10 in the UI) | Enough context without diluting the prompt |
| **Temperature** | 0.0 | Factual assistant; makes evaluation reproducible |

---

## How retrieval works

Four mechanisms, in order:

**1. Entity grounding — *before* retrieving anything.**
Vector search *always* returns its k nearest neighbours, even when the nearest neighbour is
irrelevant. Ask about a drug that isn't in the corpus and naive RAG confidently answers from a
**different drug**. So the query is first checked against the corpus vocabulary. Drug names are
indexed by *identity tokens* (salt/dosage-form words stripped), so `metformin` matches
`Metformin Hydrochloride`.

```
"side effects of aspirin?"  →  aspirin ∉ vocabulary  →  REFUSE (+ "did you mean…?")
```

**2. Metadata restriction.** Once the query names a known drug, retrieval is restricted to that
drug's chunks. This turns a 15,000-way problem into a ~30-way one ("which *field* of this drug?"),
which is where most of the accuracy comes from.

**3. Hybrid scoring.** BM25 and cosine are min-max normalised to `[0,1]` (they have incomparable
scales) and blended:

```
score = 0.20 · minmax(BM25) + 0.80 · minmax(cosine)
```

**4. Relevance floor + abstention.** Results below the floor are dropped. If **nothing** clears it,
retrieval returns an **empty list** — and stage 07 refuses rather than generating. Returning fewer
than k results is a feature.

**5. Optional cross-encoder rerank.** Bi-encoders embed query and chunk separately; a cross-encoder
reads the pair *jointly* and scores true relevance — far more accurate, far too slow for 15k chunks.
Hence the cascade: retrieve 40 cheaply → rerank those 40 → keep the best 5.

---

## How citations work

1. **Retrieval returns an ordered list.** Position is identity: `chunks[0]` is `[Source 1]`.
2. **Context is packed as numbered blocks**, not concatenated prose:
   ```
   [Source 1] (drug=metformin hydrochloride, field=mechanism, score=0.952)
   Metformin Hydrochloride — mechanism: 12.1 Mechanism of Action …
   ```
3. **The prompt requires inline `[Source N]` markers** (rule 2).
4. **The UI renders the same list in the same order**, so `[Source 3]` in the answer and
   *Source 3* in the expander are guaranteed to be the same chunk.
5. **Citations are mechanically verified** after generation (`verify_answer`):
   - which source numbers were cited,
   - **invalid citations** — a `[Source 7]` when only 5 were supplied is a *fabricated* citation
     and is flagged in red,
   - whether the disclaimer is present (auto-appended if the model forgot),
   - **context overlap** — the share of the answer's content words that appear in the supplied
     context. Low overlap + confident citations is the fingerprint of decorative citing.

Turn on **🔬 Evaluation mode** in the sidebar to see all of it, plus the verbatim prompt.

---

## Prompt strategy

A single system prompt carries six rules (preserved verbatim from the notebooks — the imperative,
numbered phrasing is what small models actually follow):

1. Base the answer **strictly** on the context. No outside knowledge.
2. Cite sources inline as `[Source N]`.
3. Say so explicitly when the context is insufficient.
4. Never give medical advice; always include the disclaimer.
5. **Preserve exact numbers and negation words.**
6. Refuse, with an explanation, when the question can't be answered from context.

**Layout:** `RULES → CONTEXT → QUESTION`. Attention degrades in the middle of long inputs
("lost in the middle"), so the two things the model must obey sit at the high-attention **edges**,
with the bulk material between them.

**Enforcement in code, not just in the prompt:** the disclaimer is appended if missing,
`<think>…</think>` scratchpads from reasoning models are stripped, context is capped at 12,000
characters (an overflowing prompt is truncated *from the end* — which would silently cut off the
question itself).

---

## LLM providers & fallback

**OpenRouter is the primary provider.** Only `OPENROUTER_API_KEY` is required.

| # | Provider | Default model | Get a key |
|---|---|---|---|
| 1 | **OpenRouter** | free-model chain (below) | [openrouter.ai/keys](https://openrouter.ai/keys) |
| 2 | **Gemini** | `gemini-2.0-flash` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| 3 | **Groq** | `llama-3.3-70b-versatile` | [console.groq.com/keys](https://console.groq.com/keys) |
| 4 | **Ollama** (local) | `llama3.2:latest` | no key — `ollama pull llama3.2` |
| 5 | **Extractive** | — | no LLM configured: returns retrieved text verbatim, still cited |

### The OpenRouter model chain

Free model slugs get **retired without notice** — this is not hypothetical, it is exactly what broke
this app. The previous hard-coded default returned:

```
HTTP 404 {"error":{"message":"This model is unavailable for free …"}}
```

A single pinned slug therefore has a shelf life, and when it expires the provider looks broken even
though the key is perfectly valid. So `llm_providers.OPENROUTER_MODEL_CHAIN` holds an ordered list
and a model-level failure (404 / 429 / "no endpoints") advances to the next entry **while staying on
OpenRouter**. An auth failure (401/403) does *not* advance — retrying a bad key against five models
is five pointless round trips.

Order was measured, not guessed: each candidate was given the real RAG prompt and scored on whether
it emitted `[Source N]` citations, kept the disclaimer, and grounded its answer in the context.

| Model | Citations | Context overlap | Latency |
|---|---|---|---|
| `nvidia/nemotron-3-nano-30b-a3b:free` | 2 | 68% | 4.0 s |
| `openai/gpt-oss-20b:free` | 2 | 66% | 7.7 s |
| `inclusionai/ling-3.0-flash:free` | 1 | 84% | 1.4 s |
| `google/gemma-4-26b-a4b-it:free` | 1 | 70% | 12.7 s |
| `openrouter/free` | 1 | 61% | 2.0 s |

Setting `OPENROUTER_MODEL` in secrets pins one model and bypasses the chain entirely.

### Free-tier daily quota

OpenRouter's free tier caps **requests per day per account**, not per model:

```
429 "Rate limit exceeded: free-models-per-day.
     Add 10 credits to unlock 1000 free model requests per day"
```

This is an **account** limit, so walking the model chain cannot help — every extra attempt is a
guaranteed failure that burns latency. `_is_account_level_error()` detects it and stops after the
first model (5 wasted calls → 1). Adding 10 credits to the OpenRouter account raises the cap to
1000 requests/day.

### Failures are never silent

A configured provider that fails is an **incident, not a mode**. The app distinguishes three cases:

- `extractive` — no credentials anywhere. The designed, documented fallback; quiet by intent.
- `llm_failed` — a provider *was* configured and every attempt failed. A red banner shows the
  provider's own error above the answer, so a bad key or retired slug cannot hide behind a
  plausible-looking sourced answer.
- **fallback used** — the primary provider failed and a lower-priority one answered. An amber
  banner names the substitution and the underlying error.

That last case matters more than it looks. During development a local Ollama will happily cover for
a rate-limited OpenRouter, so the app looks healthy while the *actual deployment target* — Cloud,
where no local model exists — is broken. The banner is what stops that being discovered in
production.

---

## Run locally

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add your OpenRouter key (the app runs without one, but answers extractively)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#   then paste OPENROUTER_API_KEY
#   (a .env file with the same variable also works)

# 3. Launch
streamlit run streamlit_app.py
```

Boot is **~23 s**: `data/doc_embeddings.npy` ships with the repo, so the corpus is never re-encoded
on a normal clone. Stage 04 validates it against a SHA-256 content fingerprint and silently
re-encodes only if the corpus or the model actually changed.

**Run any stage on its own:**

```bash
python 01_documents.py               # corpus stats; --refresh 500 re-downloads from openFDA
python 02_preprocessing.py           # the 4 cleaning profiles + safety assertions
python 03_chunking.py                # chunk statistics + the overlap guarantee
python 04_vector_representation.py   # sparse vs dense, side by side
python 05_create_vector_store.py     # FAISS exact vs approximate, verified
python 06_retrieve_context.py        # retrieval + abstention demo
python 07_prompting.py               # full RAG loop with verification
python llm_providers.py              # which providers are configured
```

**Lightweight mode:** comment out `sentence-transformers` and `faiss-cpu` in `requirements.txt`.
The app detects their absence and runs BM25-only — instant startup, lower field recall (see below).

---

## Deploy to Streamlit Cloud

1. Push the repo to GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo →
   main file **`streamlit_app.py`** → *Advanced settings* → Python **3.11**.
3. **Settings → Secrets** → paste `OPENROUTER_API_KEY` (see next section).
4. Deploy.

`streamlit_app.py` is the **only** entry point. (An `app.py` shim existed for backwards
compatibility and was removed — two entry points for one app is a deployment foot-gun.)

### Measured boot cost

These numbers decide whether the deploy succeeds, so they were measured rather than assumed:

| | Cold boot | Peak RSS |
|---|---|---|
| Embedding cache **absent** (what a `.gitignore`'d cache gives you) | **308 s** | **1,176 MB** ❌ |
| Embedding cache **committed** (current behaviour) | **23 s** | **868 MB** ✅ |

That is why `data/doc_embeddings.npy` is committed rather than ignored: without it the container
both blows the boot timeout and exceeds the memory limit. Breakdown of the 868 MB: torch +245 MB,
chunking +185 MB (transient tokenizer allocation), BM25 index +138 MB, embedding matrix +23 MB,
DDInter +61 MB.

Headroom notes:
- The cross-encoder reranker loads a **second** transformer. Leave the toggle **off** on Cloud
  unless you have confirmed the extra headroom.
- `faiss.index` is *not* committed — it rebuilds from the committed vectors in ~0.05 s.

---

## Secrets configuration

**No API key appears anywhere in the source.** `llm_providers.get_secret()` resolves at runtime, first hit wins:

1. `st.secrets[...]` — Streamlit Cloud
2. `os.environ[...]` — CI, Docker, shell
3. `.env` — local development (git-ignored)

**Streamlit Cloud** — *Settings → Secrets*:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
# optional — pin one model instead of using the auto-advancing chain:
# OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
# optional extra providers, only used if OpenRouter is unreachable:
# GROQ_API_KEY   = "gsk_..."
# GEMINI_API_KEY = "AIza..."
```

**Locally** — either `.streamlit/secrets.toml` (copy from
`.streamlit/secrets.toml.example`) or `.env` (copy from `.env.example`):

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

`.gitignore` excludes `.env` and `.streamlit/secrets.toml`; only the `.example` templates are
tracked. `eval/run_e2e.py` **Test 7** scans the tracked source for key-shaped literals and fails
the build if it finds any.

**Verifying your key works:**

```bash
python llm_providers.py "Reply with exactly the word: PONG"
```

This prints the provider status table and the full attempt log, so a bad key, a retired model, or a
rate limit is visible immediately rather than being inferred from a degraded answer.

---

## Evaluation & results

```bash
python eval/run_eval.py               # retrieval recall + abstention (fast, BM25-only)
python eval/run_eval.py --embeddings  # hybrid
python eval/run_e2e.py                # full pipeline, 26 checks incl. the grounding proof
python eval/run_e2e.py --no-llm       # retrieval-only (no generation)
python eval/tune_alpha.py             # reproduce the α sweep
```

**Retrieval (22 answerable + 10 refusal cases, k=5):**

| Configuration | Drug recall@5 | Field recall@5 | False refusals | Refusal accuracy |
|---|---|---|---|---|
| BM25 only | 100% | 41% | 0% | 100% |
| **Hybrid (α=0.20, default)** | **100%** | **91%** | **0%** | **100%** |
| Hybrid with the notebooks' α=0.60 | 100% | 68% | 0% | 100% |

Interaction lookup (DDInter): **9/9 exact severity match**.

**Why α = 0.20 and not the notebooks' 0.60** — `eval/tune_alpha.py` sweeps it:

| α | Drug recall@5 | Field recall@5 | Field MRR |
|---|---|---|---|
| 0.00 (dense only) | 95% ⚠️ | 95% | 0.867 |
| **0.20 (chosen)** | **100%** | **91%** | **0.822** |
| 0.60 (notebooks) | 100% | 68% | 0.502 |
| 1.00 (BM25 only) | 100% | 41% | 0.409 |

The choice is **lexicographic**, not a single score. Missing the right *field* answers the wrong
section of the right drug — unhelpful, and visible in the citation. Missing the right *drug*
answers from a different medicine — fluent, confidently cited, and clinically wrong. So drug recall
is maximised **first**, and field recall optimised only among settings that keep it at 100%. That
rules out pure-dense despite its better field recall: with no lexical weight, an exact drug-name
match earns no score of its own.

**End-to-end (`run_e2e.py`, 26 checks) — all passing**, including:

- row alignment: chunks ↔ embeddings ↔ BM25 all 14,955
- every retrieved chunk appears verbatim in the prompt
- **the grounding proof (Test 5)**, described below
- **the mirror test (Test 6)**: 10/10 out-of-corpus questions refused
- no API-key literals in tracked source

### How "the answer comes from retrieved context" is *proved*

Word overlap is weak evidence: ask *"how does metformin work?"* and a model answering purely from
pretraining still overlaps heavily with an FDA label, because both describe the same real drug.

So Test 5 uses a **counterfactual**. It injects a document about **Zorbaxin** — a drug that does not
exist — dosed *250 mg every eight hours*, contraindicated *under 12 years*. No language model can
know these facts. The system answers:

> *"The dose of Zorbaxin is 250 mg every eight hours, not to exceed 750 mg per day."*

Those numbers are unobtainable from model weights, so they can only have come through retrieval.
Test 6 runs the mirror image — *aspirin*, which every model knows perfectly well, is absent from the
corpus and is **refused**, proving pretrained knowledge does not leak past the retrieval gate.

> **Honest caveat:** with a small local model (`llama3.2:3b`) the answer is correct and grounded but
> emits `[Source N]` markers inconsistently — roughly 1 of 5 sources cited. That is instruction-
> following quality, not a pipeline defect: the same prompt with a 70B model on OpenRouter cites
> reliably. `run_e2e.py` reports this as a **warning**, not a pass, so the suite never overstates it.

---

## Design decisions

### FAISS was kept; Chroma was not adopted

The notebooks already use FAISS, and they use it **correctly**: embeddings are unit-normalised, then
searched with `IndexFlatIP`, so inner product *is* cosine similarity — exact ranking, not an
approximation. That passes the correctness bar, so the only question was whether Chroma buys
anything at this scale:

| | This project | Chroma's strength |
|---|---|---|
| **Size** | 15k × 384 = 23 MB, exact scan ≈ 0.9 ms | sublinear scaling we don't need |
| **Persistence** | index is a pure derivative of JSON, rebuilt in ~40s | persistent mutable collections |
| **Metadata filter** | pandas boolean mask, ~0.2 ms | `where` clauses |
| **Deployment** | faiss-cpu ≈ 15 MB, no writable disk | chromadb ≈ 120 MB + sqlite |

Adopting Chroma would add a dependency, a storage layer and **approximate** recall in exchange for
capabilities this project doesn't use. **The answer would change** past ~1M vectors, with concurrent
writers, or if incremental upserts became the primary write pattern — none of which apply.

**What was actually fixed:** the previous Streamlit app never used FAISS at all — it brute-forced
cosine through scikit-learn, throwing away the notebooks' index work. Stage 5 wires the real FAISS
index into the app, behind a NumPy fallback for hosts without the wheel.

### Structured lookup for facts, RAG for narrative
A drug **interaction** is a fact about a *pair* of drugs — a dictionary lookup, not a retrieval
problem. So `interactions.py` does a deterministic DDInter lookup that cannot miss a pair or
hallucinate one, and RAG is used only to *explain* what the lookup found.

### Numbered filenames need a loader
`import 01_documents` is a `SyntaxError` — module names can't start with a digit. `stages.py`
imports numbered files by path and caches them, so stages reuse each other instead of duplicating
code, while each file still runs standalone.

---

## Provenance: the three source notebooks

The production pipeline was not written from a blank page — it was distilled from three teaching
notebooks, all of which were read **and executed end to end** (0 failed cells) before any
production code was written.

> **Those three files have since been consolidated into
> [`PhARMA_RAG_Final.ipynb`](PhARMA_RAG_Final.ipynb)** and removed from the working tree. Each one
> re-implemented the pipeline inline, so the repo carried four copies of the same logic that drifted
> apart the moment production changed. The final notebook instead *imports* the stage modules, which
> is why it cannot go stale.
>
> They remain in git history and can be recovered at any time:
> ```bash
> git log --oneline --diff-filter=D -- '*.ipynb'   # find the commit that removed them
> git show <commit>^:PhARMA_RAG_Scale_Edition.ipynb > recovered.ipynb
> ```

The analysis below is kept because it is the evidence behind several production decisions — most of
all the α re-tuning.

| Notebook | Cells run | Corpus | Chunking | Retrieval | Vector store | Verdict |
|---|---|---|---|---|---|---|
| `PhARMA_RAG_Pipeline.ipynb` | 35/35 ✅ | 12 drugs / 84 chunks | field-level only | TF-IDF, BM25, dense, hybrid | FAISS Flat | Original Labs 5–9. Hard imports — crashes if a package is missing. Dead cells referencing `pharma-rag/`. |
| `PhARMA_RAG_Learning_Edition.ipynb` | 33/33 ✅ | 12 drugs / 84 chunks | field-level only | same + graceful fallbacks | FAISS Flat | Same results as Pipeline, but every dependency degrades gracefully. Best *explanations*. |
| `PhARMA_RAG_Scale_Edition.ipynb` | 22/22 ✅ | **506 drugs / 14,955 chunks** | **sliding window 90/25** | + **cross-encoder rerank** | **FAISS Flat + IVF** | Production shape: live openFDA download, disk caching, auto-generated eval. |

**Measured retrieval quality (Scale Edition, 60 auto-generated queries):**

| Retriever | P@5 | R@5 | Hit@5 | MRR |
|---|---|---|---|---|
| TF-IDF | 0.100 | 0.256 | 0.417 | 0.194 |
| BM25 | 0.127 | 0.255 | 0.417 | 0.220 |
| Embeddings | 0.367 | 0.437 | 0.683 | 0.556 |
| Hybrid α=0.6 | 0.220 | 0.430 | 0.717 | 0.420 |
| **Hybrid + rerank** | **0.540** | **0.674** | **0.883** | **0.799** |

### Verdict: a hybrid, led by Scale Edition

No single notebook was sufficient, so the production system merges all three:

- **Scale Edition** → the backbone: 506-drug corpus, sliding-window chunking, batched + cached
  embeddings, FAISS Flat/IVF, cross-encoder reranking.
- **Learning Edition** → the **graceful-degradation discipline** (every optional dependency has a
  fallback) and the 4-profile preprocessing safety demonstration. This is why the app survives a
  host with no torch wheel.
- **Pipeline notebook** → the evaluation metrics (P/R/Hit/MRR) and the `fit_transform`/`transform`
  discipline, enforced structurally in stage 04.
- **The existing Streamlit app** → entity grounding, relevance floors, refusal-with-suggestions, the
  DDInter interaction checker and RxNorm — features **none** of the notebooks had.

**What none of them had, and was built here:** document upload, a real vector store wired into the
app, multi-provider LLM fallback, secrets management, citation verification, evaluation mode, and
the counterfactual grounding proof.

**What was corrected rather than copied:**

| Issue | Notebook behaviour | Fixed |
|---|---|---|
| Hybrid weight | α=0.6, tuned on 12 drugs | **α=0.20**, re-measured at scale (+23 pts field recall) |
| Embedding cache key | compares list **length** | SHA-256 **content fingerprint** |
| FAISS in production | built in notebooks, unused by the app | wired into stage 05 |
| `use_embeddings=False` | silently still built embeddings | honoured (`use_dense` flag) |
| Ollama health probe | 2 s timeout vs ~2.4 s response | 6 s + 30 s cached probe |

---

## Data sources

- **Drug labels** — [openFDA drug label API](https://open.fda.gov/apis/drug/label/) (FDA Structured
  Product Labels). Rebuild with `python 01_documents.py --refresh 500`.
- **Interactions** — [DDInter 2.0](https://ddinter2.scbdd.com) (*Nucleic Acids Research* 2025),
  160,235 pairs across 1,939 drugs, CC BY-NC-SA 4.0. See `data/ddinter/SOURCE.md`.
- **Name normalisation (optional)** — NLM [RxNorm](https://lhncbc.nlm.nih.gov/RxNav/) REST API.
  (NLM's *interaction* API was discontinued in 2024; only normalisation endpoints are used.)

---

## License & disclaimer

Educational project. **Not medical advice.** Drug data can be incomplete or outdated — always
confirm with a pharmacist or prescriber. DDInter data is CC BY-NC-SA 4.0 (non-commercial).

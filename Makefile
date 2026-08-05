# PhARMA RAG — the commands worth remembering.
#
# Every target here wraps something already runnable by hand; the Makefile exists
# so a new person does not have to read the README to find out how to check the
# build. `make help` lists everything.

.DEFAULT_GOAL := help
.PHONY: help run demo eval eval-full health health-deep stages notebook \
        docker docker-run degraded clean

PY ?= python

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Running ────────────────────────────────────────────────────────────────
run:  ## Launch the Streamlit app
	streamlit run streamlit_app.py

demo:  ## Launch the app and print the 10-minute demo script
	@echo "Demo script: DEMO.md — 8 queries, in order, ~10 minutes."
	@echo ""
	streamlit run streamlit_app.py

# ── Evaluation ─────────────────────────────────────────────────────────────
eval:  ## Fast gate: retrieval recall, abstention, interactions
	$(PY) eval/run_eval.py --embeddings

eval-full:  ## Everything: e2e (26 checks), per-persona, held-out routing
	$(PY) eval/run_e2e.py --no-llm
	@echo ""
	$(PY) eval/run_agent_eval.py
	@echo ""
	$(PY) eval/run_heldout.py

tune:  ## Re-derive the tuned constants (alpha, field_boost)
	$(PY) eval/tune_alpha.py
	@echo ""
	$(PY) eval/tune_field_boost.py

# ── Health ─────────────────────────────────────────────────────────────────
health:  ## Fast deployment check (seconds)
	$(PY) -m healthcheck

health-deep:  ## Also build the index and test the abstention gate (minutes)
	$(PY) -m healthcheck --deep

# ── Pipeline ───────────────────────────────────────────────────────────────
stages:  ## Run every numbered stage standalone — the independence guarantee
	@for s in 01_documents 02_preprocessing 03_chunking 04_vector_representation \
	          05_create_vector_store 06_retrieve_context 07_prompting; do \
	  printf "  %-30s" "$$s"; \
	  $(PY) $$s.py > /dev/null 2>&1 && echo "OK" || echo "FAIL"; \
	done

notebook:  ## Re-execute the reference notebook and regenerate docs/figures/
	$(PY) -m nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 PhARMA_RAG_Final.ipynb

# ── Deployment ─────────────────────────────────────────────────────────────
docker:  ## Build the container (runs a deep health check at build time)
	docker build -t pharma-rag .

docker-run:  ## Run the container on :8501
	docker compose up --build

# ── Degradation ────────────────────────────────────────────────────────────
degraded:  ## Prove the documented fallbacks still answer (no torch / no LLM)
	$(PY) eval/test_degraded.py

clean:  ## Remove caches and generated indexes (NOT the embedding cache)
	@$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) \
	  for p in pathlib.Path('.').rglob('__pycache__')]"
	@rm -f data/faiss.index
	@echo "cleaned (data/doc_embeddings.npy kept — see .gitignore for why)"

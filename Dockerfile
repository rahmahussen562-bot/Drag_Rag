# ═══════════════════════════════════════════════════════════════════════════
# PhARMA RAG — container image
#
# # WHY does this exist when the app already runs on Streamlit Cloud?
# Because a hospital will not deploy on Streamlit Community Cloud. The moment a
# buyer is serious, the question becomes "can we run this inside our network?" and
# the answer has to be a container, not a link.
#
# Build:  docker build -t pharma-rag .
# Run:    docker run -p 8501:8501 -e OPENROUTER_API_KEY=sk-or-v1-... pharma-rag
# ═══════════════════════════════════════════════════════════════════════════

# Pinned to 3.11 to match the documented Streamlit Cloud target — a dev/prod
# Python skew is exactly the kind of difference that only shows up in production.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Requirements first, so a source edit does not invalidate the (slow) torch layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# # WHY bake the embedding model into the image?
# Otherwise the first request in a fresh container downloads ~90 MB from the HF
# Hub — which fails outright in an air-gapped hospital network, and is a cold-start
# penalty everywhere else. Failing at BUILD time is much better than at 3am.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')" || \
    echo "WARNING: model pre-fetch failed; the container will run BM25-only."

# Fail the build if the image cannot actually answer AND still refuse. A container
# that builds but cannot serve is worse than a build error, because it reaches
# production. --deep is right here: it costs minutes once, at build time, rather
# than on every 30 s probe.
RUN python -m healthcheck --deep || \
    (echo "deep healthcheck failed at build time" && exit 1)

# Non-root: nothing here needs to write outside /tmp, and a clinical-adjacent
# service running as root is an easy finding in any security review.
RUN useradd --create-home --uid 10001 pharma && chown -R pharma:pharma /app
USER pharma

EXPOSE 8501

# Streamlit's own endpoint — cheap, and it is what the platform probes anyway.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8501/healthz', timeout=4).status==200 else 1)"

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true", "--browser.gatherUsageStats=false"]

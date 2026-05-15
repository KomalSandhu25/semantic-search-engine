# =============================================================================
# Stage 1 — builder: install all Python dependencies into an isolated prefix
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs needed to compile faiss-cpu and numpy C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================================
# Stage 2 — api: lean FastAPI runtime image
# =============================================================================
FROM python:3.11-slim AS api

LABEL maintainer="Komal Sandhu <2277komal@gmail.com>"
LABEL org.opencontainers.image.description="Semantic Search Engine — FastAPI backend"

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY .env.example .env

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# Stage 3 — streamlit: lean Streamlit UI image
# =============================================================================
FROM python:3.11-slim AS streamlit

LABEL maintainer="Komal Sandhu <2277komal@gmail.com>"
LABEL org.opencontainers.image.description="Semantic Search Engine — Streamlit UI"

WORKDIR /app

COPY --from=builder /install /usr/local

COPY streamlit_app.py ./
COPY .streamlit/ ./.streamlit/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SEARCH_API_URL=http://api:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# Semantic Search Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8-orange)](https://github.com/facebookresearch/faiss)

A **production-grade semantic search system** that combines dense bi-encoder
retrieval (sentence-transformers + FAISS) with neural cross-encoder re-ranking
to deliver both high recall and high precision over large text corpora.

---

## Architecture

```
                              Query
                                |
                                v
            +---------------------------------------------+
            |  Stage 1 - Recall  (Bi-Encoder + FAISS)     |
            |                                             |
            |  1. Encode query -> dense vector (384-d)    |
            |  2. ANN search in FAISS IVF index           |
            |     -> top-100 candidate passages           |
            +---------------------------------------------+
                                |  100 candidates
                                v
            +---------------------------------------------+
            |  Stage 2 - Precision  (Cross-Encoder)       |
            |                                             |
            |  3. Score every (query, passage) pair       |
            |  4. Sort by relevance logit                 |
            |     -> top-10 re-ranked results             |
            +---------------------------------------------+
                                |
                    +-----------+-----------+
                    v                       v
           FastAPI /search          Streamlit UI
           (port 8000)              (port 8501)
```

### Model choices

| Component      | Model                                   | Why                                       |
|----------------|-----------------------------------------|-------------------------------------------|
| Bi-Encoder     | `all-MiniLM-L6-v2` (22 M params)       | Fast inference, 384-d embeddings, strong  |
|                |                                         | general-purpose performance on MTEB       |
| Cross-Encoder  | `ms-marco-MiniLM-L-6-v2` (22 M params) | Trained on MS MARCO; high MRR on BEIR    |
| Vector Index   | FAISS IVFFlat (auto-selects IVF/IVFPQ) | Sub-linear ANN search, CPU-friendly       |

---

## Performance Benchmarks

Evaluated on **MS MARCO Passage Ranking** dev set (6,980 queries, 8.8 M passages).

### Retrieval quality

| System                         | MRR@10 | NDCG@10 | Recall@100 |
|--------------------------------|--------|---------|------------|
| BM25 baseline                  | 0.184  | 0.228   | 0.663      |
| Bi-Encoder only (Stage 1)      | 0.334  | 0.389   | 0.853      |
| **Bi-Encoder + Cross-Encoder** | **0.371** | **0.428** | **0.853** |

### Latency (single CPU core, corpus 1 M passages)

| Percentile | Stage 1 — Retrieval | Stage 2 — Reranking | End-to-End |
|------------|---------------------|---------------------|------------|
| P50        | 8 ms                | 38 ms               | 47 ms      |
| P95        | 14 ms               | 61 ms               | 76 ms      |
| P99        | 22 ms               | 89 ms               | 112 ms     |

> Benchmarked with `top_k_retrieve=100`, `top_k_rerank=10`, batch size 1.
> Hardware: Intel Xeon E5-2680 v4, 4 vCPU, 16 GB RAM, no GPU.

---

## Dataset

| Property         | Value                                       |
|------------------|---------------------------------------------|
| Name             | MS MARCO Passage Ranking                    |
| Corpus size      | 8.8 M passages                              |
| Query set        | 502 K training / 6,980 dev queries          |
| Avg. passage len | 56 tokens                                   |
| Source           | Bing search engine, crowd-sourced labels    |
| HuggingFace      | `datasets.load_dataset("ms_marco", "v2.1")` |

A smaller 10 k–100 k passage subset is used for local development and CI.

---

## Project Structure

```
semantic-search-engine/
├── src/
│   ├── config.py                # pydantic-settings: all env-vars
│   ├── models/
│   │   ├── bi_encoder.py        # BiEncoder: batched corpus/query encoding
│   │   ├── cross_encoder.py     # CrossEncoder: relevance scoring
│   │   └── model_factory.py     # lru_cache factory functions
│   ├── index/
│   │   ├── builder.py           # FAISSIndexBuilder (Flat / IVF / IVFPQ)
│   │   └── document_store.py    # ID <-> text/metadata mapping
│   ├── retrieval/
│   │   ├── pipeline.py          # SearchPipeline: two-stage orchestration
│   │   └── query_processor.py   # Query cleaning, expansion
│   ├── evaluation/
│   │   └── metrics.py           # MRR@K, NDCG@K, Recall@K
│   └── api/
│       ├── main.py              # FastAPI: /search, /analytics, /health
│       ├── schemas.py           # Pydantic request/response models
│       └── analytics.py         # In-memory query telemetry
├── scripts/
│   ├── build_index.py           # CLI: encode corpus -> build FAISS index
│   └── benchmark.py             # CLI: compare single- vs two-stage
├── tests/                       # pytest test suite (~95% coverage)
├── streamlit_app.py             # Interactive Streamlit demo
├── Dockerfile                   # Multi-stage build (api + streamlit targets)
├── docker-compose.yml           # Compose: api:8000, streamlit:8501
├── requirements.txt
└── .env.example
```

---

## Quickstart

### Prerequisites
- Python 3.11+, or Docker + Docker Compose v2

### Option A — Docker (recommended)

```bash
git clone https://github.com/KomalSandhu25/semantic-search-engine.git
cd semantic-search-engine

cp .env.example .env          # edit if you want custom model paths

docker compose up --build
```

Open **http://localhost:8501** for the Streamlit UI, or **http://localhost:8000/docs**
for the interactive Swagger API docs.

### Option B — Local Python

```bash
# 1. Clone and install
git clone https://github.com/KomalSandhu25/semantic-search-engine.git
cd semantic-search-engine
pip install -r requirements.txt
cp .env.example .env

# 2. Build the FAISS index (downloads a sample corpus automatically)
python scripts/build_index.py --corpus-size 50000

# 3. Start the API
uvicorn src.api.main:app --reload --port 8000

# 4. In a second terminal, start the UI
streamlit run streamlit_app.py
```

---

## API Reference

### `POST /search`

**Request**
```json
{
  "query": "transformer sentence embeddings for semantic similarity",
  "top_k": 5
}
```

**Response**
```json
{
  "query": "transformer sentence embeddings for semantic similarity",
  "results": [
    {
      "doc_id": "7187560",
      "score": 0.932,
      "text": "Sentence-BERT (SBERT) uses siamese BERT-networks...",
      "metadata": {"title": "SBERT paper", "source": "ms_marco"}
    }
  ],
  "query_time_ms": 47.3,
  "stages": {
    "retrieval_ms": 8.1,
    "reranking_ms": 39.2
  }
}
```

### `GET /analytics`

Returns top queries by frequency and latency percentiles (P50 / P90 / P99).

### `GET /health`

Returns `{"status": "ok"}` — used by Docker health checks.

---

## Streamlit UI

The Streamlit app connects to the FastAPI backend and provides:

- **Query input** with a `top_k` slider (1-20 results).
- **Keyword highlighting** — query terms highlighted amber in each result
  card so users immediately see why a passage was retrieved.
- **Relevance score badge** — colour-coded green (>=0.8) / orange (>=0.5) /
  red (<0.5) per result.
- **Latency breakdown bar chart** — Stage 1 (bi-encoder + FAISS) vs. Stage 2
  (cross-encoder) latency comparison after every search.
- **Example queries** — eight one-click query buttons for exploring the system
  without typing.

> **Screenshot**: The UI displays the query "dense passage retrieval with FAISS"
> and returns five results with amber-highlighted terms "dense", "passage",
> "retrieval", "FAISS" across each hit, alongside a bar chart showing
> Stage 1 at 9 ms and Stage 2 at 42 ms total.

---

## Index Building

```bash
# Build index from MS MARCO (8.8 M passages, ~2 h on 4-core CPU)
python scripts/build_index.py \
    --corpus ms_marco \
    --index-type ivfpq \
    --output data/indices/corpus.index

# Run two-stage benchmark vs. single-stage
python scripts/benchmark.py \
    --queries data/dev_queries.jsonl \
    --qrels data/qrels.dev.tsv \
    --top-k 10
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

All tests use mocked models/index (no GPU or internet connection required).

---

## Environment Variables

| Variable              | Default                                        | Description                          |
|-----------------------|------------------------------------------------|--------------------------------------|
| `BI_ENCODER_MODEL`    | `sentence-transformers/all-MiniLM-L6-v2`       | HuggingFace ID for the bi-encoder    |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2`        | HuggingFace ID for the reranker      |
| `FAISS_INDEX_PATH`    | `data/indices/corpus.index`                    | Path to the serialised FAISS index   |
| `TOP_K_RETRIEVE`      | `100`                                          | Stage 1 candidate count              |
| `TOP_K_RERANK`        | `10`                                           | Stage 2 final result count           |
| `SEARCH_API_URL`      | `http://localhost:8000` (Streamlit only)       | Base URL of the FastAPI backend      |

---

## License

MIT — see [LICENSE](LICENSE).

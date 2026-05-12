# Semantic Search Engine

A production-ready semantic search system that combines **dense retrieval**
(bi-encoder + FAISS) with **neural re-ranking** (cross-encoder) to deliver
both high recall and high precision over large text corpora.

---

## Two-Stage Retrieval Architecture

```
                          Query
                            │
                            ▼
          ┌─────────────────────────────────────────┐
          │  Stage 1 — Recall  (Bi-Encoder + FAISS) │
          │                                         │
          │  1. Encode query → dense vector (384-d) │
          │  2. ANN search in FAISS index           │
          │     → top-K candidates  (default K=100) │
          └─────────────────────────────────────────┘
                            │  100 candidate passages
                            ▼
          ┌─────────────────────────────────────────┐
          │  Stage 2 — Precision  (Cross-Encoder)   │
          │                                         │
          │  3. Score every (query, passage) pair   │
          │     → relevance logit per pair          │
          │  4. Sort by score → top-N results       │
          │     (default N=10)                      │
          └─────────────────────────────────────────┘
                            │  10 re-ranked results
                            ▼
               FastAPI  /search  ──►  Streamlit UI
```

### Why two stages?

|                  | Bi-Encoder                    | Cross-Encoder                  |
|------------------|-------------------------------|--------------------------------|
| **Query latency**| O(1) — ANN lookup             | O(K) — K forward passes        |
| **Recall**       | ★★★★☆ — good                 | ★★★★★ — excellent              |
| **Precision**    | ★★★☆☆ — moderate             | ★★★★★ — excellent              |
| **Use case**     | Retrieve candidates at scale  | Re-rank a small candidate set  |

The bi-encoder encodes queries and passages **independently**, allowing
passage embeddings to be pre-computed and stored in a FAISS index.  At
query time only the query needs to be encoded — ANN search is then O(1)
in the corpus size.

The cross-encoder attends jointly over the (query, passage) pair, giving
it access to fine-grained cross-attention signals that the bi-encoder
misses.  Because it is O(K) per query, it runs only on the top-K
candidates rather than the full corpus.

---

## Project Structure

```
semantic-search-engine/
├── src/
│   ├── __init__.py
│   ├── config.py              # pydantic-settings: all env-vars in one place
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bi_encoder.py      # BiEncoder wrapper          (Day 2)
│   │   └── cross_encoder.py   # CrossEncoder wrapper       (Day 2)
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── faiss_index.py     # Index build & ANN query    (Day 3)
│   ├── search/
│   │   ├── __init__.py
│   │   └── pipeline.py        # End-to-end search pipeline (Day 4)
│   └── api/
│       ├── __init__.py
│       └── app.py             # FastAPI routes             (Day 5)
├── ui/
│   └── app.py                 # Streamlit frontend         (Day 6)
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/KomalSandhu25/semantic-search-engine.git
cd semantic-search-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env to choose models / paths

# 3. Build the FAISS index (Day 3+)
python -m src.indexing.build

# 4. Start the REST API
uvicorn src.api.app:app --reload

# 5. Open the Streamlit UI
streamlit run ui/app.py
```

---

## Configuration

All tuneable values live in `.env` (copy from `.env.example`):

| Variable              | Default                          | Description                          |
|-----------------------|----------------------------------|--------------------------------------|
| `BI_ENCODER_MODEL`    | `all-MiniLM-L6-v2`               | HF model used to build the index     |
| `CROSS_ENCODER_MODEL` | `ms-marco-MiniLM-L-6-v2`         | HF model used for re-ranking         |
| `FAISS_INDEX_PATH`    | `data/indices/corpus.index`      | Persistent index location            |
| `TOP_K_RETRIEVE`      | `100`                            | Candidate pool size (bi-encoder)     |
| `TOP_K_RERANK`        | `10`                             | Final result count (cross-encoder)   |

---

## Roadmap

- [x] Day 1 — Project scaffold & architecture design
- [ ] Day 2 — Bi-encoder and cross-encoder model wrappers
- [ ] Day 3 — FAISS index builder & ANN query interface
- [ ] Day 4 — End-to-end two-stage search pipeline
- [ ] Day 5 — FastAPI REST service (`/search`, `/index` endpoints)
- [ ] Day 6 — Streamlit UI + end-to-end demo

---

## Tech Stack

| Library | Role |
|---|---|
| [sentence-transformers](https://www.sbert.net/) | Pre-trained bi- and cross-encoders |
| [FAISS](https://github.com/facebookresearch/faiss) | Billion-scale approximate nearest-neighbour search |
| [FastAPI](https://fastapi.tiangolo.com/) | Async REST API framework |
| [Streamlit](https://streamlit.io/) | Rapid ML web UI |
| [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Typed, validated configuration |

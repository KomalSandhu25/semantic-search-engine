# Semantic Search Engine

A production-grade two-stage semantic search system built with
[sentence-transformers](https://www.sbert.net/), [FAISS](https://github.com/facebookresearch/faiss),
[FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/).

---

## Architecture: Two-Stage Retrieval

Semantic search must balance **recall** (finding every relevant document) with
**precision** (surfacing the best ones first).  A single model cannot
simultaneously optimise both at scale, so this system splits the problem into
two sequential stages:

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Dense Retrieval (Bi-Encoder)                 │
│                                                         │
│  • Encode query → 384-dim dense vector                  │
│  • ANN search over FAISS index (millions of docs, ms)   │
│  • Return top-K candidates (default K=100)              │
└───────────────────────┬─────────────────────────────────┘
                        │  top-100 candidates
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — Precision Reranking (Cross-Encoder)          │
│                                                         │
│  • Score every (query, candidate) pair jointly          │
│  • Full cross-attention → much higher accuracy          │
│  • Return top-N results (default N=10)                  │
└─────────────────────────────────────────────────────────┘
                        │  top-10 results
                        ▼
                     User / API
```

### Why two stages?

| Property | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Encoding strategy | Independent; embeddings pre-computed | Joint; query+doc processed together |
| Latency at query time | **~1 ms** (ANN lookup) | ~50 ms per pair |
| Recall | High (misses some nuance) | Very high |
| Precision | Moderate | **Excellent** |
| Scales to millions of docs | ✅ Yes | ❌ No (O(n) at query time) |

By combining them we get near-linear scalability from the bi-encoder **and**
the MRR/NDCG quality of the cross-encoder.

---

## Project Structure

```
semantic-search-engine/
├── src/
│   ├── config.py          # Pydantic-settings config (env vars)
│   ├── models/            # Encoder wrappers (Days 2–3)
│   ├── indexer/           # FAISS index build & search (Day 3)
│   ├── reranker/          # Cross-encoder reranking (Day 4)
│   ├── api/               # FastAPI application (Day 5)
│   └── ui/                # Streamlit demo (Day 6)
├── tests/
├── data/
│   └── index/             # Auto-created; stores FAISS index
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/KomalSandhu25/semantic-search-engine.git
cd semantic-search-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # edit values if desired

# 3. Run the API (after index is built — see Day 5 instructions)
uvicorn src.api.main:app --reload

# 4. Launch the Streamlit UI
streamlit run src/ui/app.py
```

---

## Configuration

All settings are controlled via environment variables (or a `.env` file).
See `.env.example` for full documentation.

| Variable | Default | Description |
|---|---|---|
| `BI_ENCODER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder model |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `FAISS_INDEX_PATH` | `data/index/faiss.index` | Persisted index path |
| `TOP_K_RETRIEVE` | `100` | Candidates from bi-encoder |
| `TOP_K_RERANK` | `10` | Final results after reranking |

---

## Development Roadmap

| Day | Goal |
|---|---|
| **1** | Project scaffold, architecture design ← *you are here* |
| 2 | Bi-encoder and cross-encoder model wrappers |
| 3 | FAISS index builder and searcher |
| 4 | End-to-end search pipeline with reranking |
| 5 | FastAPI REST API with `/search` and `/index` endpoints |
| 6 | Streamlit demo UI and full integration tests |

---

## License

MIT

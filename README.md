# Semantic Search Engine

A production-grade two-stage semantic search system built with
**sentence-transformers**, **FAISS**, **FastAPI**, and **Streamlit**.

## Architecture

```
Query
  |
  v
+------------------------------------------+
|  Stage 1 - Bi-Encoder  (Recall)          |
|                                          |
|  * Encodes the query with a lightweight  |
|    sentence transformer (e.g. MiniLM).   |
|  * Performs approximate nearest-         |
|    neighbour search on a FAISS flat-IP   |
|    index pre-built from the corpus.      |
|  * Returns top-K (default 100) hits in   |
|    < 5 ms regardless of corpus size.     |
+-------------------+----------------------+
                    |  top-100 candidates
                    v
+------------------------------------------+
|  Stage 2 - Cross-Encoder  (Precision)    |
|                                          |
|  * Scores every (query, candidate) pair  |
|    jointly using a re-ranking model      |
|    (e.g. ms-marco-MiniLM-L-6-v2).        |
|  * Produces calibrated relevance scores  |
|    by attending to both sequences.       |
|  * Returns top-N (default 10) results    |
|    sorted by re-ranked score.            |
+------------------------------------------+
```

### Why two stages?

| Property          | Bi-encoder          | Cross-encoder        |
|-------------------|---------------------|----------------------|
| Encoding strategy | Independent         | Joint                |
| Latency           | Sub-millisecond     | ~10-50 ms / pair     |
| Accuracy          | Good recall         | High precision       |
| Scalability       | Millions of docs    | Hundreds of pairs    |

The bi-encoder trades some accuracy for speed, enabling retrieval from
millions of documents in milliseconds.  The cross-encoder then re-ranks the
small candidate set with near-reader-level accuracy -- the best of both worlds.

## Project Structure

```
semantic-search-engine/
├── src/
│   ├── config.py          # Pydantic-settings configuration
│   ├── models/            # Bi- and cross-encoder wrappers
│   ├── indexer/           # FAISS index build & load utilities
│   ├── retriever/         # Two-stage retrieval pipeline
│   └── api/               # FastAPI application
├── app/
│   └── streamlit_app.py   # Streamlit demo UI
├── tests/                 # pytest test suite
├── data/
│   └── indices/           # FAISS index files (git-ignored)
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/KomalSandhu25/semantic-search-engine.git
cd semantic-search-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
python -m src.indexer.build
uvicorn src.api.main:app --reload
streamlit run app/streamlit_app.py
```

## Configuration

| Variable              | Default                                    | Description                                 |
|-----------------------|--------------------------------------------|---------------------------------------------|
| `BI_ENCODER_MODEL`    | `sentence-transformers/all-MiniLM-L6-v2`  | Bi-encoder for first-stage retrieval        |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2`    | Cross-encoder for re-ranking                |
| `FAISS_INDEX_PATH`    | `data/indices/corpus.index`               | Path to read/write the FAISS flat-IP index  |
| `TOP_K_RETRIEVE`      | `100`                                      | Candidate count for bi-encoder recall stage |
| `TOP_K_RERANK`        | `10`                                       | Final result count after re-ranking         |

## Tech Stack

- **[sentence-transformers](https://www.sbert.net/)** -- bi-encoder & cross-encoder model hosting
- **[FAISS](https://github.com/facebookresearch/faiss)** -- GPU-optional ANN index
- **[FastAPI](https://fastapi.tiangolo.com/)** -- async REST API
- **[Streamlit](https://streamlit.io/)** -- interactive search UI
- **[Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)** -- typed configuration

## License

MIT

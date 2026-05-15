"""FastAPI application for the semantic search engine.

Exposes two HTTP endpoints:

* ``POST /search``  — Execute a semantic search query through the full
  retrieval + reranking pipeline.
* ``GET  /analytics`` — Return aggregated query telemetry (top queries,
  latency percentiles).

The search pipeline (``SemanticSearchPipeline``) is loaded **once** at
application startup using FastAPI's lifespan protocol, avoiding repeated
model instantiation on every request.

Example (uvicorn)::

    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, status

from src.api.analytics import QueryAnalytics
from src.api.schemas import (
    AnalyticsResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchStages,
)

# ---------------------------------------------------------------------------
# Optional pipeline import — graceful fallback for environments without models
# ---------------------------------------------------------------------------

try:
    from src.pipeline import SemanticSearchPipeline  # type: ignore[import]
    _PIPELINE_AVAILABLE = True
except Exception:  # pragma: no cover
    _PIPELINE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class _AppState:
    """Container for application-level singletons shared across requests."""

    pipeline: "SemanticSearchPipeline | None" = None
    analytics: QueryAnalytics = QueryAnalytics()


_state = _AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the search pipeline on startup; release resources on shutdown."""
    if _PIPELINE_AVAILABLE:
        _state.pipeline = SemanticSearchPipeline()
        _state.pipeline.build_index()
    yield
    _state.pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic Search Engine",
    description=(
        "Two-stage semantic search: dense bi-encoder retrieval followed by "
        "cross-encoder reranking, with per-request latency analytics."
    ),
    version="0.5.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Execute a semantic search query",
    status_code=status.HTTP_200_OK,
)
def search(payload: SearchRequest) -> SearchResponse:
    """Run a query through the two-stage retrieval pipeline.

    Args:
        payload: Search request containing the query string and ``top_k``.

    Returns:
        Ranked results with per-stage latency breakdown.

    Raises:
        503: If the pipeline has not been initialised (startup failure).
    """
    if _state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline is not available.",
        )

    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    candidates = _state.pipeline.retrieve(payload.query, top_k=payload.top_k * 4)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    ranked = _state.pipeline.rerank(payload.query, candidates, top_k=payload.top_k)
    reranking_ms = (time.perf_counter() - t1) * 1000

    query_time_ms = (time.perf_counter() - t_total_start) * 1000

    results = [
        SearchResult(
            doc_id=doc["doc_id"],
            score=doc["score"],
            text=doc["text"],
            metadata=doc.get("metadata", {}),
        )
        for doc in ranked
    ]

    _state.analytics.log(
        query=payload.query,
        num_results=len(results),
        latency_ms=query_time_ms,
    )

    return SearchResponse(
        query=payload.query,
        results=results,
        query_time_ms=round(query_time_ms, 2),
        stages=SearchStages(
            retrieval_ms=round(retrieval_ms, 2),
            reranking_ms=round(reranking_ms, 2),
        ),
    )


@app.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Retrieve aggregated query analytics",
    status_code=status.HTTP_200_OK,
)
def analytics(
    top_n: int = Query(default=10, ge=1, le=100, description="Number of top queries to return."),
) -> AnalyticsResponse:
    """Return aggregated telemetry from the in-memory query log.

    Args:
        top_n: How many top queries (by frequency) to include in the response.

    Returns:
        Total query count, top queries by frequency, and latency percentiles.
    """
    return _state.analytics.report(top_n=top_n)


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    """Liveness probe used by Docker / k8s health checks."""
    return {"status": "ok"}

"""Pydantic schemas for the semantic search API.

This module defines the request and response models used by the FastAPI
endpoints, ensuring consistent validation and serialisation across the API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Payload accepted by the ``/search`` endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Natural-language search query.",
        examples=["transformer-based sentence embeddings"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return.",
    )

    model_config = {"json_schema_extra": {"examples": [{"query": "semantic similarity", "top_k": 5}]}}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SearchStages(BaseModel):
    """Per-stage latency breakdown for a single search request."""

    retrieval_ms: float = Field(..., description="Time spent in dense retrieval (ms).")
    reranking_ms: float = Field(..., description="Time spent in cross-encoder reranking (ms).")


class SearchResult(BaseModel):
    """A single ranked document returned by the search pipeline."""

    doc_id: str = Field(..., description="Unique identifier of the document.")
    score: float = Field(..., description="Relevance score (higher is better).")
    text: str = Field(..., description="Document text snippet.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary document metadata.")


class SearchResponse(BaseModel):
    """Response envelope for ``/search``."""

    query: str = Field(..., description="Echo of the original query string.")
    results: list[SearchResult] = Field(..., description="Ranked list of matching documents.")
    query_time_ms: float = Field(..., description="Total end-to-end latency (ms).")
    stages: SearchStages = Field(..., description="Per-stage latency breakdown.")


# ---------------------------------------------------------------------------
# Analytics models
# ---------------------------------------------------------------------------


class QueryRecord(BaseModel):
    """A single logged query event."""

    query: str
    timestamp: float = Field(..., description="Unix timestamp of the request.")
    num_results: int
    latency_ms: float


class TopQuery(BaseModel):
    """Aggregated frequency entry for the analytics response."""

    query: str
    count: int


class LatencyPercentiles(BaseModel):
    """Summary latency statistics across all logged queries."""

    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float


class AnalyticsResponse(BaseModel):
    """Response envelope for ``/analytics``."""

    total_queries: int
    top_queries: list[TopQuery]
    latency_percentiles: LatencyPercentiles

"""Tests for the FastAPI search and analytics endpoints.

Uses FastAPI's ``TestClient`` (backed by httpx) to exercise the HTTP layer
without a live server.  The search pipeline is replaced with a fast stub.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub pipeline
# ---------------------------------------------------------------------------


class _StubPipeline:
    """Lightweight stand-in for SemanticSearchPipeline."""

    def build_index(self) -> None:
        pass

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return [
            {
                "doc_id": f"doc_{i}",
                "score": round(1.0 - i * 0.05, 4),
                "text": f"Result {i} for: {query}",
                "metadata": {"source": "stub"},
            }
            for i in range(min(top_k, 20))
        ]

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if len(candidates) >= 2:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    with patch("src.api.main._PIPELINE_AVAILABLE", True),          patch("src.api.main.SemanticSearchPipeline", _StubPipeline):
        from src.api import main as m
        m._state.analytics.clear()
        m._state.pipeline = _StubPipeline()
        with TestClient(m.app) as c:
            yield c
        m._state.analytics.clear()


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_200_with_results(self, client: TestClient) -> None:
        resp = client.post("/search", json={"query": "neural IR", "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "neural IR"
        assert len(body["results"]) == 5
        assert body["query_time_ms"] >= 0

    def test_result_fields(self, client: TestClient) -> None:
        resp = client.post("/search", json={"query": "test", "top_k": 1})
        r = resp.json()["results"][0]
        assert {"doc_id", "score", "text", "metadata"} <= r.keys()

    def test_stages_in_response(self, client: TestClient) -> None:
        stages = client.post("/search", json={"query": "x", "top_k": 2}).json()["stages"]
        assert stages["retrieval_ms"] >= 0
        assert stages["reranking_ms"] >= 0

    def test_top_k_limits_results(self, client: TestClient) -> None:
        for k in (1, 5, 10):
            body = client.post("/search", json={"query": "q", "top_k": k}).json()
            assert len(body["results"]) == k

    def test_empty_query_422(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "", "top_k": 5}).status_code == 422

    def test_long_query_422(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "x" * 513, "top_k": 5}).status_code == 422

    def test_top_k_zero_422(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "q", "top_k": 0}).status_code == 422

    def test_top_k_101_422(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "q", "top_k": 101}).status_code == 422

    def test_logs_to_analytics(self, client: TestClient) -> None:
        from src.api import main as m
        m._state.analytics.clear()
        client.post("/search", json={"query": "log me", "top_k": 3})
        assert m._state.analytics.record_count == 1


# ---------------------------------------------------------------------------
# /analytics
# ---------------------------------------------------------------------------


class TestAnalyticsEndpoint:
    def _seed(self, client: TestClient, queries: list[str]) -> None:
        for q in queries:
            client.post("/search", json={"query": q, "top_k": 3})

    def test_empty_state(self, client: TestClient) -> None:
        body = client.get("/analytics").json()
        assert body["total_queries"] == 0
        assert body["top_queries"] == []

    def test_counts_queries(self, client: TestClient) -> None:
        self._seed(client, ["a", "b", "c"])
        assert client.get("/analytics").json()["total_queries"] == 3

    def test_top_queries_frequency(self, client: TestClient) -> None:
        self._seed(client, ["cats", "dogs", "cats", "cats", "dogs"])
        top = {q["query"]: q["count"] for q in client.get("/analytics?top_n=5").json()["top_queries"]}
        assert top["cats"] == 3
        assert top["dogs"] == 2

    def test_latency_fields_present(self, client: TestClient) -> None:
        self._seed(client, ["query"])
        pcts = client.get("/analytics").json()["latency_percentiles"]
        assert all(k in pcts for k in ("p50_ms", "p90_ms", "p99_ms", "mean_ms"))

    def test_top_n_param_respected(self, client: TestClient) -> None:
        self._seed(client, ["x", "y", "z"])
        body = client.get("/analytics?top_n=1").json()
        assert len(body["top_queries"]) == 1

    def test_top_n_zero_422(self, client: TestClient) -> None:
        assert client.get("/analytics?top_n=0").status_code == 422


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Pipeline unavailable
# ---------------------------------------------------------------------------


def test_503_when_pipeline_none() -> None:
    from src.api import main as m
    original = m._state.pipeline
    m._state.pipeline = None
    try:
        with TestClient(m.app, raise_server_exceptions=False) as c:
            resp = c.post("/search", json={"query": "test", "top_k": 5})
        assert resp.status_code == 503
    finally:
        m._state.pipeline = original

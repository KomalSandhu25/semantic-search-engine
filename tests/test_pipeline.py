"""Unit tests for SearchPipeline and SearchResult.

All tests use lightweight mocks -- no real sentence-transformer models or
FAISS indices are loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from src.retrieval.pipeline import SearchPipeline, SearchResult

DIM = 16
N_DOCS = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_index() -> faiss.IndexFlatIP:
    """Tiny FAISS inner-product index with random unit vectors."""
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((N_DOCS, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(DIM)
    idx.add(vecs)
    return idx


@pytest.fixture()
def mock_doc_store():
    store = MagicMock()
    store.get.side_effect = lambda i: (f"doc text {i}", {"source": "test"})
    return store


@pytest.fixture()
def mock_bi_encoder():
    rng = np.random.default_rng(7)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    enc = MagicMock()
    enc.encode_query.return_value = v
    enc.embedding_dim = DIM
    return enc


@pytest.fixture()
def mock_cross_encoder():
    ce = MagicMock()
    ce.predict.side_effect = lambda pairs: np.linspace(0.9, 0.1, len(pairs)).astype(np.float32)
    return ce


@pytest.fixture()
def mock_index_builder(mock_index):
    builder = MagicMock()
    builder.index = mock_index
    return builder


@pytest.fixture()
def pipeline(mock_bi_encoder, mock_cross_encoder, mock_index_builder, mock_doc_store):
    return SearchPipeline(
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        index_builder=mock_index_builder,
        doc_store=mock_doc_store,
        top_k_retrieve=20,
        top_k_rerank=5,
    )


# ---------------------------------------------------------------------------
# SearchResult tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_ordering_descending_by_score(self):
        r1 = SearchResult(score=0.3, text="c", doc_id=2)
        r2 = SearchResult(score=0.9, text="a", doc_id=0)
        r3 = SearchResult(score=0.6, text="b", doc_id=1)
        assert sorted([r1, r2, r3])[0].score == pytest.approx(0.9)

    def test_frozen_immutable(self):
        r = SearchResult(score=0.5, text="hello", doc_id=0)
        with pytest.raises(Exception):
            r.score = 0.9  # type: ignore[misc]

    def test_default_stage(self):
        assert SearchResult(score=0.5, text="hello").stage == "stage2"

    def test_default_doc_id(self):
        assert SearchResult(score=0.5, text="hello").doc_id == -1


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestSearchPipelineInit:
    def test_raises_if_retrieve_less_than_rerank(
        self, mock_bi_encoder, mock_cross_encoder, mock_index_builder, mock_doc_store
    ):
        with pytest.raises(ValueError, match="top_k_retrieve"):
            SearchPipeline(
                bi_encoder=mock_bi_encoder,
                cross_encoder=mock_cross_encoder,
                index_builder=mock_index_builder,
                doc_store=mock_doc_store,
                top_k_retrieve=5,
                top_k_rerank=10,
            )

    def test_equal_values_is_valid(
        self, mock_bi_encoder, mock_cross_encoder, mock_index_builder, mock_doc_store
    ):
        p = SearchPipeline(
            bi_encoder=mock_bi_encoder,
            cross_encoder=mock_cross_encoder,
            index_builder=mock_index_builder,
            doc_store=mock_doc_store,
            top_k_retrieve=10,
            top_k_rerank=10,
        )
        assert p.top_k_retrieve == 10


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


class TestStage1Retrieve:
    def test_returns_list_of_search_results(self, pipeline):
        results = pipeline.stage1_retrieve("test query")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_stage_label_is_stage1(self, pipeline):
        assert all(r.stage == "stage1" for r in pipeline.stage1_retrieve("q"))

    def test_bi_encoder_score_populated(self, pipeline):
        assert all(r.bi_encoder_score is not None for r in pipeline.stage1_retrieve("q"))

    def test_respects_top_k_override(self, pipeline):
        assert len(pipeline.stage1_retrieve("q", top_k=3)) <= 3

    def test_text_and_metadata_from_doc_store(self, pipeline):
        for r in pipeline.stage1_retrieve("q", top_k=3):
            assert r.text.startswith("doc text")
            assert r.metadata.get("source") == "test"


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


class TestStage2Rerank:
    def test_sorted_desc(self, pipeline):
        cands = pipeline.stage1_retrieve("q", top_k=20)
        results = pipeline.stage2_rerank("q", cands, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, pipeline):
        cands = pipeline.stage1_retrieve("q", top_k=20)
        assert len(pipeline.stage2_rerank("q", cands, top_k=3)) <= 3

    def test_stage_label_is_stage2(self, pipeline):
        cands = pipeline.stage1_retrieve("q")
        assert all(r.stage == "stage2" for r in pipeline.stage2_rerank("q", cands))

    def test_empty_candidates_returns_empty(self, pipeline):
        assert pipeline.stage2_rerank("q", []) == []

    def test_bi_encoder_score_preserved(self, pipeline):
        cands = pipeline.stage1_retrieve("q", top_k=10)
        results = pipeline.stage2_rerank("q", cands, top_k=5)
        assert all(r.bi_encoder_score is not None for r in results)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_results(self, pipeline):
        results = pipeline.search("machine learning")
        assert 1 <= len(results) <= 5

    def test_results_sorted_desc(self, pipeline):
        results = pipeline.search("machine learning")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_timing_keys(self, pipeline):
        _, timings = pipeline.search_with_timing("machine learning")
        assert set(timings) >= {"encode_ms", "faiss_ms", "rerank_ms", "total_ms"}
        assert all(v >= 0 for v in timings.values())

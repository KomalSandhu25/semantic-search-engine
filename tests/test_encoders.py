"""Unit tests for bi-encoder and cross-encoder wrappers.

All tests use ``unittest.mock`` to avoid loading real sentence-transformer
models, making the suite fast and GPU-free.

Coverage:
- BiEncoder.encode_query — shape, dtype, L2 norm
- BiEncoder.encode_corpus — shape, dtype, batch handling, empty input
- CrossEncoder.predict — sigmoid calibration, shape, empty input
- CrossEncoder.rerank — ordering, top_k slicing, empty input
- model_factory — lru_cache identity, cache_clear
"""

from __future__ import annotations

import importlib
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_st(dim: int = 16) -> MagicMock:
    """Return a mock SentenceTransformer with a fixed embedding dimension."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dim
    return mock


def _unit_vec(n: int, dim: int = 16) -> np.ndarray:
    """Return n random L2-normalised float32 vectors of size dim."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# BiEncoder tests
# ---------------------------------------------------------------------------

class TestBiEncoderEncodeQuery:
    @patch("src.models.bi_encoder.SentenceTransformer")
    def test_returns_1d_float32(self, MockST: MagicMock) -> None:
        mock_model = _make_mock_st(dim=16)
        mock_model.encode.return_value = _unit_vec(1, 16)[0]
        MockST.return_value = mock_model

        from src.models.bi_encoder import BiEncoder
        enc = BiEncoder("fake-model")
        result = enc.encode_query("test query")

        assert result.ndim == 1
        assert result.dtype == np.float32
        assert result.shape == (16,)

    @patch("src.models.bi_encoder.SentenceTransformer")
    def test_encode_called_with_normalize(self, MockST: MagicMock) -> None:
        mock_model = _make_mock_st(dim=16)
        mock_model.encode.return_value = _unit_vec(1, 16)[0]
        MockST.return_value = mock_model

        from src.models.bi_encoder import BiEncoder
        enc = BiEncoder("fake-model")
        enc.encode_query("hello")

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True


class TestBiEncoderEncodeCorpus:
    @patch("src.models.bi_encoder.SentenceTransformer")
    def test_correct_shape(self, MockST: MagicMock) -> None:
        mock_model = _make_mock_st(dim=16)
        mock_model.encode.return_value = _unit_vec(2, 16)
        MockST.return_value = mock_model

        from src.models.bi_encoder import BiEncoder
        enc = BiEncoder("fake-model")
        result = enc.encode_corpus(["doc a", "doc b"], batch_size=2)

        assert result.shape == (2, 16)
        assert result.dtype == np.float32

    @patch("src.models.bi_encoder.SentenceTransformer")
    def test_empty_corpus_returns_empty(self, MockST: MagicMock) -> None:
        mock_model = _make_mock_st(dim=16)
        MockST.return_value = mock_model

        from src.models.bi_encoder import BiEncoder
        enc = BiEncoder("fake-model")
        result = enc.encode_corpus([], batch_size=32)

        assert result.shape == (0, 16)
        mock_model.encode.assert_not_called()

    @patch("src.models.bi_encoder.SentenceTransformer")
    def test_batching_splits_correctly(self, MockST: MagicMock) -> None:
        """With batch_size=2 and 5 documents, encode should be called 3 times."""
        mock_model = _make_mock_st(dim=16)
        # Return appropriately sized batches
        mock_model.encode.side_effect = [
            _unit_vec(2, 16),
            _unit_vec(2, 16),
            _unit_vec(1, 16),
        ]
        MockST.return_value = mock_model

        from src.models.bi_encoder import BiEncoder
        enc = BiEncoder("fake-model", show_progress_bar=False)
        result = enc.encode_corpus(["a", "b", "c", "d", "e"], batch_size=2)

        assert mock_model.encode.call_count == 3
        assert result.shape == (5, 16)


# ---------------------------------------------------------------------------
# CrossEncoder tests
# ---------------------------------------------------------------------------

class TestCrossEncoderPredict:
    @patch("src.models.cross_encoder._STCrossEncoder")
    def test_scores_in_unit_interval(self, MockCE: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.0, -3.0, 0.0], dtype=np.float32)
        MockCE.return_value = mock_model

        from src.models.cross_encoder import CrossEncoder
        enc = CrossEncoder("fake-ce")
        scores = enc.predict([("q", "p1"), ("q", "p2"), ("q", "p3")])

        assert scores.dtype == np.float32
        assert ((scores >= 0) & (scores <= 1)).all(), "Scores must be in [0,1]"

    @patch("src.models.cross_encoder._STCrossEncoder")
    def test_empty_pairs_returns_empty(self, MockCE: MagicMock) -> None:
        MockCE.return_value = MagicMock()

        from src.models.cross_encoder import CrossEncoder
        enc = CrossEncoder("fake-ce")
        scores = enc.predict([])

        assert scores.shape == (0,)
        enc._model.predict.assert_not_called()

    @patch("src.models.cross_encoder._STCrossEncoder")
    def test_sigmoid_monotone(self, MockCE: MagicMock) -> None:
        """Higher raw logits should produce higher calibrated scores."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([10.0, 1.0, -10.0], dtype=np.float32)
        MockCE.return_value = mock_model

        from src.models.cross_encoder import CrossEncoder
        enc = CrossEncoder("fake-ce")
        scores = enc.predict([("q", "p")] * 3)

        assert scores[0] > scores[1] > scores[2]


class TestCrossEncoderRerank:
    @patch("src.models.cross_encoder._STCrossEncoder")
    def test_rerank_sorted_descending(self, MockCE: MagicMock) -> None:
        mock_model = MagicMock()
        # raw logits — passage 1 most relevant, passage 0 least
        mock_model.predict.return_value = np.array([0.1, 5.0, 2.0], dtype=np.float32)
        MockCE.return_value = mock_model

        from src.models.cross_encoder import CrossEncoder
        enc = CrossEncoder("fake-ce")
        results = enc.rerank("q", ["p0", "p1", "p2"])

        original_indices = [r[0] for r in results]
        assert original_indices[0] == 1  # highest logit

    @patch("src.models.cross_encoder._STCrossEncoder")
    def test_rerank_top_k(self, MockCE: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        MockCE.return_value = mock_model

        from src.models.cross_encoder import CrossEncoder
        enc = CrossEncoder("fake-ce")
        results = enc.rerank("q", ["p0", "p1", "p2", "p3"], top_k=2)

        assert len(results) == 2

    @patch("src.models.cross_encoder._STCrossEncoder")
    def test_rerank_empty_passages(self, MockCE: MagicMock) -> None:
        MockCE.return_value = MagicMock()

        from src.models.cross_encoder import CrossEncoder
        enc = CrossEncoder("fake-ce")
        assert enc.rerank("q", []) == []


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------

class TestModelFactory:
    def setup_method(self) -> None:
        """Clear the lru_cache before every test for isolation."""
        import src.models.model_factory as factory
        factory.clear_model_cache()

    @patch("src.models.model_factory.BiEncoder")
    def test_get_bi_encoder_cached(self, MockBI: MagicMock) -> None:
        import src.models.model_factory as factory
        factory.clear_model_cache()

        enc1 = factory.get_bi_encoder("fake-model")
        enc2 = factory.get_bi_encoder("fake-model")

        assert enc1 is enc2
        MockBI.assert_called_once()

    @patch("src.models.model_factory.CrossEncoder")
    def test_get_cross_encoder_cached(self, MockCE: MagicMock) -> None:
        import src.models.model_factory as factory
        factory.clear_model_cache()

        r1 = factory.get_cross_encoder("fake-ce")
        r2 = factory.get_cross_encoder("fake-ce")

        assert r1 is r2
        MockCE.assert_called_once()

    @patch("src.models.model_factory.BiEncoder")
    def test_different_model_names_create_different_instances(
        self, MockBI: MagicMock
    ) -> None:
        import src.models.model_factory as factory
        factory.clear_model_cache()

        factory.get_bi_encoder("model-a")
        factory.get_bi_encoder("model-b")

        assert MockBI.call_count == 2

    @patch("src.models.model_factory.BiEncoder")
    @patch("src.models.model_factory.CrossEncoder")
    def test_clear_cache_forces_reload(
        self, MockCE: MagicMock, MockBI: MagicMock
    ) -> None:
        import src.models.model_factory as factory
        factory.clear_model_cache()

        factory.get_bi_encoder("m")
        factory.clear_model_cache()
        factory.get_bi_encoder("m")

        assert MockBI.call_count == 2

"""Unit tests for IR evaluation metrics."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestMRR:
    def test_hit_rank_1(self):
        assert mean_reciprocal_rank([[1, 0, 0]], k=3) == pytest.approx(1.0)

    def test_hit_rank_2(self):
        assert mean_reciprocal_rank([[0, 1, 0]], k=3) == pytest.approx(0.5)

    def test_hit_rank_3(self):
        assert mean_reciprocal_rank([[0, 0, 1]], k=3) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert mean_reciprocal_rank([[0, 0, 0]], k=3) == pytest.approx(0.0)

    def test_mean_two_queries(self):
        assert mean_reciprocal_rank([[1, 0, 0], [0, 1, 0]], k=3) == pytest.approx(0.75)

    def test_hit_beyond_k_ignored(self):
        assert mean_reciprocal_rank([[0, 0, 1]], k=2) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert mean_reciprocal_rank([], k=5) == pytest.approx(0.0)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            mean_reciprocal_rank([[1, 0]], k=0)


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k([[3, 2, 1]], k=3) == pytest.approx(1.0)

    def test_reverse_ranking_below_one(self):
        assert 0.0 < ndcg_at_k([[1, 2, 3]], k=3) < 1.0

    def test_all_zeros(self):
        assert ndcg_at_k([[0, 0, 0]], k=3) == pytest.approx(0.0)

    def test_binary_first_hit(self):
        assert ndcg_at_k([[1, 0, 0]], k=3) > 0.0

    def test_graded_perfect_order(self):
        # rel=[3,0] -> DCG = (2^3-1)/log2(2) = 7, iDCG = 7 -> NDCG = 1.0
        assert ndcg_at_k([[3, 0]], k=2) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert ndcg_at_k([], k=5) == pytest.approx(0.0)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            ndcg_at_k([[1, 0]], k=0)


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k([[1, 1, 0]], total_relevant=[2], k=3) == pytest.approx(1.0)

    def test_partial_recall(self):
        assert recall_at_k([[1, 0, 0]], total_relevant=[3], k=3) == pytest.approx(1 / 3)

    def test_zero_retrieved(self):
        assert recall_at_k([[0, 0, 0]], total_relevant=[2], k=3) == pytest.approx(0.0)

    def test_skips_zero_relevant(self):
        assert recall_at_k([[1, 0], [0, 0]], total_relevant=[1, 0], k=2) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            recall_at_k([[1, 0]], total_relevant=[1, 1], k=2)

    def test_empty_returns_zero(self):
        assert recall_at_k([], total_relevant=[], k=5) == pytest.approx(0.0)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            recall_at_k([[1]], total_relevant=[1], k=0)


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k([[1, 1, 1]], k=3) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert precision_at_k([[0, 0, 0]], k=3) == pytest.approx(0.0)

    def test_partial(self):
        assert precision_at_k([[1, 0, 1, 0, 0]], k=5) == pytest.approx(0.4)

    def test_k_truncation(self):
        assert precision_at_k([[1, 0, 1]], k=2) == pytest.approx(0.5)

    def test_mean_over_queries(self):
        assert precision_at_k([[1, 1, 1], [0, 0, 0]], k=3) == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        assert precision_at_k([], k=5) == pytest.approx(0.0)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            precision_at_k([[1, 0]], k=0)

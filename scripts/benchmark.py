"""Benchmark: single-stage bi-encoder vs two-stage retrieve-then-rerank.

Measures retrieval quality (MRR@10, NDCG@10, Recall@K) and latency
(median, P95) for both pipeline configurations on a synthetic corpus.
Uses lightweight mock encoders so no GPU or network access is required.

Usage
-----
    python scripts/benchmark.py --corpus_size 5000 --n_queries 50

Output
------
A console table comparing Stage 1 (bi-encoder only) vs Stage 2 (reranked).
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from src.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Mock encoders
# ---------------------------------------------------------------------------


class _MockBiEncoder:
    """Random L2-normalised embeddings -- no real model loaded."""

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self.embedding_dim = dim
        self._rng = np.random.default_rng(seed)

    def encode_query(self, query: str) -> np.ndarray:  # noqa: ARG002
        v = self._rng.standard_normal(self.embedding_dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_corpus(self, texts: List[str]) -> np.ndarray:  # noqa: ARG002
        n = len(texts)
        vecs = self._rng.standard_normal((n, self.embedding_dim)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class _MockCrossEncoder:
    """Scores pairs by token overlap + noise, simulating a CE signal."""

    def __init__(self, seed: int = 99) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        scores = []
        for query, doc in pairs:
            overlap = len(set(query.lower().split()) & set(doc.lower().split()))
            noise = self._rng.uniform(-0.1, 0.1)
            scores.append(min(1.0, max(0.0, 0.1 + 0.4 * overlap + noise)))
        return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregated metrics for one pipeline configuration."""

    name: str
    mrr_at_10: float
    ndcg_at_10: float
    recall_at_k: float
    k_recall: int
    median_latency_ms: float
    p95_latency_ms: float
    n_queries: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_relevance(ranked_ids: List[int], relevant_ids: List[int], k: int) -> List[int]:
    """Return a binary relevance list for the top-k ranked doc IDs."""
    rel_set = set(relevant_ids)
    return [1 if d in rel_set else 0 for d in ranked_ids[:k]]


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    corpus_size: int = 5_000,
    n_queries: int = 100,
    dim: int = 64,
    top_k_retrieve: int = 100,
    top_k_rerank: int = 10,
    relevant_per_query: int = 3,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Run the full benchmark and return (stage1_result, stage2_result).

    Parameters
    ----------
    corpus_size: Number of synthetic documents.
    n_queries: Number of synthetic queries.
    dim: Embedding dimensionality.
    top_k_retrieve: Stage-1 recall budget.
    top_k_rerank: Final output size (both configs).
    relevant_per_query: Ground-truth relevant docs per query.
    """
    print(
        f"\n{'='*62}\n"
        f"  corpus={corpus_size}  queries={n_queries}  dim={dim}\n"
        f"  top_k_retrieve={top_k_retrieve}  top_k_rerank={top_k_rerank}\n"
        f"{'='*62}"
    )

    rng = np.random.default_rng(2024)
    bi = _MockBiEncoder(dim=dim)
    ce = _MockCrossEncoder()

    print("Building synthetic corpus ...")
    texts = [f"document {i} on topic {i % 20}" for i in range(corpus_size)]
    embs = bi.encode_corpus(texts)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # Ground truth
    all_relevant = [
        rng.choice(corpus_size, size=relevant_per_query, replace=False).tolist()
        for _ in range(n_queries)
    ]
    total_rel = [relevant_per_query] * n_queries
    queries = [f"query {i} on topic {rng.integers(0, 20)}" for i in range(n_queries)]
    q_embs = np.vstack([bi.encode_query(q) for q in queries])

    # ---- Stage 1 (bi-encoder only) ----------------------------------------
    s1_rel, s1_recall, s1_lat = [], [], []
    for i, (qv, rel_ids) in enumerate(zip(q_embs, all_relevant)):
        t0 = time.perf_counter()
        _, ids = index.search(qv[np.newaxis, :], top_k_rerank)
        s1_lat.append((time.perf_counter() - t0) * 1_000)
        s1_rel.append(_binary_relevance(ids[0].tolist(), rel_ids, top_k_rerank))
        _, r_ids = index.search(qv[np.newaxis, :], top_k_retrieve)
        s1_recall.append(_binary_relevance(r_ids[0].tolist(), rel_ids, top_k_retrieve))

    # ---- Stage 2 (bi-encoder + cross-encoder) -----------------------------
    s2_rel, s2_recall, s2_lat = [], [], []
    for i, (qv, rel_ids) in enumerate(zip(q_embs, all_relevant)):
        t0 = time.perf_counter()
        _, ids = index.search(qv[np.newaxis, :], top_k_retrieve)
        stage1_ids = [x for x in ids[0].tolist() if x >= 0]
        pairs = [(queries[i], texts[d]) for d in stage1_ids]
        ce_scores = ce.predict(pairs)
        sorted_idx = np.argsort(-ce_scores)
        reranked = [stage1_ids[j] for j in sorted_idx[:top_k_rerank]]
        s2_lat.append((time.perf_counter() - t0) * 1_000)
        s2_rel.append(_binary_relevance(reranked, rel_ids, top_k_rerank))
        s2_recall.append(_binary_relevance(stage1_ids, rel_ids, top_k_retrieve))

    K = min(10, top_k_rerank)
    stage1 = BenchmarkResult(
        name="Stage 1 only (bi-encoder)",
        mrr_at_10=mean_reciprocal_rank(s1_rel, k=K),
        ndcg_at_10=ndcg_at_k(s1_rel, k=K),
        recall_at_k=recall_at_k(s1_recall, total_rel, k=top_k_retrieve),
        k_recall=top_k_retrieve,
        median_latency_ms=float(np.median(s1_lat)),
        p95_latency_ms=float(np.percentile(s1_lat, 95)),
        n_queries=n_queries,
    )
    stage2 = BenchmarkResult(
        name="Two-stage (bi-enc + cross-enc)",
        mrr_at_10=mean_reciprocal_rank(s2_rel, k=K),
        ndcg_at_10=ndcg_at_k(s2_rel, k=K),
        recall_at_k=recall_at_k(s2_recall, total_rel, k=top_k_retrieve),
        k_recall=top_k_retrieve,
        median_latency_ms=float(np.median(s2_lat)),
        p95_latency_ms=float(np.percentile(s2_lat, 95)),
        n_queries=n_queries,
    )
    return stage1, stage2


def _print_table(s1: BenchmarkResult, s2: BenchmarkResult) -> None:
    """Print a formatted comparison table to stdout."""
    W = 76
    print()
    print(f"{'Metric':<30} {'Stage 1':>14} {'Two-stage':>14} {'Delta':>8}")
    print("-" * W)

    def row(label: str, v1: float, v2: float, fmt: str = ".4f") -> None:
        d = v2 - v1
        arrow = " (+)" if d > 0.001 else (" (-)" if d < -0.001 else "   ~")
        print(f"{label:<30} {v1:>14{fmt}} {v2:>14{fmt}} {arrow:>8}")

    row("MRR@10", s1.mrr_at_10, s2.mrr_at_10)
    row("NDCG@10", s1.ndcg_at_10, s2.ndcg_at_10)
    row(f"Recall@{s1.k_recall}", s1.recall_at_k, s2.recall_at_k)
    row("Median latency (ms)", s1.median_latency_ms, s2.median_latency_ms, ".2f")
    row("P95 latency (ms)", s1.p95_latency_ms, s2.p95_latency_ms, ".2f")
    print("-" * W)
    print(f"  n_queries={s1.n_queries}")
    print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark single-stage vs two-stage retrieval."
    )
    parser.add_argument("--corpus_size", type=int, default=5_000)
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--top_k_retrieve", type=int, default=100)
    parser.add_argument("--top_k_rerank", type=int, default=10)
    parser.add_argument("--relevant_per_query", type=int, default=3)
    args = parser.parse_args()

    s1, s2 = run_benchmark(
        corpus_size=args.corpus_size,
        n_queries=args.n_queries,
        dim=args.dim,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        relevant_per_query=args.relevant_per_query,
    )
    _print_table(s1, s2)


if __name__ == "__main__":
    main()

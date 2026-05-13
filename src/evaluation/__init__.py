"""Evaluation sub-package for information retrieval metrics.

Exports standard IR metrics implemented in pure NumPy:

* :func:`mean_reciprocal_rank` (MRR@K)
* :func:`ndcg_at_k` (NDCG@K)
* :func:`recall_at_k` (Recall@K)
* :func:`precision_at_k` (Precision@K)

Example
-------
>>> from src.evaluation import mean_reciprocal_rank
>>> mean_reciprocal_rank([[1, 0, 1], [0, 1, 0]], k=3)
0.75
"""

from src.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = ["mean_reciprocal_rank", "ndcg_at_k", "precision_at_k", "recall_at_k"]

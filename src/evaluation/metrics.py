"""Standard information-retrieval evaluation metrics.

All functions operate on *ranked relevance lists* -- Python lists (or NumPy
arrays) where element i is the binary (0/1) or graded (int >= 0) relevance
of the result at rank i+1.

Only NumPy is required as a dependency.

Metric definitions
------------------
MRR@K  : Mean Reciprocal Rank -- mean of (1 / rank_of_first_relevant).
NDCG@K : Normalised Discounted Cumulative Gain -- graded relevance,
         log2 discount: gain / log2(rank + 1).
Recall@K : Fraction of all relevant docs that appear in the top-K results.
Precision@K : Fraction of the top-K results that are relevant.

Example
-------
>>> from src.evaluation.metrics import ndcg_at_k
>>> ndcg_at_k([[3, 2, 3, 0, 1, 2]], k=6)  # doctest: +ELLIPSIS
0.9...
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

RelevanceList = Union[List[int], NDArray[np.float64]]


# ---------------------------------------------------------------------------
# MRR@K
# ---------------------------------------------------------------------------


def mean_reciprocal_rank(
    relevances: Sequence[RelevanceList],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank at cut-off K.

    Parameters
    ----------
    relevances:
        Sequence of per-query relevance lists.  ``r[i]`` = relevance of
        the result at rank i+1.  Binary (0/1) relevance is typical.
    k:
        Rank cut-off.  Positions beyond k are ignored.

    Returns
    -------
    float
        MRR@K averaged over all queries.  Returns 0.0 for empty input.

    Raises
    ------
    ValueError
        If ``k < 1``.

    Examples
    --------
    >>> mean_reciprocal_rank([[0, 1, 0], [1, 0, 0]], k=3)
    0.75
    >>> mean_reciprocal_rank([[0, 0, 0]], k=3)
    0.0
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if not relevances:
        return 0.0

    rr_scores: list[float] = []
    for rel in relevances:
        arr = np.asarray(rel[:k], dtype=float)
        hits = np.where(arr > 0)[0]
        rr_scores.append(1.0 / (hits[0] + 1) if hits.size > 0 else 0.0)

    return float(np.mean(rr_scores))


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------


def _dcg_at_k(relevances: NDArray[np.float64], k: int) -> float:
    """Discounted Cumulative Gain for a single query up to rank k."""
    rel = np.asarray(relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    gains = (2.0 ** rel - 1.0) / np.log2(np.arange(2, rel.size + 2))
    return float(gains.sum())


def ndcg_at_k(
    relevances: Sequence[RelevanceList],
    k: int = 10,
) -> float:
    """Compute Normalised Discounted Cumulative Gain at cut-off K.

    Supports graded relevance (integer >= 0).  iDCG is the ideal DCG
    obtained by sorting relevance descending.  Queries with iDCG = 0
    are excluded from the average.

    Parameters
    ----------
    relevances:
        Sequence of per-query relevance lists.
    k:
        Rank cut-off.

    Returns
    -------
    float
        Mean NDCG@K.  Returns 0.0 for empty input.

    Raises
    ------
    ValueError
        If ``k < 1``.

    Examples
    --------
    >>> ndcg_at_k([[3, 2, 3, 0, 1, 2]], k=6)  # doctest: +ELLIPSIS
    0.9...
    >>> ndcg_at_k([[0, 0, 0]], k=3)
    0.0
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if not relevances:
        return 0.0

    ndcg_scores: list[float] = []
    for rel in relevances:
        arr = np.asarray(rel, dtype=float)
        ideal = np.sort(arr)[::-1]
        idcg = _dcg_at_k(ideal, k)
        if idcg == 0.0:
            continue
        ndcg_scores.append(_dcg_at_k(arr, k) / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


def recall_at_k(
    relevances: Sequence[RelevanceList],
    total_relevant: Sequence[int],
    k: int = 10,
) -> float:
    """Compute Recall at cut-off K.

    Parameters
    ----------
    relevances:
        Sequence of per-query relevance lists.
    total_relevant:
        Number of relevant documents in the corpus for each query.
        Must have the same length as ``relevances``.
    k:
        Rank cut-off.

    Returns
    -------
    float
        Mean Recall@K.  Queries where ``total_relevant[q] == 0`` are skipped.

    Raises
    ------
    ValueError
        If lengths differ or ``k < 1``.

    Examples
    --------
    >>> recall_at_k([[1, 0, 1, 0]], [2], k=4)
    1.0
    >>> recall_at_k([[1, 0, 0, 0]], [3], k=2)  # doctest: +ELLIPSIS
    0.333...
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if len(relevances) != len(total_relevant):
        raise ValueError(
            f"relevances and total_relevant must have the same length; "
            f"got {len(relevances)} and {len(total_relevant)}."
        )
    if not relevances:
        return 0.0

    recall_scores: list[float] = []
    for rel, n_total in zip(relevances, total_relevant):
        if n_total == 0:
            continue
        arr = np.asarray(rel[:k], dtype=float)
        recall_scores.append(float((arr > 0).sum()) / n_total)

    return float(np.mean(recall_scores)) if recall_scores else 0.0


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------


def precision_at_k(
    relevances: Sequence[RelevanceList],
    k: int = 10,
) -> float:
    """Compute Precision at cut-off K.

    Parameters
    ----------
    relevances:
        Sequence of per-query relevance lists.
    k:
        Rank cut-off.

    Returns
    -------
    float
        Mean Precision@K.  Returns 0.0 for empty input.

    Raises
    ------
    ValueError
        If ``k < 1``.

    Examples
    --------
    >>> precision_at_k([[1, 1, 0, 1, 0]], k=5)
    0.6
    >>> precision_at_k([[0, 0, 0]], k=3)
    0.0
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if not relevances:
        return 0.0

    precision_scores: list[float] = []
    for rel in relevances:
        arr = np.asarray(rel[:k], dtype=float)
        precision_scores.append(float((arr > 0).mean()))

    return float(np.mean(precision_scores))

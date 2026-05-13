"""Two-stage retrieve-then-rerank search pipeline.

Stage 1 - Recall (bi-encoder + FAISS)
    The query is encoded with a bi-encoder and used to search a FAISS index
    for the top-``top_k_retrieve`` approximate nearest-neighbour passages.
    This stage is fast but uses a weaker relevance signal.

Stage 2 - Precision (cross-encoder reranking)
    The ``top_k_retrieve`` candidates are scored by a cross-encoder that
    attends jointly over each (query, passage) pair. Results are sorted by
    this score and the top ``top_k_rerank`` are returned.

Example
-------
>>> from src.retrieval.pipeline import SearchPipeline
>>> pipeline = SearchPipeline(bi_encoder, cross_encoder, faiss_builder, doc_store)
>>> results = pipeline.search("What is machine learning?")
>>> results[0].score
0.92...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from src.index.builder import FAISSIndexBuilder
from src.index.document_store import DocumentStore
from src.models.bi_encoder import BiEncoder
from src.models.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass(order=True, frozen=True)
class SearchResult:
    """A single ranked result returned by :class:`SearchPipeline`.

    Ordering is by *score* descending (higher = more relevant), so
    ``sorted(results)`` places the best match first.

    Attributes
    ----------
    score:
        Cross-encoder relevance score in [0, 1].
    text:
        Raw passage text as stored in the :class:`DocumentStore`.
    doc_id:
        Internal FAISS integer ID.
    metadata:
        Arbitrary key/value pairs from the document store.
    bi_encoder_score:
        Inner-product similarity from the FAISS search (Stage 1).
    stage:
        ``"stage1"`` (bi-encoder only) or ``"stage2"`` (reranked).

    Examples
    --------
    >>> r = SearchResult(score=0.95, text="hello", doc_id=0)
    >>> r.score
    0.95
    """

    sort_index: float = field(init=False, repr=False, compare=True)
    score: float = field(compare=False)
    text: str = field(compare=False)
    doc_id: int = field(compare=False, default=-1)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict, hash=False)
    bi_encoder_score: Optional[float] = field(compare=False, default=None)
    stage: str = field(compare=False, default="stage2")

    def __post_init__(self) -> None:
        # Negate so ascending sort == descending score
        object.__setattr__(self, "sort_index", -self.score)


class SearchPipeline:
    """Orchestrates the two-stage dense retrieval pipeline.

    Parameters
    ----------
    bi_encoder:
        Loaded :class:`~src.models.bi_encoder.BiEncoder` instance.
    cross_encoder:
        Loaded :class:`~src.models.cross_encoder.CrossEncoder` instance.
    index_builder:
        Populated :class:`~src.index.builder.FAISSIndexBuilder`.
    doc_store:
        Populated :class:`~src.index.document_store.DocumentStore`.
    top_k_retrieve:
        Number of candidates to fetch from FAISS in Stage 1.
        Must be >= ``top_k_rerank``.
    top_k_rerank:
        Number of results to return after reranking in Stage 2.

    Raises
    ------
    ValueError
        If ``top_k_retrieve < top_k_rerank``.

    Examples
    --------
    >>> pipeline = SearchPipeline(bi_enc, cross_enc, idx, store, 100, 10)
    >>> results = pipeline.search("neural machine translation")
    >>> len(results) <= 10
    True
    """

    def __init__(
        self,
        bi_encoder: BiEncoder,
        cross_encoder: CrossEncoder,
        index_builder: FAISSIndexBuilder,
        doc_store: DocumentStore,
        top_k_retrieve: int = 100,
        top_k_rerank: int = 10,
    ) -> None:
        if top_k_retrieve < top_k_rerank:
            raise ValueError(
                f"top_k_retrieve ({top_k_retrieve}) must be >= top_k_rerank ({top_k_rerank})."
            )
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.index_builder = index_builder
        self.doc_store = doc_store
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
    ) -> List[SearchResult]:
        """Run the full two-stage pipeline for *query*.

        Parameters
        ----------
        query:
            Raw query string.
        top_k_retrieve:
            Override the instance-level recall budget for this call only.
        top_k_rerank:
            Override the instance-level rerank budget for this call only.

        Returns
        -------
        List[SearchResult]
            Results sorted by cross-encoder score descending.

        Examples
        --------
        >>> results = pipeline.search("What is BERT?")
        >>> all(isinstance(r, SearchResult) for r in results)
        True
        """
        k1 = top_k_retrieve or self.top_k_retrieve
        k2 = top_k_rerank or self.top_k_rerank

        candidates = self.stage1_retrieve(query, top_k=k1)
        if not candidates:
            logger.warning("Stage 1 returned 0 candidates for query: %r", query)
            return []

        return self.stage2_rerank(query, candidates, top_k=k2)

    def stage1_retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Encode the query and retrieve nearest neighbours from FAISS.

        Parameters
        ----------
        query:
            Raw query string.
        top_k:
            Number of candidates to retrieve.

        Returns
        -------
        List[SearchResult]
            Candidate results with ``stage="stage1"`` and
            ``bi_encoder_score`` populated.
        """
        import faiss as _faiss

        k = top_k or self.top_k_retrieve
        t0 = time.perf_counter()

        query_vec: NDArray[np.float32] = self.bi_encoder.encode_query(query)
        query_matrix = query_vec[np.newaxis, :]  # shape (1, dim)

        distances, indices = self.index_builder.index.search(query_matrix, k)
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug("Stage 1: %d candidates in %.1f ms", len(indices[0]), elapsed_ms)

        candidates: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            try:
                text, meta = self.doc_store.get(int(idx))
            except KeyError:
                logger.warning("Unknown doc_id=%d from FAISS, skipping.", idx)
                continue

            # Inner-product index: distance IS the similarity.
            # L2 index: convert distance to a pseudo-similarity.
            if (
                hasattr(self.index_builder.index, "metric_type")
                and self.index_builder.index.metric_type == _faiss.METRIC_INNER_PRODUCT
            ):
                bi_score = float(dist)
            else:
                bi_score = float(1.0 / (1.0 + max(float(dist), 0.0)))

            candidates.append(
                SearchResult(
                    score=bi_score,
                    text=text,
                    doc_id=int(idx),
                    metadata=meta,
                    bi_encoder_score=bi_score,
                    stage="stage1",
                )
            )

        return candidates

    def stage2_rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Score candidate (query, passage) pairs with the cross-encoder.

        Parameters
        ----------
        query:
            Original query string.
        candidates:
            Output of :meth:`stage1_retrieve`.
        top_k:
            Number of top results to return after reranking.

        Returns
        -------
        List[SearchResult]
            Results sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        k = top_k or self.top_k_rerank
        t0 = time.perf_counter()

        pairs: List[tuple[str, str]] = [(query, c.text) for c in candidates]
        ce_scores: NDArray[np.float32] = self.cross_encoder.predict(pairs)

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug("Stage 2: reranked %d in %.1f ms", len(candidates), elapsed_ms)

        reranked: List[SearchResult] = [
            SearchResult(
                score=float(ce_score),
                text=cand.text,
                doc_id=cand.doc_id,
                metadata=cand.metadata,
                bi_encoder_score=cand.bi_encoder_score,
                stage="stage2",
            )
            for cand, ce_score in zip(candidates, ce_scores)
        ]
        reranked.sort()  # ascending sort_index == descending score
        return reranked[:k]

    def search_with_timing(
        self,
        query: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
    ) -> tuple[List[SearchResult], Dict[str, float]]:
        """Like :meth:`search` but also returns per-stage latencies in ms.

        Returns
        -------
        tuple[List[SearchResult], Dict[str, float]]
            ``(results, timings)`` where ``timings`` has keys
            ``encode_ms``, ``faiss_ms``, ``rerank_ms``, ``total_ms``.

        Examples
        --------
        >>> results, t = pipeline.search_with_timing("quantum computing")
        >>> "rerank_ms" in t
        True
        """
        import faiss as _faiss

        k1 = top_k_retrieve or self.top_k_retrieve
        k2 = top_k_rerank or self.top_k_rerank

        t_start = time.perf_counter()

        t0 = time.perf_counter()
        query_vec = self.bi_encoder.encode_query(query)
        encode_ms = (time.perf_counter() - t0) * 1_000

        t0 = time.perf_counter()
        query_matrix = query_vec[np.newaxis, :]
        distances, indices = self.index_builder.index.search(query_matrix, k1)
        faiss_ms = (time.perf_counter() - t0) * 1_000

        candidates: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            try:
                text, meta = self.doc_store.get(int(idx))
            except KeyError:
                continue
            if (
                hasattr(self.index_builder.index, "metric_type")
                and self.index_builder.index.metric_type == _faiss.METRIC_INNER_PRODUCT
            ):
                bi_score = float(dist)
            else:
                bi_score = float(1.0 / (1.0 + max(float(dist), 0.0)))
            candidates.append(
                SearchResult(
                    score=bi_score, text=text, doc_id=int(idx),
                    metadata=meta, bi_encoder_score=bi_score, stage="stage1",
                )
            )

        t0 = time.perf_counter()
        if candidates:
            pairs = [(query, c.text) for c in candidates]
            ce_scores = self.cross_encoder.predict(pairs)
            reranked: List[SearchResult] = [
                SearchResult(
                    score=float(s), text=c.text, doc_id=c.doc_id,
                    metadata=c.metadata, bi_encoder_score=c.bi_encoder_score,
                    stage="stage2",
                )
                for c, s in zip(candidates, ce_scores)
            ]
            reranked.sort()
            results = reranked[:k2]
        else:
            results = []
        rerank_ms = (time.perf_counter() - t0) * 1_000
        total_ms = (time.perf_counter() - t_start) * 1_000

        return results, {
            "encode_ms": round(encode_ms, 2),
            "faiss_ms": round(faiss_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "total_ms": round(total_ms, 2),
        }

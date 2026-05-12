"""Cross-encoder wrapper for neural re-ranking.

Wraps a ``sentence_transformers.CrossEncoder`` model.  Unlike the bi-encoder,
the cross-encoder attends *jointly* over the query and passage, giving it a
richer relevance signal at the cost of O(K) inference per query (where K is
the candidate pool size).

The raw model logits are passed through a sigmoid so that every score lies in
[0, 1], making cross-encoder scores directly comparable across queries.

Typical usage
-------------
>>> from src.models.cross_encoder import CrossEncoder
>>> reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
>>> scores = reranker.predict([("what is AI?", "AI is a field of CS.")],
...                           batch_size=32)
>>> 0.0 <= float(scores[0]) <= 1.0
True
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import CrossEncoder as _STCrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoder:
    """Pairwise cross-encoder for (query, passage) relevance scoring.

    Scores are sigmoid-calibrated to [0, 1].  Higher score = more relevant.

    Parameters
    ----------
    model_name_or_path:
        Hugging Face model ID or local path for the cross-encoder.
    max_length:
        Maximum total token length for the (query, passage) pair.  Longer
        inputs are truncated.  ``None`` uses the model's default.
    device:
        PyTorch device string.  ``None`` lets sentence-transformers decide.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int | None = 512,
        device: str | None = None,
    ) -> None:
        logger.info("Loading cross-encoder: %s", model_name_or_path)
        self.model_name = model_name_or_path
        self._model = _STCrossEncoder(
            model_name_or_path,
            max_length=max_length,
            device=device,
        )
        logger.info("Cross-encoder ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]:
        """Score a list of (query, passage) pairs.

        Parameters
        ----------
        pairs:
            List of ``(query, passage)`` tuples to score.
        batch_size:
            Number of pairs per forward pass.
        show_progress_bar:
            Whether to display a tqdm bar.

        Returns
        -------
        NDArray[np.float32]
            1-D float32 array of shape ``(len(pairs),)`` with sigmoid-
            calibrated scores in [0, 1].

        Examples
        --------
        >>> reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> scores = reranker.predict([("query", "passage")])
        >>> scores.dtype
        dtype('float32')
        >>> ((scores >= 0) & (scores <= 1)).all()
        True
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        raw_scores: NDArray[np.float32] = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        # Apply sigmoid so scores are in [0, 1]
        calibrated = (1.0 / (1.0 + np.exp(-raw_scores))).astype(np.float32)
        return calibrated

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> List[Tuple[int, float, str]]:
        """Re-rank a list of passages for a single query.

        Parameters
        ----------
        query:
            The search query.
        passages:
            Candidate passage strings to re-rank.
        top_k:
            Return only the top-K results.  ``None`` returns all passages.
        batch_size:
            Forward-pass batch size.

        Returns
        -------
        list[tuple[int, float, str]]
            Sorted list of ``(original_index, score, passage)`` tuples,
            highest score first.

        Examples
        --------
        >>> reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> results = reranker.rerank("AI", ["Machine learning", "Cooking"], top_k=1)
        >>> len(results)
        1
        >>> results[0][0] in (0, 1)  # original index
        True
        """
        if not passages:
            return []

        pairs = [(query, p) for p in passages]
        scores = self.predict(pairs, batch_size=batch_size)

        ranked = sorted(
            enumerate(zip(scores, passages)),
            key=lambda x: x[1][0],
            reverse=True,
        )
        results = [(idx, float(score), text) for idx, (score, text) in ranked]

        if top_k is not None:
            results = results[:top_k]
        return results

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(model={self.model_name!r})"

"""Bi-encoder wrapper for dense passage retrieval.

Wraps a ``sentence_transformers.SentenceTransformer`` model and exposes
corpus-encoding (with batching + progress bar) and query-encoding helpers.
All embeddings are L2-normalised before being returned so that inner-product
search in FAISS is equivalent to cosine similarity.

Typical usage
-------------
>>> from src.models.bi_encoder import BiEncoder
>>> encoder = BiEncoder("sentence-transformers/all-MiniLM-L6-v2")
>>> q_emb = encoder.encode_query("What is machine learning?")
>>> q_emb.shape
(384,)
>>> corpus_embs = encoder.encode_corpus(["doc one", "doc two"], batch_size=32)
>>> corpus_embs.shape
(2, 384)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BiEncoder:
    """Dense bi-encoder for symmetric or asymmetric retrieval.

    Encodes queries and passages **independently** using the same
    ``SentenceTransformer`` backbone.  Embeddings are L2-normalised so that
    dot-product similarity equals cosine similarity.

    Parameters
    ----------
    model_name_or_path:
        Hugging Face model ID or local path accepted by
        ``SentenceTransformer``.
    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).  When *None*,
        sentence-transformers chooses automatically.
    show_progress_bar:
        Whether to display a tqdm progress bar during corpus encoding.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        show_progress_bar: bool = True,
    ) -> None:
        logger.info("Loading bi-encoder: %s", model_name_or_path)
        self.model_name = model_name_or_path
        self.show_progress_bar = show_progress_bar
        self._model = SentenceTransformer(model_name_or_path, device=device)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Bi-encoder ready (dim=%d)", self.embedding_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Encode a single query string into a normalised dense vector.

        Parameters
        ----------
        query:
            Raw query text.

        Returns
        -------
        NDArray[np.float32]
            1-D float32 array of shape ``(embedding_dim,)``, L2-normalised.

        Examples
        --------
        >>> enc = BiEncoder("sentence-transformers/all-MiniLM-L6-v2")
        >>> v = enc.encode_query("hello world")
        >>> v.shape
        (384,)
        >>> float(np.linalg.norm(v))  # doctest: +ELLIPSIS
        1.0...
        """
        embedding: NDArray[np.float32] = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def encode_corpus(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> NDArray[np.float32]:
        """Encode a list of passages into a normalised embedding matrix.

        Passages are processed in mini-batches to keep GPU memory usage
        manageable.  A tqdm bar is shown if ``show_progress_bar`` is True.

        Parameters
        ----------
        texts:
            List of passage strings to encode.
        batch_size:
            Number of passages per forward pass.

        Returns
        -------
        NDArray[np.float32]
            2-D float32 array of shape ``(len(texts), embedding_dim)``,
            each row L2-normalised.

        Examples
        --------
        >>> enc = BiEncoder("sentence-transformers/all-MiniLM-L6-v2")
        >>> embs = enc.encode_corpus(["doc a", "doc b"])
        >>> embs.shape
        (2, 384)
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        logger.info("Encoding %d passages (batch_size=%d)", len(texts), batch_size)

        all_embeddings: list[NDArray[np.float32]] = []
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        for batch in tqdm(
            batches,
            desc="Encoding corpus",
            disable=not self.show_progress_bar,
            unit="batch",
        ):
            batch_emb: NDArray[np.float32] = self._model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_embeddings.append(batch_emb.astype(np.float32))

        matrix = np.vstack(all_embeddings)
        logger.info("Corpus encoded — shape %s", matrix.shape)
        return matrix

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, dim={self.embedding_dim})"
        )

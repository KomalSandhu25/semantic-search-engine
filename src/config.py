"""Application configuration loaded from environment variables.

All settings are declared once as a typed ``Settings`` class backed by
pydantic-settings.  Values are read from the process environment and,
optionally, a ``.env`` file at project root.

Usage
-----
>>> from src.config import settings
>>> settings.bi_encoder_model
'sentence-transformers/all-MiniLM-L6-v2'
>>> settings.top_k_retrieve
100
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated, typed application settings.

    Attributes
    ----------
    bi_encoder_model:
        Hugging Face model ID used for the bi-encoder (recall stage).
        The model encodes queries and passages into a shared dense vector
        space.  Passage vectors are pre-computed and stored in the FAISS
        index; at query time only the query is encoded.
    cross_encoder_model:
        Hugging Face model ID used for the cross-encoder (re-ranking stage).
        Takes a ``(query, passage)`` pair and returns a scalar relevance
        score.  Slower than the bi-encoder but significantly more precise.
    faiss_index_path:
        Filesystem path where the FAISS index is stored.  The parent
        directory is created automatically when the index is first built.
    top_k_retrieve:
        Number of candidate passages retrieved from the FAISS index during
        the recall stage.  Must be larger than ``top_k_rerank``.
    top_k_rerank:
        Number of passages returned to the caller after the cross-encoder
        re-ranks the ``top_k_retrieve`` candidates.

    Examples
    --------
    >>> s = Settings()
    >>> s.top_k_retrieve > s.top_k_rerank
    True
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    faiss_index_path: Path = Path("data/indices/corpus.index")
    top_k_retrieve: int = 100
    top_k_rerank: int = 10

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def faiss_index_dir(self) -> Path:
        """Parent directory of :attr:`faiss_index_path`.

        Returns
        -------
        Path
            Directory that must exist before the index can be written to
            disk.

        Examples
        --------
        >>> s = Settings(faiss_index_path="data/indices/corpus.index")
        >>> s.faiss_index_dir
        PosixPath('data/indices')
        """
        return self.faiss_index_path.parent


# Module-level singleton — callers should import this rather than
# instantiating Settings() themselves, so the .env file is read once.
settings = Settings()

"""Application-wide configuration loaded from environment variables.

Uses pydantic-settings so every value can be overridden via a .env file
or real environment variables without touching source code.

Typical usage
-------------
>>> from src.config import settings
>>> print(settings.bi_encoder_model)
'sentence-transformers/all-MiniLM-L6-v2'
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the semantic search engine.

    All fields map 1-to-1 to environment variables (case-insensitive).
    Defaults represent sensible values for local development.

    Attributes
    ----------
    bi_encoder_model:
        Hugging Face model ID (or local path) for the bi-encoder used in
        the first retrieval stage.  Bi-encoders encode queries and documents
        independently, enabling sub-millisecond ANN lookup via FAISS.
    cross_encoder_model:
        Hugging Face model ID for the cross-encoder re-ranker.  Cross-
        encoders jointly encode (query, document) pairs for higher accuracy
        at the cost of more compute -- therefore applied only to the top-K
        bi-encoder candidates.
    faiss_index_path:
        Filesystem path where the serialised FAISS index lives.  The
        directory is created automatically on first build.
    top_k_retrieve:
        Number of nearest-neighbour candidates to fetch from FAISS during
        the recall stage.  Higher values improve recall at the cost of more
        cross-encoder calls.
    top_k_rerank:
        Final number of results returned to the caller after cross-encoder
        re-ranking.  Must be <= top_k_retrieve.

    Example
    -------
    >>> from src.config import settings
    >>> assert settings.top_k_rerank <= settings.top_k_retrieve
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    faiss_index_path: Path = Path("data/indices/corpus.index")
    top_k_retrieve: int = 100
    top_k_rerank: int = 10

    def model_post_init(self, __context: object) -> None:
        """Validate cross-field constraints after individual field parsing."""
        if self.top_k_rerank > self.top_k_retrieve:
            raise ValueError(
                f"top_k_rerank ({self.top_k_rerank}) must be <= "
                f"top_k_retrieve ({self.top_k_retrieve})"
            )
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)


#: Module-level singleton -- import this everywhere instead of instantiating.
settings = Settings()

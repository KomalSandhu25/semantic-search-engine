"""Application-wide configuration loaded from environment variables.

Uses pydantic-settings so every value is validated and type-checked at
startup.  All settings have sensible defaults so the service can run
out-of-the-box with no .env file; the defaults are identical to those
documented in .env.example.

Example
-------
>>> from src.config import settings
>>> print(settings.top_k_retrieve)
100
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings sourced from environment / .env file.

    Attributes
    ----------
    bi_encoder_model:
        Hugging Face model identifier for the sentence-transformer used in
        the first-stage dense retrieval step.
    cross_encoder_model:
        Hugging Face model identifier for the cross-encoder used in the
        second-stage reranking step.
    faiss_index_path:
        Filesystem path for the persisted FAISS flat-L2 index.
    top_k_retrieve:
        Number of candidate passages fetched from the FAISS index before
        reranking.  Governs recall ceiling.
    top_k_rerank:
        Number of results returned to the caller after the cross-encoder
        reranks the ``top_k_retrieve`` candidates.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    bi_encoder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face model name for bi-encoder (retrieval stage).",
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Hugging Face model name for cross-encoder (reranking stage).",
    )
    faiss_index_path: Path = Field(
        default=Path("data/index/faiss.index"),
        description="Path to the persisted FAISS index file.",
    )
    top_k_retrieve: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Candidate count fetched from FAISS before reranking.",
    )
    top_k_rerank: int = Field(
        default=10,
        ge=1,
        description="Final result count returned after cross-encoder reranking.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("top_k_rerank", mode="after")
    @classmethod
    def rerank_lte_retrieve(cls, v: int, info) -> int:  # noqa: ANN001
        """Ensure ``top_k_rerank`` does not exceed ``top_k_retrieve``.

        Args:
            v: The proposed value for ``top_k_rerank``.
            info: Pydantic validation context carrying already-validated fields.

        Returns:
            The validated value for ``top_k_rerank``.

        Raises:
            ValueError: If ``top_k_rerank`` > ``top_k_retrieve``.
        """
        retrieve = info.data.get("top_k_retrieve", 100)
        if v > retrieve:
            raise ValueError(
                f"top_k_rerank ({v}) must be <= top_k_retrieve ({retrieve})."
            )
        return v


# Module-level singleton — import this throughout the codebase.
settings: Settings = Settings()

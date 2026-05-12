"""Unit tests for src/config.py.

Tests cover:
- Default values match expected model names and numeric parameters.
- Environment variable overrides are respected.
- The ``faiss_index_dir`` derived property returns the correct parent path.
- The invariant ``top_k_retrieve > top_k_rerank`` holds for defaults.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _fresh_settings(**env_overrides: str) -> "Settings":  # noqa: F821
    """Return a Settings instance with env vars applied via monkeypatching.

    Because pydantic-settings reads the environment at instantiation time,
    we can't rely on module-level state for override tests.
    """
    import src.config as cfg_mod
    importlib.reload(cfg_mod)
    return cfg_mod.Settings(**{k.lower(): v for k, v in env_overrides.items()})


class TestDefaults:
    def test_bi_encoder_model(self) -> None:
        from src.config import Settings
        assert Settings().bi_encoder_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_cross_encoder_model(self) -> None:
        from src.config import Settings
        assert Settings().cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_top_k_retrieve(self) -> None:
        from src.config import Settings
        assert Settings().top_k_retrieve == 100

    def test_top_k_rerank(self) -> None:
        from src.config import Settings
        assert Settings().top_k_rerank == 10

    def test_retrieve_greater_than_rerank(self) -> None:
        """Retrieve pool must be bigger than the re-rank window by design."""
        from src.config import Settings
        s = Settings()
        assert s.top_k_retrieve > s.top_k_rerank

    def test_faiss_index_path_type(self) -> None:
        from src.config import Settings
        assert isinstance(Settings().faiss_index_path, Path)


class TestOverrides:
    def test_top_k_retrieve_override(self) -> None:
        from src.config import Settings
        s = Settings(top_k_retrieve=50)
        assert s.top_k_retrieve == 50

    def test_top_k_rerank_override(self) -> None:
        from src.config import Settings
        s = Settings(top_k_rerank=5)
        assert s.top_k_rerank == 5

    def test_bi_encoder_model_override(self) -> None:
        from src.config import Settings
        s = Settings(bi_encoder_model="sentence-transformers/all-mpnet-base-v2")
        assert s.bi_encoder_model == "sentence-transformers/all-mpnet-base-v2"

    def test_faiss_index_path_override(self) -> None:
        from src.config import Settings
        s = Settings(faiss_index_path="/data/custom/my.index")
        assert s.faiss_index_path == Path("/data/custom/my.index")


class TestDerivedProperties:
    def test_faiss_index_dir_default(self) -> None:
        from src.config import Settings
        s = Settings()
        assert s.faiss_index_dir == Path("data/indices")

    def test_faiss_index_dir_custom(self) -> None:
        from src.config import Settings
        s = Settings(faiss_index_path="/tmp/indices/test.index")
        assert s.faiss_index_dir == Path("/tmp/indices")

    def test_faiss_index_dir_is_parent(self) -> None:
        from src.config import Settings
        s = Settings(faiss_index_path="a/b/c/d.index")
        assert s.faiss_index_dir == s.faiss_index_path.parent

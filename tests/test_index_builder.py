"""Tests for FAISSIndexBuilder."""

from __future__ import annotations

import numpy as np
import pytest

from src.index.builder import FAISSIndexBuilder, _validate_embeddings

DIM = 64
N_SMALL = 100
N_MEDIUM = 200


def _random_embeddings(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim)).astype(np.float32)


class TestValidateEmbeddings:
    def test_2d_passthrough(self):
        arr = _random_embeddings(10)
        out = _validate_embeddings(arr, DIM)
        assert out.shape == (10, DIM)
        assert out.dtype == np.float32

    def test_1d_promoted_to_2d(self):
        arr = _random_embeddings(1)[0]
        out = _validate_embeddings(arr, DIM)
        assert out.shape == (1, DIM)

    def test_wrong_dim_raises(self):
        arr = _random_embeddings(5, dim=32)
        with pytest.raises(ValueError, match="dim=64"):
            _validate_embeddings(arr, DIM)

    def test_wrong_ndim_raises(self):
        arr = np.zeros((3, 4, DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            _validate_embeddings(arr, DIM)


class TestFAISSIndexBuilderFlat:
    def setup_method(self):
        self.emb = _random_embeddings(N_SMALL)
        self.builder = FAISSIndexBuilder(dim=DIM, corpus_size=N_SMALL)

    def test_auto_selects_flat(self):
        self.builder.build(self.emb)
        assert self.builder.index_type == "Flat"

    def test_ntotal_after_add(self):
        self.builder.build(self.emb).add_documents(self.emb)
        assert self.builder.ntotal == N_SMALL

    def test_search_returns_correct_shape(self):
        self.builder.build(self.emb).add_documents(self.emb)
        q = _random_embeddings(5)
        dists, ids = self.builder.search(q, k=3)
        assert dists.shape == (5, 3)
        assert ids.shape == (5, 3)

    def test_top1_self_retrieval(self):
        self.builder.build(self.emb).add_documents(self.emb)
        _, ids = self.builder.search(self.emb, k=1)
        expected = np.arange(N_SMALL).reshape(-1, 1)
        assert np.array_equal(ids, expected)


class TestFAISSIndexBuilderIVFFlat:
    def setup_method(self):
        self.emb = _random_embeddings(N_MEDIUM)
        self.builder = FAISSIndexBuilder(dim=DIM, corpus_size=N_MEDIUM, nlist=8)

    def test_explicit_ivfflat(self):
        self.builder.build(self.emb, index_type="IVFFlat")
        assert self.builder.index_type == "IVFFlat"

    def test_ntotal_after_batched_add(self):
        self.builder.build(self.emb, index_type="IVFFlat").add_documents(self.emb, batch_size=50)
        assert self.builder.ntotal == N_MEDIUM


class TestFAISSIndexBuilderIVFPQ:
    def setup_method(self):
        self.emb = _random_embeddings(N_MEDIUM)
        self.builder = FAISSIndexBuilder(dim=DIM, corpus_size=N_MEDIUM, nlist=8, m_subvectors=8)

    def test_explicit_ivfpq(self):
        self.builder.build(self.emb, index_type="IVFPQ")
        assert self.builder.index_type == "IVFPQ"

    def test_ntotal(self):
        self.builder.build(self.emb, index_type="IVFPQ").add_documents(self.emb)
        assert self.builder.ntotal == N_MEDIUM


class TestFAISSIndexBuilderSaveLoad:
    def test_round_trip(self, tmp_path):
        emb = _random_embeddings(N_SMALL)
        builder = FAISSIndexBuilder(dim=DIM)
        builder.build(emb).add_documents(emb)
        path = tmp_path / "test.faiss"
        builder.save(path)
        loaded = FAISSIndexBuilder.load(path, dim=DIM)
        assert loaded.ntotal == N_SMALL
        assert loaded.index_type == "Flat"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FAISSIndexBuilder.load(tmp_path / "nonexistent.faiss", dim=DIM)

    def test_save_before_build_raises(self, tmp_path):
        builder = FAISSIndexBuilder(dim=DIM)
        with pytest.raises(RuntimeError, match="Nothing to save"):
            builder.save(tmp_path / "x.faiss")

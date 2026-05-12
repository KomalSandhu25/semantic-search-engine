"""Tests for DocumentStore."""

from __future__ import annotations

import pytest
from src.index.document_store import DocumentStore


def _make_store(path) -> DocumentStore:
    store = DocumentStore(path / "store.ndjson")
    store.add(0, "Hello world", {"lang": "en"})
    store.add(1, "Bonjour monde", {"lang": "fr"})
    return store


class TestDocumentStoreAdd:
    def test_add_and_get(self, tmp_path):
        store = _make_store(tmp_path)
        text, meta = store.get(0)
        assert text == "Hello world"
        assert meta["lang"] == "en"

    def test_duplicate_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="already registered"):
            store.add(0, "duplicate", {})

    def test_len(self, tmp_path):
        assert len(_make_store(tmp_path)) == 2

    def test_contains(self, tmp_path):
        store = _make_store(tmp_path)
        assert 0 in store
        assert 99 not in store

    def test_get_missing_raises(self, tmp_path):
        with pytest.raises(KeyError):
            _make_store(tmp_path).get(999)


class TestDocumentStoreBatch:
    def test_add_batch(self, tmp_path):
        store = DocumentStore(tmp_path / "store.ndjson")
        store.add_batch([10, 11, 12], ["a", "b", "c"])
        assert len(store) == 3

    def test_add_batch_length_mismatch(self, tmp_path):
        store = DocumentStore(tmp_path / "store.ndjson")
        with pytest.raises(ValueError, match="same length"):
            store.add_batch([0, 1], ["only one"])


class TestDocumentStoreGetTexts:
    def test_known_ids(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_texts([0, 1]) == ["Hello world", "Bonjour monde"]

    def test_negative_one_placeholder(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_texts([0, -1])[1] == "<no result>"


class TestDocumentStorePersistence:
    def test_save_load_round_trip(self, tmp_path):
        store = _make_store(tmp_path)
        store.save()
        loaded = DocumentStore.load(tmp_path / "store.ndjson")
        assert len(loaded) == 2
        text, meta = loaded.get(1)
        assert text == "Bonjour monde"
        assert meta["lang"] == "fr"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DocumentStore.load(tmp_path / "nonexistent.ndjson")

    def test_stats(self, tmp_path):
        s = _make_store(tmp_path).stats()
        assert s["size"] == 2
        assert "avg_text_len" in s

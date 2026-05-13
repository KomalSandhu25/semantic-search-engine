"""Unit tests for QueryProcessor."""

from __future__ import annotations

import pytest

from src.retrieval.query_processor import QueryProcessor


class TestCleaning:
    def test_lowercases(self):
        qp = QueryProcessor(expand_synonyms=False)
        assert qp.process("BERT Model") == "bert model"

    def test_strips_whitespace(self):
        qp = QueryProcessor(expand_synonyms=False)
        assert qp.process("  hello world  ") == "hello world"

    def test_collapses_whitespace(self):
        qp = QueryProcessor(expand_synonyms=False)
        assert qp.process("hello   world") == "hello world"

    def test_unicode_normalisation(self):
        qp = QueryProcessor(expand_synonyms=False)
        result = qp.process("ＡI model")
        assert "Ａ" not in result


class TestExpansion:
    def test_expands_known_abbreviation(self):
        qp = QueryProcessor(expand_synonyms=True)
        assert "natural language processing" in qp.process("nlp")

    def test_no_duplicate_expansions(self):
        qp = QueryProcessor(expand_synonyms=True)
        result = qp.process("machine learning ml")
        assert result.count("machine learning") == 1

    def test_disabled_expansion(self):
        qp = QueryProcessor(expand_synonyms=False)
        assert qp.process("nlp") == "nlp"

    def test_add_synonyms_runtime(self):
        qp = QueryProcessor(expand_synonyms=True)
        qp.add_synonyms({"foo": ["bar", "baz"]})
        result = qp.process("foo query")
        assert "bar" in result and "baz" in result

    def test_custom_synonym_dict(self):
        qp = QueryProcessor(expand_synonyms=True, synonym_dict={"rl": ["reinforcement learning"]})
        assert "reinforcement learning" in qp.process("rl agent")


class TestTruncation:
    def test_truncates_long_query(self):
        qp = QueryProcessor(expand_synonyms=False, max_query_length=10)
        assert len(qp.process("a" * 20)) == 10

    def test_no_truncation_when_disabled(self):
        qp = QueryProcessor(expand_synonyms=False, max_query_length=None)
        assert len(qp.process("a " * 1000)) > 100


class TestSpellcheck:
    def test_disabled_spellcheck_works_without_dep(self):
        qp = QueryProcessor(spellcheck=False)
        assert qp.process("hello") == "hello"

    def test_enabled_spellcheck_skips_gracefully_if_missing(self, monkeypatch):
        import builtins
        real = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spellchecker":
                raise ImportError
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        qp = QueryProcessor(spellcheck=True)
        assert qp.process("hello") == "hello"

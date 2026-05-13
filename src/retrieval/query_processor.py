"""Query pre-processing utilities for the semantic search pipeline.

Applies a sequence of transformations to a raw query string before it is
handed to the bi-encoder:

1. Cleaning -- strip excess whitespace, normalise unicode, lower-case.
2. Spellcheck (placeholder) -- hook for an external spellchecker.
   Disabled by default; requires optional ``pyspellchecker`` package.
3. Expansion -- append synonyms from a static dictionary to broaden recall.

Example
-------
>>> from src.retrieval.query_processor import QueryProcessor
>>> qp = QueryProcessor(expand_synonyms=True)
>>> qp.process("ml algorithms")
'ml algorithms machine learning'
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "ml": ["machine learning"],
    "ai": ["artificial intelligence"],
    "nlp": ["natural language processing"],
    "cv": ["computer vision"],
    "dl": ["deep learning"],
    "nn": ["neural network"],
    "bert": ["bidirectional encoder representations"],
    "gpt": ["generative pre-trained transformer"],
    "rag": ["retrieval augmented generation"],
    "ir": ["information retrieval"],
    "knn": ["k nearest neighbours", "nearest neighbor search"],
    "ann": ["approximate nearest neighbour"],
    "sem": ["semantic"],
    "doc": ["document"],
    "qa": ["question answering"],
}


class QueryProcessor:
    """Pre-process a raw query before embedding.

    Parameters
    ----------
    expand_synonyms:
        When True, known abbreviations are expanded via ``synonym_dict``.
    synonym_dict:
        Custom synonym mapping ``{token: [expansion, ...]}``.  Merged with
        (and overrides) the built-in ``_DEFAULT_SYNONYMS``.
    spellcheck:
        When True, a spellcheck pass is attempted via ``pyspellchecker``.
        If the package is not installed, the step is silently skipped.
    max_query_length:
        Queries longer than this many characters are truncated.
        ``None`` disables the limit.

    Examples
    --------
    >>> qp = QueryProcessor(expand_synonyms=True)
    >>> qp.process("  NLP  models  ")
    'nlp models natural language processing'
    """

    def __init__(
        self,
        expand_synonyms: bool = True,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        spellcheck: bool = False,
        max_query_length: Optional[int] = 512,
    ) -> None:
        self.expand_synonyms = expand_synonyms
        self.spellcheck = spellcheck
        self.max_query_length = max_query_length

        self._synonyms: Dict[str, List[str]] = dict(_DEFAULT_SYNONYMS)
        if synonym_dict:
            self._synonyms.update(synonym_dict)

        self._spellchecker = None
        if spellcheck:
            self._spellchecker = self._load_spellchecker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, query: str) -> str:
        """Apply the full pre-processing chain to *query*.

        Steps: clean -> spellcheck -> expand.

        Parameters
        ----------
        query:
            Raw user query string.

        Returns
        -------
        str
            Processed query ready for the bi-encoder.

        Examples
        --------
        >>> qp = QueryProcessor(expand_synonyms=False)
        >>> qp.process("  Hello   WORLD  ")
        'hello world'
        """
        query = self._clean(query)
        if self.spellcheck and self._spellchecker is not None:
            query = self._apply_spellcheck(query)
        if self.expand_synonyms:
            query = self._expand(query)
        if self.max_query_length and len(query) > self.max_query_length:
            logger.warning(
                "Query truncated from %d to %d characters.",
                len(query),
                self.max_query_length,
            )
            query = query[: self.max_query_length]
        return query

    def add_synonyms(self, mapping: Dict[str, List[str]]) -> None:
        """Merge additional synonym entries at runtime.

        Parameters
        ----------
        mapping:
            Dict of ``{token: [expansion, ...]}``.

        Examples
        --------
        >>> qp = QueryProcessor()
        >>> qp.add_synonyms({"tf": ["term frequency", "tensorflow"]})
        >>> qp.process("tf model")
        'tf model term frequency tensorflow'
        """
        self._synonyms.update(mapping)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(query: str) -> str:
        """Lowercase, unicode-normalise, and collapse whitespace."""
        query = unicodedata.normalize("NFKC", query)
        query = query.lower()
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def _expand(self, query: str) -> str:
        """Append synonym expansions for matched tokens in *query*."""
        tokens = query.split()
        expansions: List[str] = []
        seen: set[str] = set(tokens)

        for token in tokens:
            for expansion in self._synonyms.get(token, []):
                if expansion not in seen:
                    expansions.append(expansion)
                    seen.add(expansion)

        if not expansions:
            return query
        return " ".join(tokens + expansions)

    @staticmethod
    def _load_spellchecker():
        """Attempt to import pyspellchecker; return None if absent."""
        try:
            from spellchecker import SpellChecker  # type: ignore
            logger.info("SpellChecker backend loaded.")
            return SpellChecker()
        except ImportError:
            logger.warning(
                "pyspellchecker not installed; spellcheck disabled. "
                "Install with: pip install pyspellchecker"
            )
            return None

    def _apply_spellcheck(self, query: str) -> str:
        """Correct likely misspellings using the loaded spellchecker."""
        if self._spellchecker is None:
            return query

        tokens = query.split()
        corrected: List[str] = []
        for token in tokens:
            correction = self._spellchecker.correction(token)
            if correction and correction != token:
                logger.debug("Spellcheck: %r -> %r", token, correction)
                corrected.append(correction)
            else:
                corrected.append(token)
        return " ".join(corrected)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"expand_synonyms={self.expand_synonyms}, "
            f"spellcheck={self.spellcheck})"
        )

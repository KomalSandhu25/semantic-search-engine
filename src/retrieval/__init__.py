"""Retrieval sub-package for the semantic search engine.

Exposes the two main public symbols used by callers:

* :class:`SearchPipeline` -- orchestrates the two-stage retrieve-then-rerank pipeline.
* :class:`SearchResult` -- lightweight dataclass returned per result.
* :class:`QueryProcessor` -- optional pre-processing applied to raw queries.

Example
-------
>>> from src.retrieval import SearchPipeline, SearchResult
"""

from src.retrieval.pipeline import SearchPipeline, SearchResult
from src.retrieval.query_processor import QueryProcessor

__all__ = ["SearchPipeline", "SearchResult", "QueryProcessor"]

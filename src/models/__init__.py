"""Model sub-package for the semantic search engine.

This package will expose:

- ``BiEncoder``  — wraps a ``sentence-transformers`` model to produce dense
  embeddings for both documents and queries (see ``bi_encoder.py``).
- ``CrossEncoder`` — wraps a cross-encoder model to score (query, passage)
  pairs for precision reranking (see ``cross_encoder.py``).

Both classes will be registered here on subsequent days so the rest of the
codebase can import them via::

    from src.models import BiEncoder, CrossEncoder
"""

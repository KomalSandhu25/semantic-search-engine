"""Model wrappers sub-package.

Exposes encoder abstractions used throughout the search pipeline:

- :class:`~src.models.bi_encoder.BiEncoder`
    Wraps a ``SentenceTransformer`` for batch-encoding passages and
    queries into dense vectors.  Added in Day 2.

- :class:`~src.models.cross_encoder.CrossEncoder`
    Wraps a ``sentence_transformers.CrossEncoder`` for pairwise
    (query, passage) relevance scoring.  Added in Day 2.

Importing this package before Day 2 is safe; the sub-modules are
simply not yet present.
"""

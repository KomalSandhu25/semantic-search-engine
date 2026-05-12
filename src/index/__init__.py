"""
index
=====
Vector indexing layer for the semantic-search-engine.

Exposes the FAISSIndexBuilder for creating / loading FAISS indices and the
DocumentStore for mapping FAISS integer IDs back to raw text and metadata.
"""

from .builder import FAISSIndexBuilder
from .document_store import DocumentStore

__all__ = ["FAISSIndexBuilder", "DocumentStore"]

"""
document_store.py
=================
Maps FAISS integer IDs back to raw document text and arbitrary metadata.

The store is backed by a newline-delimited JSON (NDJSON) file so that very
large corpora can be streamed without loading the whole store into RAM.

Example
-------
>>> store = DocumentStore("data/doc_store.ndjson")
>>> store.add(0, "The quick brown fox", {"source": "wiki"})
>>> store.save()
>>> store2 = DocumentStore.load("data/doc_store.ndjson")
>>> text, meta = store2.get(0)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class DocumentStore:
    """Bidirectional mapping between FAISS integer IDs and document text.

    Args:
        path: Path to the NDJSON file (need not exist yet).
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self._store: dict[int, tuple[str, dict[str, Any]]] = {}

    def add(
        self,
        doc_id: int,
        text: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a single document.

        Args:
            doc_id: The FAISS integer ID assigned to this document.
            text: Raw document text.
            meta: Optional dict of arbitrary metadata.

        Raises:
            ValueError: If *doc_id* is already present.
        """
        if doc_id in self._store:
            raise ValueError(f"doc_id {doc_id} is already registered.")
        self._store[doc_id] = (text, meta or {})

    def add_batch(
        self,
        doc_ids: list[int],
        texts: list[str],
        metas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Register multiple documents at once.

        Args:
            doc_ids: List of FAISS IDs.
            texts: Corresponding document texts.
            metas: Optional list of metadata dicts.

        Raises:
            ValueError: If lengths of *doc_ids* and *texts* differ.
        """
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length.")
        metas = metas or [{} for _ in doc_ids]
        for did, text, meta in zip(doc_ids, texts, metas):
            self.add(did, text, meta)

    def get(self, doc_id: int) -> tuple[str, dict[str, Any]]:
        """Retrieve a document by its FAISS ID.

        Args:
            doc_id: The FAISS integer ID.

        Returns:
            Tuple of ``(text, meta)``.

        Raises:
            KeyError: If *doc_id* is not found.
        """
        if doc_id not in self._store:
            raise KeyError(f"doc_id {doc_id} not found in store.")
        return self._store[doc_id]

    def get_texts(self, doc_ids: list[int]) -> list[str]:
        """Return just the text strings for a list of IDs.

        Args:
            doc_ids: List of FAISS integer IDs (may contain -1 for padding).

        Returns:
            List of text strings; IDs of -1 map to ``"<no result>"``.
        """
        return [
            self._store[did][0] if did != -1 and did in self._store else "<no result>"
            for did in doc_ids
        ]

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, doc_id: int) -> bool:
        return doc_id in self._store

    def __iter__(self) -> Iterator[tuple[int, str, dict[str, Any]]]:
        """Iterate as ``(doc_id, text, meta)`` tuples."""
        for did, (text, meta) in self._store.items():
            yield did, text, meta

    def save(self) -> None:
        """Write the entire store to :attr:`path` in NDJSON format."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            for did, (text, meta) in self._store.items():
                fh.write(json.dumps({"id": did, "text": text, "meta": meta}, ensure_ascii=False))
                fh.write("\n")
        logger.info("DocumentStore saved -> %s (%d docs)", self.path, len(self._store))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "DocumentStore":
        """Load a document store from an NDJSON file.

        Args:
            path: Path to the NDJSON file produced by :meth:`save`.

        Returns:
            A populated :class:`DocumentStore` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Store file not found: {path}")

        store = cls(path)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    store._store[int(obj["id"])] = (obj["text"], obj.get("meta", {}))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping malformed line %d: %s", lineno, exc)

        logger.info("DocumentStore loaded <- %s (%d docs)", path, len(store))
        return store

    def stats(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging."""
        texts = [t for t, _ in self._store.values()]
        avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
        return {
            "path": str(self.path),
            "size": len(self._store),
            "avg_text_len": round(avg_len, 1),
        }

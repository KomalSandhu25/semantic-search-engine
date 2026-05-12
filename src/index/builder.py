"""
builder.py
==========
FAISS index construction with automatic strategy selection.

Three index types are supported:

* **Flat**   – exact brute-force search; best for < 50 k vectors.
* **IVFFlat** – approximate nearest-neighbour via inverted file lists;
  best for 50 k – 2 M vectors.
* **IVFPQ**  – IVF combined with Product Quantisation (PQ) for large
  corpora (> 2 M vectors) where memory is a concern.

The index builder is deliberately stateless: it receives embeddings as
NumPy arrays and returns / persists FAISS index objects.

Example
-------
>>> builder = FAISSIndexBuilder(dim=768)
>>> builder.add_documents(embeddings, batch_size=1024)
>>> builder.save("my_index.faiss")
>>> loaded = FAISSIndexBuilder.load("my_index.faiss", dim=768)
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Corpus-size thresholds (number of vectors)
_FLAT_THRESHOLD: int = 50_000
_IVF_THRESHOLD: int = 2_000_000


@dataclass
class IndexStats:
    """Snapshot of index metrics logged after each build stage."""

    index_type: str
    dim: int
    ntotal: int
    is_trained: bool
    nlist: Optional[int] = None
    m_subvectors: Optional[int] = None
    nbits: Optional[int] = None
    build_time_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def log(self) -> None:
        """Emit all stats at INFO level."""
        logger.info(
            "IndexStats | type=%s dim=%d ntotal=%d trained=%s "
            "nlist=%s m=%s nbits=%s build_time=%.2fs %s",
            self.index_type,
            self.dim,
            self.ntotal,
            self.is_trained,
            self.nlist,
            self.m_subvectors,
            self.nbits,
            self.build_time_seconds,
            self.extra,
        )


class FAISSIndexBuilder:
    """Build, extend, persist and reload FAISS vector indices.

    The class auto-selects the index type based on *corpus_size* if provided
    at construction time; otherwise it defers the decision to the first call
    of :meth:`build`.

    Args:
        dim: Dimensionality of the embedding vectors.
        corpus_size: Expected total number of vectors (used for auto-selection).
            Pass ``None`` to decide at build time.
        nlist: Number of Voronoi cells for IVF indices.  Defaults to
            ``max(4 * sqrt(corpus_size), 16)``.
        m_subvectors: Number of sub-quantiser groups for PQ.  Must divide *dim*
            evenly.  Defaults to ``dim // 8`` (clamped to >= 1).
        nbits: Bits per sub-vector code for PQ.  Defaults to 8.
        use_gpu: If ``True`` and a CUDA device is available, move the index to
            GPU after training.
    """

    def __init__(
        self,
        dim: int,
        corpus_size: Optional[int] = None,
        nlist: Optional[int] = None,
        m_subvectors: Optional[int] = None,
        nbits: int = 8,
        use_gpu: bool = False,
    ) -> None:
        self.dim = dim
        self.corpus_size = corpus_size
        self.nlist = nlist
        self.m_subvectors = m_subvectors
        self.nbits = nbits
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        self._index: Optional[faiss.Index] = None
        self._index_type: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Index construction                                                   #
    # ------------------------------------------------------------------ #

    def _resolve_index_type(self, n_vectors: int) -> str:
        if n_vectors < _FLAT_THRESHOLD:
            return "Flat"
        if n_vectors < _IVF_THRESHOLD:
            return "IVFFlat"
        return "IVFPQ"

    def _default_nlist(self, n_vectors: int) -> int:
        return max(int(4 * math.sqrt(n_vectors)), 16)

    def _default_m(self) -> int:
        return max(self.dim // 8, 1)

    def _make_index(self, index_type: str) -> faiss.Index:
        """Instantiate a FAISS index of the requested type.

        Args:
            index_type: One of ``"Flat"``, ``"IVFFlat"``, ``"IVFPQ"``.

        Returns:
            An untrained (or fully trained for Flat) FAISS index.

        Raises:
            ValueError: If *index_type* is not recognised.
        """
        quantiser = faiss.IndexFlatL2(self.dim)
        if index_type == "Flat":
            idx = faiss.IndexFlatL2(self.dim)
        elif index_type == "IVFFlat":
            nlist = self.nlist or self._default_nlist(self.corpus_size or _FLAT_THRESHOLD * 2)
            idx = faiss.IndexIVFFlat(quantiser, self.dim, nlist, faiss.METRIC_L2)
            self.nlist = nlist
        elif index_type == "IVFPQ":
            nlist = self.nlist or self._default_nlist(self.corpus_size or _IVF_THRESHOLD * 2)
            m = self.m_subvectors or self._default_m()
            idx = faiss.IndexIVFPQ(quantiser, self.dim, nlist, m, self.nbits)
            self.nlist = nlist
            self.m_subvectors = m
        else:
            raise ValueError(f"Unknown index_type: {index_type!r}")
        return idx

    def build(
        self,
        embeddings: np.ndarray,
        index_type: Optional[str] = None,
    ) -> "FAISSIndexBuilder":
        """Train the index on *embeddings* (required for IVF variants).

        Args:
            embeddings: Float32 array of shape ``(n, dim)``.
            index_type: Override auto-selection.  One of ``"Flat"``,
                ``"IVFFlat"``, ``"IVFPQ"``, or ``None`` to auto-select.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *embeddings* has wrong dtype or shape.
        """
        embeddings = _validate_embeddings(embeddings, self.dim)
        n = len(embeddings)

        chosen = index_type or self._resolve_index_type(n)
        if self.corpus_size is None:
            self.corpus_size = n

        logger.info("Building %s index | n=%d dim=%d", chosen, n, self.dim)
        t0 = time.perf_counter()

        self._index = self._make_index(chosen)
        self._index_type = chosen

        if chosen != "Flat":
            logger.info("Training %s index on %d vectors...", chosen, n)
            self._index.train(embeddings)

        elapsed = time.perf_counter() - t0
        IndexStats(
            index_type=chosen,
            dim=self.dim,
            ntotal=0,
            is_trained=self._index.is_trained,
            nlist=getattr(self, "nlist", None),
            m_subvectors=getattr(self, "m_subvectors", None),
            nbits=self.nbits if chosen == "IVFPQ" else None,
            build_time_seconds=elapsed,
        ).log()

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            logger.info("Index moved to GPU")

        return self

    # ------------------------------------------------------------------ #
    # Adding vectors                                                       #
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        embeddings: np.ndarray,
        batch_size: int = 4096,
    ) -> "FAISSIndexBuilder":
        """Add *embeddings* to the index in batches.

        The index must be trained before calling this method (either by
        calling :meth:`build` first, or by loading a pre-built index).

        Args:
            embeddings: Float32 array of shape ``(n, dim)``.
            batch_size: Number of vectors added per FAISS call.

        Returns:
            ``self`` for chaining.

        Raises:
            RuntimeError: If the index has not been initialised yet.
        """
        if self._index is None:
            raise RuntimeError("Call build() or load() before add_documents().")

        embeddings = _validate_embeddings(embeddings, self.dim)
        n = len(embeddings)
        logger.info("Adding %d vectors in batches of %d...", n, batch_size)

        t0 = time.perf_counter()
        for batch in _iter_batches(embeddings, batch_size):
            self._index.add(batch)

        elapsed = time.perf_counter() - t0
        logger.info(
            "add_documents complete | ntotal=%d time=%.2fs",
            self._index.ntotal,
            elapsed,
        )
        IndexStats(
            index_type=self._index_type or "unknown",
            dim=self.dim,
            ntotal=self._index.ntotal,
            is_trained=self._index.is_trained,
            build_time_seconds=elapsed,
        ).log()
        return self

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the *k* nearest neighbours for each query vector.

        Args:
            query: Float32 array of shape ``(q, dim)`` or ``(dim,)``.
            k: Number of results per query.
            nprobe: Number of IVF cells to probe (ignored for Flat).

        Returns:
            Tuple of ``(distances, indices)`` each of shape ``(q, k)``.
        """
        if self._index is None:
            raise RuntimeError("Index is not initialised.")

        query = _validate_embeddings(query, self.dim)
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = nprobe

        distances, indices = self._index.search(query, k)
        return distances, indices

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | os.PathLike) -> None:
        """Persist the index to disk.

        Args:
            path: File path (e.g. ``"data/my_index.faiss"``).

        Raises:
            RuntimeError: If the index has not been initialised.
        """
        if self._index is None:
            raise RuntimeError("Nothing to save -- index is not initialised.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_index = (
            faiss.index_gpu_to_cpu(self._index) if self.use_gpu else self._index
        )
        faiss.write_index(cpu_index, str(path))
        logger.info("Index saved -> %s (ntotal=%d)", path, cpu_index.ntotal)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        dim: int,
        use_gpu: bool = False,
    ) -> "FAISSIndexBuilder":
        """Load a persisted FAISS index from disk.

        Args:
            path: Path to the ``.faiss`` file.
            dim: Dimensionality of the stored vectors.
            use_gpu: Move the index to GPU after loading.

        Returns:
            A :class:`FAISSIndexBuilder` instance with :attr:`_index` set.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        obj = cls(dim=dim, use_gpu=use_gpu)
        obj._index = faiss.read_index(str(path))
        obj._index_type = _infer_index_type(obj._index)
        logger.info(
            "Index loaded <- %s (type=%s ntotal=%d)",
            path,
            obj._index_type,
            obj._index.ntotal,
        )
        if use_gpu:
            res = faiss.StandardGpuResources()
            obj._index = faiss.index_cpu_to_gpu(res, 0, obj._index)
        return obj

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def ntotal(self) -> int:
        """Total number of vectors currently stored in the index."""
        return self._index.ntotal if self._index else 0

    @property
    def index_type(self) -> Optional[str]:
        """Human-readable index type string."""
        return self._index_type


# -- Helpers -----------------------------------------------------------------

def _validate_embeddings(arr: np.ndarray, dim: int) -> np.ndarray:
    """Ensure *arr* is a 2-D float32 array with the expected dimension."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    if arr.shape[1] != dim:
        raise ValueError(f"Expected dim={dim}, got {arr.shape[1]}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _iter_batches(arr: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield successive row-slices of *arr*."""
    for start in range(0, len(arr), batch_size):
        yield arr[start : start + batch_size]


def _infer_index_type(index: faiss.Index) -> str:
    """Return a short string identifying the FAISS index class."""
    name = type(index).__name__
    if "IVFPQ" in name:
        return "IVFPQ"
    if "IVFFlat" in name:
        return "IVFFlat"
    if "Flat" in name:
        return "Flat"
    return name

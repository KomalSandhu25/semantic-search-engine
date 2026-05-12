#!/usr/bin/env python3
"""
build_index.py
==============
CLI entry-point for encoding a corpus and building a FAISS index.

Supports two corpus modes:

1. **MS MARCO passage** (``--corpus msmarco``) -- downloads via HuggingFace datasets.
2. **Custom CSV / JSONL** (``--corpus path/to/file``) -- expects a ``text`` column.

Usage example
-------------
.. code-block:: bash

    python scripts/build_index.py \
        --corpus msmarco \
        --encoder sentence-transformers/multi-qa-mpnet-base-dot-v1 \
        --index-out data/index.faiss \
        --store-out data/doc_store.ndjson \
        --batch-size 512 \
        --sample 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def _iter_msmarco(sample: int) -> Iterator[tuple[int, str]]:
    """Yield ``(id, text)`` pairs from MS MARCO.

    Args:
        sample: Maximum number of passages to yield (0 = all).

    Yields:
        Tuples of ``(passage_id, text)``.
    """
    from datasets import load_dataset

    logger.info("Loading MS MARCO passages (sample=%s)...", sample or "all")
    ds = load_dataset("ms_marco", "v1.1", split="train", trust_remote_code=True)
    count = 0
    for row in ds:
        for pid, passage in zip(row["passages"]["passage_id"], row["passages"]["passage_text"]):
            yield int(pid), passage
            count += 1
            if sample and count >= sample:
                return


def _iter_custom(path: Path, sample: int) -> Iterator[tuple[int, str]]:
    """Yield ``(index, text)`` pairs from a CSV or JSONL file.

    Args:
        path: Path to a ``.csv`` or ``.jsonl`` file.
        sample: Maximum number of records to yield (0 = all).

    Yields:
        Tuples of ``(row_index, text)``.

    Raises:
        ValueError: If the file extension is not recognised.
    """
    suffix = path.suffix.lower()
    count = 0

    if suffix == ".csv":
        import csv
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                text = row.get("text") or row.get("passage") or next(iter(row.values()))
                yield idx, text.strip()
                count += 1
                if sample and count >= sample:
                    return

    elif suffix in {".jsonl", ".json"}:
        import json
        with open(path, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("passage") or ""
                yield idx, text
                count += 1
                if sample and count >= sample:
                    return
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Use .csv or .jsonl.")


def encode_corpus(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    """Encode a list of texts into L2-normalised embedding vectors.

    Args:
        texts: Raw text strings.
        model_name: HuggingFace model identifier or local path.
        batch_size: Batch size passed to sentence-transformers.

    Returns:
        Float32 NumPy array of shape ``(n, dim)``.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading encoder: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Encoding %d texts (batch_size=%d)...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Encode a text corpus and build a FAISS index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", default="msmarco", help="'msmarco' or path to .csv/.jsonl")
    parser.add_argument("--encoder", default="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    parser.add_argument("--index-out", default="data/index.faiss")
    parser.add_argument("--store-out", default="data/doc_store.ndjson")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sample", type=int, default=0, help="Max docs (0=all)")
    parser.add_argument("--index-type", choices=["auto","Flat","IVFFlat","IVFPQ"], default="auto")
    parser.add_argument("--nlist", type=int, default=None)
    parser.add_argument("--m-subvectors", type=int, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    from src.index.builder import FAISSIndexBuilder
    from src.index.document_store import DocumentStore

    args = parse_args(argv)

    corpus_path = Path(args.corpus)
    if args.corpus == "msmarco":
        pairs = list(_iter_msmarco(args.sample))
    elif corpus_path.exists():
        pairs = list(_iter_custom(corpus_path, args.sample))
    else:
        logger.error("Corpus not found: %s", args.corpus)
        return 1

    if not pairs:
        logger.error("Corpus is empty.")
        return 1

    doc_ids, texts = zip(*pairs)
    doc_ids, texts = list(doc_ids), list(texts)
    logger.info("Corpus loaded: %d documents", len(texts))

    embeddings = encode_corpus(texts, args.encoder, args.batch_size)
    dim = embeddings.shape[1]
    logger.info("Embeddings shape: %s", embeddings.shape)

    builder = FAISSIndexBuilder(
        dim=dim,
        corpus_size=len(texts),
        nlist=args.nlist,
        m_subvectors=args.m_subvectors,
        use_gpu=args.use_gpu,
    )
    forced_type = None if args.index_type == "auto" else args.index_type
    builder.build(embeddings, index_type=forced_type)
    builder.add_documents(embeddings)
    builder.save(args.index_out)
    logger.info("Index | type=%s ntotal=%d", builder.index_type, builder.ntotal)

    store = DocumentStore(args.store_out)
    store.add_batch(doc_ids, texts)
    store.save()
    logger.info("DocumentStore stats: %s", store.stats())
    logger.info("Done. Index -> %s  Store -> %s", args.index_out, args.store_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

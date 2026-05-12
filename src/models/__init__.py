"""Encoder model wrappers for the two-stage retrieval pipeline.

Stage 1 -- Bi-encoder (recall)
    Encodes queries and documents into a shared dense vector space using a
    Siamese network architecture.  At search time only the query is encoded;
    document embeddings are pre-computed and stored in a FAISS index for
    sub-millisecond ANN retrieval.

Stage 2 -- Cross-encoder (precision / re-ranking)
    Takes the top-K bi-encoder candidates and scores each (query, document)
    pair jointly.  More expensive than the bi-encoder but substantially more
    accurate, because the attention layers can attend to both sequences
    simultaneously.

Submodules
----------
bi_encoder
    BiEncoder -- wraps SentenceTransformer for batch encoding and
    FAISS-compatible L2-normalised embeddings.
cross_encoder
    CrossEncoder -- wraps sentence_transformers.CrossEncoder for
    pairwise re-ranking.
"""

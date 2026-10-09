"""A tiny in-memory vector store with cosine-similarity search.

Vectors are L2-normalized by the embedder, so cosine similarity is just a dot
product. For a few hundred chunks this brute-force search is instant; the same
interface would be swapped for FAISS/Chroma at larger scale.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, chunks: List[Chunk], vectors: np.ndarray):
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks and vectors must match.")
        self.chunks = chunks
        self.vectors = vectors  # shape (n_chunks, dim), unit-normalized

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs, best first."""
        # query_vector: shape (dim,) or (1, dim)
        q = query_vector.reshape(-1)
        scores = self.vectors @ q  # cosine similarity for unit vectors
        top_k = min(top_k, len(self.chunks))
        # argpartition for speed, then sort just the top_k
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.chunks)

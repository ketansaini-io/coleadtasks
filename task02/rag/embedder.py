"""Text embedding with a graceful fallback.

Two backends are supported:

1. ``sentence-transformers`` (model: all-MiniLM-L6-v2) — true semantic embeddings.
   Used automatically if the library is installed.
2. ``TF-IDF`` (scikit-learn) — a classic lexical vector space. Always available and
   needs no model download, so the project runs offline out of the box.

Both expose the same interface:
    embedder.fit(corpus)          # learn vocabulary (no-op for transformers)
    embedder.encode(list_of_str)  # -> L2-normalized numpy array (n, dim)

Normalizing to unit length means a dot product equals cosine similarity, which
keeps the retriever simple and fast.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class TfidfEmbedder:
    """Lexical TF-IDF embedder (always-available fallback)."""

    backend = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        # sublinear_tf dampens very frequent terms; 1-2 grams capture short phrases.
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, corpus: List[str]) -> "TfidfEmbedder":
        self.vectorizer.fit(corpus)
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.encode called before fit().")
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return _l2_normalize(matrix)

    @property
    def dim(self) -> int:
        return len(self.vectorizer.vocabulary_)


class SentenceTransformerEmbedder:
    """Dense semantic embedder using sentence-transformers."""

    backend = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def fit(self, corpus: List[str]) -> "SentenceTransformerEmbedder":
        # Pretrained model needs no fitting.
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        return _l2_normalize(vecs)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


def build_embedder(prefer_semantic: bool = True):
    """Return the best available embedder.

    Tries sentence-transformers first (when ``prefer_semantic``), then falls back
    to TF-IDF. The choice is reported so the README/demo can state which backend ran.
    """
    if prefer_semantic:
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # library missing or model can't load
            print(
                f"[embedder] sentence-transformers unavailable ({exc.__class__.__name__}); "
                "falling back to TF-IDF."
            )
    return TfidfEmbedder()

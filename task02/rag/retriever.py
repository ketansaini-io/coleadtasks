"""Retriever: builds the index from documents and fetches relevant chunks.

This is the "R" in RAG. Given a query string it returns the most relevant chunks
together with their similarity scores and source citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .chunking import Chunk, load_documents
from .embedder import build_embedder
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def text(self) -> str:
        return self.chunk.text


class Retriever:
    def __init__(self, data_dir: str, prefer_semantic: bool = True):
        self.chunks: List[Chunk] = load_documents(data_dir)
        if not self.chunks:
            raise RuntimeError(f"No documents found in {data_dir!r}.")

        self.embedder = build_embedder(prefer_semantic=prefer_semantic)
        # Embed the heading trail together with the body: headings such as
        # "Fee Structure > B.Tech Tuition Fees" carry strong topical signal and
        # markedly improve retrieval (the body alone often omits those keywords).
        # We still display only the body text to the user.
        corpus = [self._embed_text(c) for c in self.chunks]
        self.embedder.fit(corpus)
        vectors = self.embedder.encode(corpus)
        self.store = VectorStore(self.chunks, vectors)

    @staticmethod
    def _embed_text(chunk: Chunk) -> str:
        return f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text

    @property
    def backend(self) -> str:
        return self.embedder.backend

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        q_vec = self.embedder.encode([query])[0]
        hits = self.store.search(q_vec, top_k=top_k)
        return [RetrievedChunk(chunk=c, score=s) for c, s in hits]

    def stats(self) -> dict:
        return {
            "num_documents": len({c.source for c in self.chunks}),
            "num_chunks": len(self.chunks),
            "embedding_backend": self.backend,
            "embedding_dim": getattr(self.embedder, "dim", None),
        }

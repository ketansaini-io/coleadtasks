"""Answer generation grounded in retrieved chunks (the "G" in RAG).

Two modes, chosen automatically:

1. **LLM mode** — if an OpenAI-compatible API key is configured (env var
   ``OPENAI_API_KEY``), the retrieved chunks are passed as context to a chat model
   with a strict instruction to answer *only* from that context and to say when the
   answer isn't present. This is the "ideal" RAG generation path.

2. **Extractive mode (default, offline)** — no API key needed. The answer is built
   from the most query-relevant sentences inside the retrieved chunks, so every
   sentence shown is literally grounded in the source documents. This guarantees the
   assistant never hallucinates facts that aren't in the college documents.

Either way the answer is accompanied by the source chunk(s) it came from.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy as np

from .retriever import RetrievedChunk

# Below this top similarity we assume the documents don't cover the question.
# Out-of-scope questions score ~0.0 with either backend, while a valid single-keyword
# match scores ~0.1+, so a low floor cleanly separates the two.
RELEVANCE_FLOOR = 0.05


def _split_sentences(text: str) -> List[str]:
    # Split on sentence punctuation, but keep bullet / table lines as their own units.
    parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "|")) or line[:2].strip().isdigit():
            parts.append(line)
        else:
            parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return parts


class AnswerGenerator:
    def __init__(self, embedder=None):
        # Reusing the retriever's embedder keeps sentence scoring consistent with
        # retrieval. Optional: extractive mode still works without it (lexical overlap).
        self.embedder = embedder
        self.use_llm = bool(os.environ.get("OPENAI_API_KEY"))
        self.llm_model = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")

    # ---- public API -----------------------------------------------------

    def generate(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        if not retrieved or retrieved[0].score < RELEVANCE_FLOOR:
            return (
                "I could not find this information in the Aurora Institute of "
                "Technology documents. Please rephrase, or contact the college office."
            )
        if self.use_llm:
            try:
                return self._generate_llm(query, retrieved)
            except Exception as exc:
                print(f"[generator] LLM call failed ({exc}); using extractive mode.")
        return self._generate_extractive(query, retrieved)

    # ---- extractive (offline) mode -------------------------------------

    def _score_sentences(self, query: str, sentences: List[str]) -> np.ndarray:
        if self.embedder is not None:
            q = self.embedder.encode([query])[0]
            s = self.embedder.encode(sentences)
            return s @ q
        # Lexical fallback: Jaccard-style token overlap.
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scores = []
        for sent in sentences:
            s_tokens = set(re.findall(r"\w+", sent.lower()))
            overlap = len(q_tokens & s_tokens)
            scores.append(overlap / (len(q_tokens) + 1e-6))
        return np.array(scores, dtype=np.float32)

    def _generate_extractive(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        # Gather candidate sentences from the top chunks, remembering their source.
        candidates: List[tuple] = []  # (sentence, citation)
        for rc in retrieved:
            for sent in _split_sentences(rc.text):
                # Skip tiny prose fragments (line-wrap artifacts like "The minimum")
                # but keep bullet/table rows, which are legitimately short.
                is_listish = sent.startswith(("-", "*", "|")) or sent[:2].strip().isdigit()
                if not is_listish and len(sent.split()) < 4:
                    continue
                candidates.append((sent, rc.citation))

        if not candidates:
            return retrieved[0].text

        sentences = [c[0] for c in candidates]
        scores = self._score_sentences(query, sentences)

        # Pick the top few sentences, keep original order for readability,
        # and avoid near-duplicates.
        order = np.argsort(-scores)
        top_score = float(scores[order[0]]) if len(order) else 0.0
        # Keep a supporting sentence only if it is reasonably close to the best
        # one; this trims weakly-related lines from the answer.
        keep_floor = max(0.0, 0.45 * top_score)
        chosen_idx = []
        seen = set()
        for i in order:
            if scores[i] <= 0 or scores[i] < keep_floor:
                break
            key = sentences[i][:60].lower()
            if key in seen:
                continue
            seen.add(key)
            chosen_idx.append(i)
            if len(chosen_idx) >= 3:
                break

        if not chosen_idx:
            chosen_idx = [int(np.argmax(scores))]

        # Keep the most relevant sentence first (chosen_idx is already in
        # descending-score order) rather than reordering by position.
        answer_lines = [sentences[i] for i in chosen_idx]
        return "\n".join(answer_lines)

    # ---- LLM mode -------------------------------------------------------

    def _generate_llm(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        from openai import OpenAI

        client = OpenAI()
        context_blocks = []
        for i, rc in enumerate(retrieved, 1):
            context_blocks.append(f"[Source {i}: {rc.citation}]\n{rc.text}")
        context = "\n\n".join(context_blocks)

        system = (
            "You are the Aurora Institute of Technology assistant. Answer the "
            "student's question using ONLY the provided context. If the answer is "
            "not in the context, say you don't have that information. Be concise "
            "and do not invent facts."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        resp = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

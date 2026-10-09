"""The college assistant: ties retrieval + generation together and (bonus)
maintains conversation history for context-aware follow-up questions.

Conversation history handling
-----------------------------
Short follow-ups like "what about ECE?" or "and its fees?" are not self-contained,
so retrieving on them alone fails. Before retrieval we build an *expanded query*
that folds in the recent conversation when the current question looks like a
follow-up (it is short, or opens with a pronoun / conjunction). This is a light,
dependency-free form of query rewriting that noticeably improves follow-up answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .generator import AnswerGenerator
from .retriever import RetrievedChunk, Retriever

# Referential words that, appearing ANYWHERE in the question, signal it depends on
# earlier context (e.g. "who heads that department?", "and its fees?").
REFERENTIAL_CUES = {
    "it", "its", "they", "them", "that", "this", "those", "these",
    "there", "he", "she", "their", "his", "her",
}
# Weak cues that only signal a follow-up when the question opens with them.
LEADING_CUES = ("and", "also", "same", "then", "what about", "how about")


@dataclass
class Turn:
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)


@dataclass
class AssistantResponse:
    answer: str
    sources: List[RetrievedChunk]
    expanded_query: str  # the query actually used for retrieval


class CollegeAssistant:
    def __init__(
        self,
        data_dir: str,
        prefer_semantic: bool = True,
        top_k: int = 3,
        keep_history: bool = True,
    ):
        self.retriever = Retriever(data_dir, prefer_semantic=prefer_semantic)
        self.generator = AnswerGenerator(embedder=self.retriever.embedder)
        self.top_k = top_k
        self.keep_history = keep_history
        self.history: List[Turn] = []

    # ---- query expansion (conversation history) ------------------------

    def _looks_like_followup(self, question: str) -> bool:
        q = question.lower().strip()
        tokens = set(re.findall(r"\w+", q))
        if len(q.split()) <= 4:
            return True
        if tokens & REFERENTIAL_CUES:  # a demonstrative/pronoun anywhere
            return True
        return any(q.startswith(cue) for cue in LEADING_CUES)

    def _expand_query(self, question: str) -> str:
        """Fold recent context into a short follow-up question."""
        if not self.keep_history or not self.history:
            return question
        if not self._looks_like_followup(question):
            return question
        # Use the most recent question (and its answer keywords) as context.
        prev = self.history[-1]
        prev_keywords = self._keywords(prev.question + " " + prev.answer)
        return f"{prev_keywords} {question}".strip()

    @staticmethod
    def _keywords(text: str, limit: int = 8) -> str:
        stop = {
            "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is",
            "are", "what", "which", "how", "much", "many", "do", "does", "per",
            "about", "at", "with", "can", "i", "my", "you", "your",
        }
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        seen, out = set(), []
        for w in words:
            lw = w.lower()
            if lw in stop or lw in seen or len(lw) < 3:
                continue
            seen.add(lw)
            out.append(w)
            if len(out) >= limit:
                break
        return " ".join(out)

    # ---- main entry point ----------------------------------------------

    def ask(self, question: str) -> AssistantResponse:
        expanded = self._expand_query(question)
        retrieved = self.retriever.retrieve(expanded, top_k=self.top_k)
        answer = self.generator.generate(question, retrieved)

        if self.keep_history:
            self.history.append(
                Turn(
                    question=question,
                    answer=answer,
                    sources=[rc.citation for rc in retrieved],
                )
            )
        return AssistantResponse(
            answer=answer, sources=retrieved, expanded_query=expanded
        )

    def reset(self) -> None:
        self.history.clear()

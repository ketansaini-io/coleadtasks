"""Non-interactive demo.

Runs a fixed set of sample questions through the assistant and prints, for each,
the grounded answer and the retrieved source chunk(s). The last block demonstrates
the conversation-history bonus with context-dependent follow-up questions.

Run:
    python demo.py
"""

from __future__ import annotations

import os

from rag.assistant import CollegeAssistant

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SINGLE_QUESTIONS = [
    "What is the tuition fee for the AI and Data Science program?",
    "What is the minimum attendance required to sit for exams?",
    "Who is the head of the AI-DS department and when are their office hours?",
    "When does the odd semester begin?",
    "How do I join a club and how many clubs can I join?",
    "What is the passing mark in a course?",
    "Is ragging allowed on campus?",
    "What is the capital of France?",  # out-of-scope: should refuse gracefully
]

# A multi-turn conversation that relies on earlier context (bonus feature).
CONVERSATION = [
    "What courses does the college offer?",
    "What is the fee for CSE?",
    "What about ECE?",                 # follow-up: "ECE fee"
    "Who is the head of that department?",  # follow-up: ECE HOD
]


def print_response(question, resp):
    print(f"Q: {question}")
    print(f"A: {resp.answer}")
    if resp.expanded_query != question:
        print(f"   (retrieved using: \"{resp.expanded_query}\")")
    print("   Sources:")
    for i, rc in enumerate(resp.sources, 1):
        preview = rc.text.replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + " ..."
        print(f"     {i}. [{rc.citation}] (score={rc.score:.3f}) \"{preview}\"")
    print("-" * 70)


def main():
    assistant = CollegeAssistant(DATA_DIR)
    stats = assistant.retriever.stats()
    print("=" * 70)
    print("Aurora Institute of Technology - RAG Assistant - DEMO")
    print(
        f"Indexed {stats['num_chunks']} chunks from {stats['num_documents']} docs "
        f"| embeddings: {stats['embedding_backend']} "
        f"(dim={stats['embedding_dim']})"
    )
    print("=" * 70)

    print("\n### Part 1 — Single questions (fresh context each time)\n")
    for q in SINGLE_QUESTIONS:
        assistant.reset()  # isolate each single question
        print_response(q, assistant.ask(q))

    print("\n### Part 2 — Multi-turn conversation (history / bonus)\n")
    assistant.reset()
    for q in CONVERSATION:
        print_response(q, assistant.ask(q))


if __name__ == "__main__":
    main()

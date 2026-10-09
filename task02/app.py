"""Interactive command-line interface for the Aurora Institute of Technology
RAG assistant.

Run:
    python app.py

Type a question and press Enter. The assistant prints a grounded answer followed
by the source chunk(s) it used. Conversation history is kept, so follow-up
questions ("what about ECE?") work. Commands: 'reset' clears history, 'exit' quits.
"""

from __future__ import annotations

import os

from rag.assistant import CollegeAssistant

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

BANNER = r"""
======================================================================
  Aurora Institute of Technology  -  Student Assistant (RAG)
  Ask about courses, fees, clubs, timetable, faculty or rules.
  Commands:  reset  (clear history)   |   exit / quit
======================================================================
"""


def main():
    print(BANNER)
    assistant = CollegeAssistant(DATA_DIR)
    stats = assistant.retriever.stats()
    print(
        f"Indexed {stats['num_chunks']} chunks from {stats['num_documents']} "
        f"documents  |  embeddings: {stats['embedding_backend']}\n"
    )

    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if question.lower() == "reset":
            assistant.reset()
            print("(conversation history cleared)\n")
            continue

        resp = assistant.ask(question)

        print(f"\nAssistant > {resp.answer}\n")
        print("  Sources (retrieved chunks):")
        for i, rc in enumerate(resp.sources, 1):
            preview = rc.text.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + " ..."
            print(f"   {i}. [{rc.citation}]  (score={rc.score:.3f})")
            print(f"      \"{preview}\"")
        if resp.expanded_query != question:
            print(f"  (interpreted as: \"{resp.expanded_query}\")")
        print()


if __name__ == "__main__":
    main()

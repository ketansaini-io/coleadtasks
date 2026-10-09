# Task 2 — RAG-Based College Assistant

A small **Retrieval-Augmented Generation (RAG)** assistant for a fictional college,
**Aurora Institute of Technology (AIT)**. It answers student questions (courses, fees,
clubs, timetable, faculty, rules) **using only the provided documents**, and it shows
the exact **source chunk** each answer came from. Conversation history is supported
(the bonus), so follow-up questions work.

---

## What it does

- Loads the college documents in `data/`, splits them into chunks (keeping the
  heading trail as context), and builds a vector index.
- For each question it **retrieves** the most relevant chunks and **generates** a
  grounded answer from them, always printing the **retrieved source/chunk** + a
  similarity score.
- Refuses gracefully when the answer isn't in the documents (e.g. "What is the
  capital of France?") instead of making something up.
- **Bonus — conversation history:** short follow-ups like *"What about ECE?"* or
  *"Who is the head of that department?"* are rewritten using the previous turns
  before retrieval.

---

## How to run

The project runs **fully offline** with only `numpy` and `scikit-learn`.

```bash
pip install -r requirements.txt

# Option A — run the scripted demo (sample Q&A with retrieved chunks)
python demo.py

# Option B — chat with it interactively
python app.py
```

In the interactive app, type a question and press Enter. Commands: `reset` clears
the conversation history, `exit` / `quit` leaves.

A saved run of the demo is in **`sample_output.txt`**.

---

## Architecture

```
            ┌─────────────┐   chunks    ┌──────────────┐   vectors   ┌──────────────┐
 data/*.md  │  chunking   │ ──────────▶ │   embedder   │ ──────────▶ │ vector store │
            └─────────────┘             │ (ST / TF-IDF)│             │ (cosine sim) │
                                        └──────────────┘             └──────┬───────┘
                                                                            │ top-k
   question ─▶ CollegeAssistant ─▶ (history-aware query expansion) ─▶ Retriever
                                                                            │
                                                              retrieved chunks
                                                                            ▼
                                        grounded answer ◀─────────── AnswerGenerator
                                        + source citations
```

| File | Role |
|------|------|
| `rag/chunking.py`    | Load docs, split into heading-aware, overlapping chunks. |
| `rag/embedder.py`    | Embeddings: `sentence-transformers` if installed, else **TF-IDF** fallback. |
| `rag/vector_store.py`| In-memory cosine-similarity search over chunk vectors. |
| `rag/retriever.py`   | Builds the index and returns top-k chunks with scores + citations. |
| `rag/generator.py`   | Builds a grounded answer from retrieved chunks (extractive, or LLM if a key is set). |
| `rag/assistant.py`   | Orchestration + **conversation history** / follow-up query expansion. |
| `app.py`             | Interactive CLI. |
| `demo.py`            | Scripted sample questions + a multi-turn conversation. |
| `data/`              | The fictional college documents. |

### Retrieval (the "R")
- Documents are chunked on markdown headings/paragraphs with a small overlap. Each
  chunk keeps its **heading trail** (e.g. `fees.md > B.Tech Tuition Fees`), which is
  embedded *with* the body — headings carry strong topical signal and noticeably
  improve retrieval.
- Vectors are L2-normalized, so **cosine similarity is a dot product**; search is a
  single matrix–vector multiply. At larger scale this store would be swapped for
  FAISS/Chroma behind the same interface.

### Generation (the "G")
Chosen automatically:
- **Extractive (default, offline):** the answer is assembled from the most
  query-relevant sentences *inside* the retrieved chunks, so every line shown is
  literally grounded in the source documents — no hallucinated facts.
- **LLM (optional):** set `OPENAI_API_KEY` (and optionally `RAG_LLM_MODEL`) and the
  retrieved chunks are passed to a chat model with a strict "answer only from this
  context" instruction. This gives more fluent answers; retrieval is unchanged.

### Embeddings backend
- If `sentence-transformers` is installed, dense semantic embeddings
  (`all-MiniLM-L6-v2`) are used automatically and give cleaner answers.
- Otherwise the assistant falls back to **TF-IDF** so it always runs with no model
  download and no internet. The demo header prints which backend is active.

---

## Key results (TF-IDF offline run)

- Indexed **38 chunks from 6 documents**.
- Correctly answers course/fee/faculty/timetable/rule questions and **shows the
  retrieved source chunk** for each (see `sample_output.txt`).
- **Out-of-scope** questions are refused rather than answered.
- **Follow-ups resolve from history**, e.g.:
  - `What is the fee for CSE?` → CSE tuition.
  - `What about ECE?` → *(retrieved as ECE fee)* → ECE tuition.
  - `Who is the head of that department?` → *(resolved to ECE)* → **Dr. Kavita Deshmukh (ECE)**.

---

## Design notes & honest limitations

- The default offline path uses **lexical** TF-IDF scoring. It is robust and
  dependency-light, but because it matches on words rather than meaning, an
  occasional supporting sentence can be loosely related (e.g. a question about
  "hostel charges" may surface a nearby "hostel" rule line). The **lead sentence and
  the retrieved source chunk are correct**; installing `sentence-transformers` (one
  line in `requirements.txt`) upgrades scoring to semantic and cleans these up.
- Query expansion for follow-ups is an intentionally lightweight, dependency-free
  rewriter (keyword carry-over from the previous turn). An LLM-based query rewriter
  would be the production upgrade.

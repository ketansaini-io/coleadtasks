"""Document loading and chunking.

Each document is split into reasonably sized chunks so that retrieval can return
a focused, relevant passage rather than a whole file. We split on markdown headings
and paragraphs, and we attach the heading trail to every chunk so the retrieved
chunk carries its own context (e.g. "fees.md > B.Tech Tuition Fees").
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """A single retrievable piece of a document."""

    chunk_id: int
    source: str            # file name, e.g. "fees.md"
    heading: str           # heading trail, e.g. "Fee Structure > B.Tech Tuition Fees"
    text: str              # the chunk body (what we embed and show)
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Short human-readable source label shown next to answers."""
        if self.heading:
            return f"{self.source} > {self.heading}"
        return self.source


def _split_into_sections(markdown: str):
    """Yield (heading_trail, body_text) sections based on markdown headings.

    We track H1 (#) and H2 (##) headings to build a readable heading trail.
    """
    lines = markdown.splitlines()
    h1 = ""
    h2 = ""
    buffer: List[str] = []

    def heading_trail() -> str:
        parts = [p for p in (h1, h2) if p]
        return " > ".join(parts)

    for line in lines:
        h1_match = re.match(r"^#\s+(.*)", line)
        h2_match = re.match(r"^##\s+(.*)", line)
        h3_match = re.match(r"^###\s+(.*)", line)

        if h1_match or h2_match or h3_match:
            # flush the current buffer as a section before switching heading
            if buffer:
                body = "\n".join(buffer).strip()
                if body:
                    yield heading_trail(), body
                buffer = []
            if h1_match:
                h1 = h1_match.group(1).strip()
                h2 = ""
            elif h2_match:
                h2 = h2_match.group(1).strip()
            elif h3_match:
                # treat H3 as a sub-part appended to H2
                h2 = (h2.split(" / ")[0] if h2 else "") + " / " + h3_match.group(1).strip()
                h2 = h2.strip(" /")
        else:
            buffer.append(line)

    if buffer:
        body = "\n".join(buffer).strip()
        if body:
            yield heading_trail(), body


def _pack_paragraphs(body: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Pack paragraphs of a section into chunks of at most ``max_chars``.

    Long sections are broken up; small paragraphs are merged so a chunk is a
    meaningful unit. A small character overlap keeps context across the split.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= max_chars:
            current = current + "\n\n" + para
        else:
            chunks.append(current)
            # start the next chunk with a small tail overlap for continuity
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = (tail + "\n\n" + para).strip() if tail else para

    if current:
        chunks.append(current)

    # Hard-split any single chunk that is still too long (e.g. a big table).
    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars - overlap_chars):
                final.append(c[i : i + max_chars])
    return final


def chunk_markdown(
    markdown: str,
    source: str,
    start_id: int = 0,
    max_chars: int = 700,
    overlap_chars: int = 80,
) -> List[Chunk]:
    """Split one markdown document into a list of Chunk objects."""
    chunks: List[Chunk] = []
    cid = start_id
    for heading, body in _split_into_sections(markdown):
        for piece in _pack_paragraphs(body, max_chars, overlap_chars):
            chunks.append(
                Chunk(chunk_id=cid, source=source, heading=heading, text=piece)
            )
            cid += 1
    return chunks


def load_documents(data_dir: str) -> List[Chunk]:
    """Load and chunk every .md / .txt file in ``data_dir``.

    Returns a flat list of chunks with globally unique ids.
    """
    chunks: List[Chunk] = []
    files = sorted(
        f for f in os.listdir(data_dir) if f.endswith((".md", ".txt"))
    )
    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        chunks.extend(chunk_markdown(text, source=fname, start_id=len(chunks)))
    return chunks

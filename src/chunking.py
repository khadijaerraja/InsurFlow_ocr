"""Two chunking strategies, kept comparable so retrieval quality can be measured head to head.

Chunk boundaries decide what an embedding model has to compress into one vector,
which in turn decides what a retriever can find. Fixed-size chunking is the
default in most RAG tutorials because it needs no knowledge of document
structure, but it can split a fact from the heading that scopes it (e.g. a
waiting-period figure landing in a different chunk to the "Waiting Periods"
heading it belongs under). Section-aware chunking uses this corpus's Markdown
headings to keep each fact with its section label. Both are implemented here
so eval/questions.yaml can be run against each and the difference measured
rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    section: str
    text: str
    page: int | None = None  # 1-based page for chunks that come from paged documents
    document_id: int | None = None  # database id of the source document, when there is one
    contract: str | None = None  # contract number the document refers to (used for filtering)


def fixed_size_chunks(
    text: str, doc_id: str, size: int = 400, overlap: int = 80
) -> list[Chunk]:
    """Split text into overlapping fixed-width windows, ignoring document structure."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    step = size - overlap
    start = 0
    index = 0
    while start < len(text):
        window = text[start : start + size].strip()
        if window:
            chunks.append(
                Chunk(doc_id=doc_id, chunk_id=f"{doc_id}::fixed{index}", section="", text=window)
            )
            index += 1
        start += step
    return chunks


_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def section_aware_chunks(text: str, doc_id: str, max_size: int = 700) -> list[Chunk]:
    """Split on Markdown '## ' headings; oversized sections fall back to fixed windows.

    Each chunk is prefixed with the document's H1 title (e.g. "# Gold Family
    Health Policy") rather than emitting the title as its own standalone
    chunk. An early version of this function *did* emit a separate title
    /policy-code preamble chunk; measuring hit rate against eval/questions.yaml
    showed that turning the title into its own chunk (a) crowded out real
    section chunks when several documents' near-identical preamble text
    tied on similarity, and (b) discarding the title text entirely instead
    of just not chunking it separately was worse again -- the title carries
    the tier/variant wording ("Gold", "Family") that many gold questions
    reuse verbatim, and losing it from every chunk cost roughly 0.33 hit
    rate. Prefixing keeps that signal on every retrievable chunk without
    creating a low-information chunk of its own.
    """
    matches = list(_HEADING_RE.finditer(text))
    title_line = text.splitlines()[0].lstrip("# ").strip() if text.startswith("#") else ""
    chunks = []

    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        if len(body) <= max_size:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::{_slug(section_name)}",
                    section=section_name,
                    text=f"{title_line}\n{section_name}\n{body}",
                )
            )
        else:
            sub_chunks = fixed_size_chunks(body, doc_id, size=max_size, overlap=100)
            for j, sub in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}::{_slug(section_name)}{j}",
                        section=section_name,
                        text=f"{title_line}\n{section_name}\n{sub.text}",
                    )
                )
    return chunks


def _slug(section_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section_name.lower()).strip("-")


def split_passages(text: str, max_chars: int = 700) -> list[str]:
    """Cut one page of text into passages for search.

    Short, focused passages embed and rank better than a whole page. Cuts happen at
    upper-case headings (COVERAGE, EXCLUSIONS...) first, then at line boundaries when a
    section is longer than `max_chars`, so a heading always stays with its content.
    """
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines_in, i = [], 0
    while i < len(raw):  # digital PDFs emit "Expiry date:" and its value as two lines: keep them together
        if raw[i].endswith(":") and i + 1 < len(raw) and not raw[i + 1].endswith(":"):
            lines_in.append(f"{raw[i]} {raw[i + 1]}")
            i += 2
        else:
            lines_in.append(raw[i])
            i += 1
    sections, current = [], []
    for line in lines_in:
        stripped = line
        if stripped.isupper() and len(stripped) > 3 and current and len(stripped) < 40:
            sections.append(current)
            current = []
        current.append(stripped)
    if current:
        sections.append(current)

    passages = []
    for lines in sections:
        buf, size = [], 0
        for line in lines:
            if buf and size + len(line) > max_chars:
                passages.append("\n".join(buf))
                buf, size = [], 0
            buf.append(line)
            size += len(line) + 1
        if buf:
            passages.append("\n".join(buf))
    return passages or ([text.strip()] if text.strip() else [])

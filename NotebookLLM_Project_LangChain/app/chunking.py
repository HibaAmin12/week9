"""Hierarchical chunking with LangChain's RecursiveCharacterTextSplitter.

Step 1 (loaders): split by the file's real structure (page, heading, slide, row group) -> list of Documents ("units").
Step 2 (here): inside each unit, cut ~CHUNK_SIZE character windows with CHUNK_OVERLAP overlap, preferring paragraph
breaks, then sentence ends, then words. A chunk never crosses a unit boundary, so the citation label
(page, slide, section) is always correct.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def merge_small(units: list, min_chars: int = None) -> list:
    """Merge a tiny mergeable unit into the one after it, so headings are not left alone."""
    min_chars = config.MIN_SECTION_CHARS if min_chars is None else min_chars
    out, carry = [], None
    for u in units:
        if carry is not None:
            u = Document(page_content=carry.page_content + "\n\n" + u.page_content, metadata=dict(carry.metadata))
            carry = None
        if u.metadata.get("mergeable") and len(u.page_content) < min_chars:
            carry = u
        else:
            out.append(u)
    if carry is not None:
        out.append(carry)
    return out


def get_splitter(size: int = None, overlap: int = None) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size or config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP if overlap is None else overlap,
        separators=[r"\n\n", r"(?<=[.!?\u06d4])\s", r"\s", ""],   # paragraph, sentence end (also Urdu full stop), word
        is_separator_regex=True,
        add_start_index=True)


def make_chunks(units: list, size: int = None, overlap: int = None) -> list:
    """Turn unit Documents into chunk dicts: {unit, label, text, start, ctx}.

    `text` is what the user sees and the LLM reads. `ctx` (heading + text) is what gets embedded.
    """
    splitter = get_splitter(size, overlap)
    chunks = []
    for ui, u in enumerate(units):
        base = Document(page_content=u.page_content, metadata={"unit": ui})
        for piece in splitter.split_documents([base]):
            text = piece.page_content.strip()
            if not text:
                continue
            heading = u.metadata.get("heading") or ""
            prefix = f"{heading}\n" if heading else ""
            chunks.append({"unit": ui, "label": u.metadata["label"], "text": text,
                           "start": piece.metadata.get("start_index", 0), "ctx": prefix + text})
    return chunks


def units_to_dicts(units: list) -> list:
    """Plain JSON form of the units (stored in doc.json and shown by the source viewer)."""
    return [{"label": u.metadata["label"], "text": u.page_content, "heading": u.metadata.get("heading", ""),
             "mergeable": bool(u.metadata.get("mergeable"))} for u in units]

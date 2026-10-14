"""Cross-encoder reranker as a LangChain `BaseDocumentCompressor` (runs locally through fastembed, free).

It re-orders the documents returned by the FAISS retriever by true relevance to the query.
"""
import threading
from typing import Any, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from . import config
from .costs import estimate_tokens

_lock = threading.Lock()
_model = None


def _load():
    global _model
    with _lock:
        if _model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _model = TextCrossEncoder(model_name=config.RERANK_MODEL)
        return _model


class FastEmbedReranker(BaseDocumentCompressor):
    top_n: int = 8

    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Optional[Callbacks] = None) -> Sequence[Document]:
        scores = list(_load().rerank(query, [d.metadata.get("text", d.page_content) for d in documents]))
        ranked = sorted(zip(documents, scores), key=lambda t: -float(t[1]))
        return [Document(page_content=d.page_content, metadata={**d.metadata, "relevance_score": float(s)})
                for d, s in ranked[:self.top_n]]


def rerank(query: str, docs: list):
    """Return (reranked_docs, tokens). (None, 0) when reranking is off or unavailable (vector order is kept)."""
    if not config.RERANK_ENABLED or config.EMBED_PROVIDER == "fake" or len(docs) < 2:
        return None, 0
    try:
        out = FastEmbedReranker(top_n=len(docs)).compress_documents(docs, query)
    except Exception as exc:  # model missing or failed: keep vector order
        print(f"[rerank] disabled for this query: {exc}")
        return None, 0
    return list(out), estimate_tokens(query) * len(docs) + sum(estimate_tokens(d.metadata.get("text", d.page_content)) for d in docs)

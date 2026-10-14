"""Embeddings as LangChain `Embeddings` objects.

local  : BAAI/bge-small-en-v1.5 through fastembed (CPU, free)           -> LocalEmbeddings
openai : OpenAI embeddings API (billed)                                 -> langchain_openai.OpenAIEmbeddings
fake   : deterministic hashing embedder, used by tests only             -> FakeEmbeddings

The vector store (FAISS) calls these through the standard `embed_documents` / `embed_query` interface.
"""
import hashlib
import re
import threading

import numpy as np
from langchain_core.embeddings import Embeddings

from . import config
from .costs import estimate_tokens

_lock = threading.Lock()
_cache = {}


def embed_id() -> str:
    """Identifies the embedding setup; stored with the FAISS index so a stale index is detected."""
    name = {"local": config.EMBED_MODEL, "openai": config.OPENAI_EMBED_MODEL}.get(config.EMBED_PROVIDER, "fake")
    return f"{config.EMBED_PROVIDER}:{name}"


class LocalEmbeddings(Embeddings):
    """fastembed (ONNX, CPU) behind the LangChain Embeddings interface. BGE needs a prefix on queries."""

    def __init__(self, model_name: str, query_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._model = None
        self._mlock = threading.Lock()

    def _get(self):
        with self._mlock:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            return self._model

    def embed_documents(self, texts):
        return [v.tolist() for v in self._get().embed(list(texts), batch_size=config.EMBED_BATCH)]

    def embed_query(self, text):
        return next(iter(self._get().embed([self.query_prefix + text]))).tolist()


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashing embedder. Deterministic, no network, no model download."""

    def _vec(self, text):
        v = np.zeros(256, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        return (v + 1e-6).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _build() -> Embeddings:
    if config.EMBED_PROVIDER == "fake":
        return FakeEmbeddings()
    if config.EMBED_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set (needed for OpenAI embeddings).")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY,
                                base_url=config.OPENAI_BASE_URL, timeout=config.LLM_TIMEOUT,
                                check_embedding_ctx_length=False)
    prefix = config.BGE_QUERY_PREFIX if "bge" in config.EMBED_MODEL.lower() else ""
    return LocalEmbeddings(config.EMBED_MODEL, prefix)


def get_embeddings() -> Embeddings:
    key = embed_id()
    with _lock:
        if key not in _cache:
            _cache[key] = _build()
        return _cache[key]


def _unit_length(vectors):
    m = np.array(vectors, dtype=np.float32)
    m = m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)
    return m.tolist()


class LazyEmbeddings(Embeddings):
    """What FAISS uses. Resolves the real embedder at call time (so the app starts, and shows 'stale' sources,
    even without a key) and returns unit-length vectors, so FAISS inner product == cosine similarity."""

    def embed_documents(self, texts):
        return _unit_length(get_embeddings().embed_documents(texts))

    def embed_query(self, text):
        return _unit_length([get_embeddings().embed_query(text)])[0]


def count_tokens(texts) -> int:
    """Token estimate (about 4 chars per token). LangChain embedders do not return usage."""
    return sum(estimate_tokens(t) for t in texts)

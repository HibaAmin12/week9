"""Central configuration. Every value can be overridden with an environment variable (see .env.example)."""
import os


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---------- Paths ----------
DATA_DIR = _env("DATA_DIR", "data")
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 50)
ALLOWED_EXT = {e.strip().lower() for e in _env("ALLOWED_EXT", ".pdf,.docx,.pptx,.xlsx,.csv,.md,.txt,.json,.png,.jpg,.jpeg").split(",") if e.strip()}

# ---------- Chunking ----------
CHUNK_SIZE = _int("CHUNK_SIZE", 1200)        # characters (~300 tokens)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)   # characters shared between neighbours
MIN_SECTION_CHARS = _int("MIN_SECTION_CHARS", 300)  # smaller sections merge into the next one
ROWS_PER_UNIT = _int("ROWS_PER_UNIT", 25)    # spreadsheet rows per unit
TEXT_UNIT_CHARS = _int("TEXT_UNIT_CHARS", 3000)  # plain .txt files without page markers

# ---------- OCR (scanned PDFs and images) ----------
OCR_ENABLED = _bool("OCR_ENABLED", True)
OCR_LANG = _env("OCR_LANG", "eng")
OCR_DPI = _int("OCR_DPI", 200)
OCR_MIN_CHARS = _int("OCR_MIN_CHARS", 25)    # a PDF page with less text than this is OCR'd

# ---------- Embeddings ----------
# "local"  -> fastembed on CPU, free, no network at runtime
# "openai" -> OpenAI embeddings API, billed per token
# "fake"   -> hashing embedder, used by tests only
EMBED_PROVIDER = _env("EMBED_PROVIDER", "local")
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
OPENAI_EMBED_MODEL = _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH = _int("EMBED_BATCH", 32)
BGE_QUERY_PREFIX = _env("BGE_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
# USD per 1M tokens. Local models cost nothing; OpenAI default is text-embedding-3-small.
EMBED_PRICE_PER_M = _float("EMBED_PRICE_PER_M", 0.02 if EMBED_PROVIDER == "openai" else 0.0)

# ---------- Retrieval ----------
RETRIEVE_K = _int("RETRIEVE_K", 20)          # candidates from vector search
TOP_K = _int("TOP_K", 8)                     # passages sent to the LLM
BROAD_K = _int("BROAD_K", 24)                # evenly sampled passages for summary-style questions
NEIGHBORS = _int("NEIGHBORS", 1)             # chunks before/after each hit also shown to the LLM (0 = off)
CITE_MIN_SCORE=0.25
CITE_DROP_SCORE=0.05  # below this: no word overlap at all, the reference is dropped
RERANK_ENABLED = _bool("RERANK_ENABLED", True)
RERANK_MODEL = _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_PRICE_PER_M = _float("RERANK_PRICE_PER_M", 0.0)  # local reranker is free

# ---------- LLM (OpenAI) ----------
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-6-luna")
OPENAI_FALLBACK_MODELS = [m for m in _env("OPENAI_FALLBACK_MODELS", "gpt-4.1-mini,gpt-4o-mini").split(",") if m]
MAX_OUTPUT_TOKENS = _int("MAX_OUTPUT_TOKENS", 4000)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 120)
LLM_RETRIES = _int("LLM_RETRIES", 3)
# USD per 1M tokens. Check platform.openai.com/docs/pricing for YOUR model and update these.
PRICE_IN = _float("PRICE_IN", 0.10)
PRICE_OUT = _float("PRICE_OUT", 0.50)
HISTORY_TURNS = _int("HISTORY_TURNS", 6)
SUGGESTIONS = _bool("SUGGESTIONS", True)     # 3 follow-up question cards under each answer (one small extra LLM call)

# ---------- Guardrails ----------
GUARDRAILS_ENABLED = _bool("GUARDRAILS_ENABLED", True)
MODERATION_MODEL = _env("MODERATION_MODEL", "omni-moderation-latest")
# Put VERIFIED local helpline numbers here (or in .env). Do not trust numbers from memory.
HELPLINE_TEXT = _env("HELPLINE_TEXT", "Please contact your local emergency number or a trusted crisis helpline in your country.")
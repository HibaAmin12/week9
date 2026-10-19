import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root: support-agent/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
POLICIES_FILE = DATA_DIR / "policies.md"

COLLECTION_NAME = "support_policies"


def require_env(name: str) -> str:
    """Fail early with a clear message if a required env var is missing."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name} (check your .env file)")
    return value


# The OpenAI client reads this variable by itself; we only check it exists.
require_env("OPENAI_API_KEY")

MODEL_NAME = require_env("MODEL_NAME")      # e.g. openai:gpt-4o-mini
EMBED_MODEL = require_env("EMBED_MODEL")    # e.g. openai:text-embedding-3-small
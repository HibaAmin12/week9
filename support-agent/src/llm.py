import warnings

from langchain.chat_models import init_chat_model
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from config import BASE_DIR, MODEL_NAME

# Hide the harmless SQLiteCache deprecation warning (matched by its message text,
# so no extra import is needed).
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

# Cache identical LLM calls on disk: re-running the same prompt costs nothing.
set_llm_cache(SQLiteCache(database_path=str(BASE_DIR / ".llm_cache.db")))


def get_llm(temperature: float = 0):
    """Single place to create the chat model, so every module shares the same setup.

    max_retries=2 -> fail fast instead of waiting a long time on repeated errors.
    """
    return init_chat_model(MODEL_NAME, temperature=temperature, max_retries=2)
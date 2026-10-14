"""LLM through LangChain: ChatOpenAI + automatic fallback models + exact token usage.

`run(prompt, variables)` executes the LCEL chain  prompt | llm  and returns
{"text", "input_tokens", "output_tokens", "model"}. Errors become `LLMError` with a message safe to show to the user.
"""
import openai
from langchain_openai import ChatOpenAI

from . import config


class LLMError(RuntimeError):
    """An error whose message is safe and useful to show to the user."""


def _chat_model(name: str) -> ChatOpenAI:
    return ChatOpenAI(model=name, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL,
                      timeout=config.LLM_TIMEOUT, max_retries=config.LLM_RETRIES,
                      max_completion_tokens=config.MAX_OUTPUT_TOKENS)


def get_llm():
    """Primary model, with the fallback models tried in order if it is missing / unsupported / failing."""
    main = _chat_model(config.OPENAI_MODEL)
    fallbacks = [_chat_model(m) for m in config.OPENAI_FALLBACK_MODELS if m != config.OPENAI_MODEL]
    if not fallbacks:
        return main
    return main.with_fallbacks(fallbacks, exceptions_to_handle=(
        openai.NotFoundError, openai.BadRequestError, openai.InternalServerError,
        openai.APITimeoutError, openai.APIConnectionError))


def _translate(exc: Exception) -> LLMError:
    code = getattr(exc, "code", "") or ""
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("OpenAI rejected the API key (401). Check OPENAI_API_KEY.")
    if code == "insufficient_quota":
        return LLMError("OpenAI says your account has no credit. Add billing credit at platform.openai.com.")
    if isinstance(exc, openai.RateLimitError):
        return LLMError("OpenAI rate limit reached. Wait a moment and try again.")
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return LLMError(f"Network error while calling OpenAI: {exc}")
    if isinstance(exc, openai.APIStatusError):
        return LLMError(f"OpenAI error {exc.status_code}: {getattr(exc, 'message', exc)}")
    return LLMError(f"All models failed. Last error: {exc}")


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content or []).strip()


def run(prompt, variables: dict) -> dict:
    """Run `prompt | llm` once and return text plus exact token usage."""
    if not config.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    chain = prompt | get_llm()
    try:
        msg = chain.invoke(variables)
    except LLMError:
        raise
    except Exception as exc:
        raise _translate(exc) from exc
    text = _text(msg.content)
    if not text:
        raise LLMError("The model returned an empty answer (output limit reached while thinking). "
                       "Raise MAX_OUTPUT_TOKENS in .env.")
    usage = getattr(msg, "usage_metadata", None) or {}
    meta = getattr(msg, "response_metadata", None) or {}
    return {"text": text, "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
            "model": meta.get("model_name") or config.OPENAI_MODEL}

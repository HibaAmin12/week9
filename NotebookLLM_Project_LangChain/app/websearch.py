"""Optional web search, used only when the question is not answered by the user's sources.

Off by default. Turn it on in .env with:
    WEB_SEARCH_PROVIDER=brave   (or tavily)
    WEB_SEARCH_API_KEY=<your key>

search(query) returns [{"title", "url", "snippet"}, ...]. Any failure raises WebSearchError, and the caller
then answers from general knowledge instead, so a search outage never breaks a chat.
"""
import re

import requests

from . import config


class WebSearchError(RuntimeError):
    pass


def enabled() -> bool:
    return config.WEB_SEARCH_PROVIDER in ("brave", "tavily") and bool(config.WEB_SEARCH_API_KEY)


def _clean(text: str) -> str:
    """Brave wraps matched words in <strong> tags; strip any HTML and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def _brave(query: str, k: int) -> list:
    r = requests.get("https://api.search.brave.com/res/v1/web/search",
                     headers={"X-Subscription-Token": config.WEB_SEARCH_API_KEY, "Accept": "application/json"},
                     params={"q": query, "count": k}, timeout=config.WEB_SEARCH_TIMEOUT)
    r.raise_for_status()
    items = (r.json().get("web") or {}).get("results") or []
    return [{"title": _clean(i.get("title")), "url": i.get("url", ""), "snippet": _clean(i.get("description"))}
            for i in items]


def _tavily(query: str, k: int) -> list:
    r = requests.post("https://api.tavily.com/search",
                      json={"api_key": config.WEB_SEARCH_API_KEY, "query": query, "max_results": k},
                      timeout=config.WEB_SEARCH_TIMEOUT)
    r.raise_for_status()
    items = r.json().get("results") or []
    return [{"title": _clean(i.get("title")), "url": i.get("url", ""), "snippet": _clean(i.get("content"))}
            for i in items]


def search(query: str, k: int = None) -> list:
    if not enabled():
        return []
    k = k or config.WEB_SEARCH_RESULTS
    try:
        fn = _brave if config.WEB_SEARCH_PROVIDER == "brave" else _tavily
        results = [x for x in fn(query, k) if x["url"].startswith(("http://", "https://")) and x["snippet"]]
    except (requests.RequestException, ValueError, AttributeError) as exc:
        raise WebSearchError(f"web search failed: {exc}") from exc
    return results[:k]

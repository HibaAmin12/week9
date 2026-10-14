"""Input guardrails: keyword check (English, Roman Urdu, Urdu) + OpenAI Moderation API.

check(message) returns None when the message is fine, or
{"category": "self_harm" | "harmful", "answer": "<fixed safe reply>"} when it must be blocked.
The message text is never logged, only the category and which layer fired.
"""
import logging
import re

import requests

from . import config

log = logging.getLogger("guardrails")

# ---- Layer 1: keywords (personal-intent phrases, to avoid blocking questions about documents) ----
_SELF_HARM_PATTERNS = [
    # English
    r"\b(i|i'm|im|me|myself)\b.{0,40}\bsu+i*c+i*d+(e|al)?\b",   # also catches typos like "sucide"
    r"\bkill\s+(my\s*self|myself)\b",
    r"\bend\s+(my\s+life|it\s+all)\b",
    r"\b(want|wanna|going|plan(ning)?)\s+to\s+die\b",
    r"\b(don'?t|do not)\s+want\s+to\s+(live|be\s+alive)\b",
    r"\b(hurt|harm|cut)\s+(my\s*self|myself)\b",
    r"\bself[\s-]?harm\b",
    # Roman Urdu
    r"\bkhud\s*kushi\b",
    r"\bkhud\s+ko\s+(khatam|maar|mar)",
    r"\bmar(na|ne)\s+(chahta|chahti|chahta\s+hun|chahti\s+hun|ka\s+dil)",
    r"\bmar\s+jana\s+(chahta|chahti)",
    r"\bjeena\s+nahi\s+chahta|\bjeena\s+nahi\s+chahti",
    r"\bzindagi\s+khatam\s+(kar|karna)",
    r"\bapni\s+jaan\s+(le|de)",
    # Urdu script
    r"خودکشی",
    r"مرنا\s+چاہت[ایی]",
    r"خود\s+کو\s+(ختم|مار)",
    r"جینا\s+نہیں\s+چاہت[ایی]",
    r"اپنی\s+جان\s+(لے|دے)",
]
_SELF_HARM_RE = re.compile("|".join(_SELF_HARM_PATTERNS), re.I | re.S)

# ---- Fixed replies ----
SELF_HARM_REPLY = (
    "I'm really sorry you're going through this. You don't have to face it alone, and your life matters.\n\n"
    "If you might act on these thoughts, or you are in danger right now, please call your local emergency number "
    "or go to the nearest emergency department. If you can, tell someone you trust (a family member or a friend) "
    "that you are not okay and ask them to stay with you.\n\n"
    f"{config.HELPLINE_TEXT}\n\n"
    "I'm not able to help with this through your documents, but I'm here to listen if you want to tell me what's going on."
)
HARMFUL_REPLY = ("I can't help with that request. If you have a question about your documents, "
                 "I'm happy to help with that.")


def _keyword_hit(message: str) -> bool:
    return bool(_SELF_HARM_RE.search(message))


def _moderate(message: str):
    """Return the moderation result dict, or None if the API is unavailable (fail-safe)."""
    if not config.OPENAI_API_KEY:
        return None
    try:
        r = requests.post(f"{config.OPENAI_BASE_URL}/moderations",
                          headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                          json={"model": config.MODERATION_MODEL, "input": message},
                          timeout=10)
        if r.status_code != 200:
            log.warning("moderation API status %s", r.status_code)
            return None
        return r.json()["results"][0]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        log.warning("moderation API failed: %s", exc)
        return None


def check(message: str):
    if not config.GUARDRAILS_ENABLED:
        return None

    # Layer 1: keywords (fast, free, works for Roman Urdu / Urdu)
    if _keyword_hit(message):
        log.warning("guardrail fired: category=self_harm layer=keyword")
        return {"category": "self_harm", "answer": SELF_HARM_REPLY}

    # Layer 2: OpenAI Moderation API
    res = _moderate(message)
    if res and res.get("flagged"):
        cats = res.get("categories", {})
        if any(v for k, v in cats.items() if k.startswith("self-harm")):
            log.warning("guardrail fired: category=self_harm layer=moderation")
            return {"category": "self_harm", "answer": SELF_HARM_REPLY}
        log.warning("guardrail fired: category=harmful layer=moderation flags=%s",
                    [k for k, v in cats.items() if v])
        return {"category": "harmful", "answer": HARMFUL_REPLY}

    return None
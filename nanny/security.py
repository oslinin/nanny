"""Security guardrail for the ClassifierAgent's chat-input entry point.

Chat text is the one place in Nanny where arbitrary external text reaches an
LLM. This module screens it *before* the model call for the two most direct
risks to a personal, data-sensitive assistant: prompt injection attempting to
override the agent's behavior, and secret-looking strings that should never
be sent to a third-party model or written to the local activity log.

Kept as a small, independently testable module — not because a baby tracker
is a high-value attack target, but because "keeping user data secure" is
exactly what a Concierge Agent is expected to demonstrate, and a guardrail
buried inline in a callback is much harder to verify than one that lives on
its own.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all|the)?\s*(previous|prior|above)\s*instructions",
        r"disregard (all|the)?\s*(previous|prior|above)",
        r"you are now\b",
        r"new instructions\s*:",
        r"reveal (your|the) (system|hidden)?\s*(prompt|instructions)",
        r"act as (if|though)\b",
        r"\bjailbreak\b",
    )
]

_SECRET_PATTERNS = [
    re.compile(p)
    for p in (
        r"AIza[0-9A-Za-z_\-]{35}",  # Google API key
        r"sk-[A-Za-z0-9]{20,}",  # OpenAI-style secret key
        r"ya29\.[0-9A-Za-z_\-]+",  # Google OAuth access token
        r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*\S+",
    )
]


def screen_text(text: str) -> str | None:
    """Returns a rejection reason if `text` looks unsafe to forward, else None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return "message looks like a prompt-injection attempt and was blocked"
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return "message appears to contain a secret or API key and was blocked"
    return None

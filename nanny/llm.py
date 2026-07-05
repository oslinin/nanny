"""Offline heuristic helpers used as agent fallbacks.

The real extraction/summarisation logic lives in ``nanny/agents.py`` as real
``google.adk.agents.LlmAgent`` nodes. This module holds the small,
dependency-free helpers those agents fall back to (via a
``before_model_callback``) when no ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is
configured — the expected state for this local, non-deployed app run
offline — so the app remains fully exercisable without network access or
credentials. The fallback is intentionally simple and is never confused for
the real model: callers can check ``used_llm_extraction`` / \
``used_llm_response`` in the API response to see which path served a request.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

from .activity import ActivityError, BabyActivity


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


_TYPE_KEYWORDS = {
    "poop": ("poop", "pooped", "poo", "dirty diaper"),
    "wet": ("wet diaper", "wet", "pee", "peed"),
    "bottle": ("bottle",),
    "milk": ("breastmilk", "breast milk", "nursed", "nursing", "milk"),
    "solids": ("solids", "puree", "food", "ate", "sweet potato", "banana"),
}

_UNIT_FOR_TYPE = {
    "poop": "count",
    "wet": "count",
    "bottle": "oz",
    "milk": "oz",
    "solids": "grams",
}

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(oz|ounce|g|gram|grams)?", re.IGNORECASE)
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)


def _extract_heuristic(text: str, *, now_iso: str) -> BabyActivity:
    """A small, transparent rule-based fallback — not a substitute for the LLM.

    Used only so the app remains runnable end-to-end without network access
    or an API key configured.
    """
    lowered = text.lower()

    activity_type: str | None = None
    for candidate, keywords in _TYPE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            activity_type = candidate
            break
    if activity_type is None:
        raise ActivityError(
            f"could not determine an activity type from the message: {text!r}"
        )

    unit = _UNIT_FOR_TYPE[activity_type]
    quantity = 1.0
    # Strip any time-of-day phrase first so its leading number ("3 PM") is
    # never mistaken for a quantity.
    time_match = _TIME_RE.search(lowered)
    quantity_search_space = (
        lowered[: time_match.start()] + lowered[time_match.end() :]
        if time_match
        else lowered
    )
    num_match = _NUMBER_RE.search(quantity_search_space)
    if num_match and num_match.group(1):
        quantity = float(num_match.group(1))
        matched_unit = (num_match.group(2) or "").lower()
        if matched_unit.startswith("oz") or matched_unit.startswith("ounce"):
            unit = "oz"
        elif matched_unit.startswith("g"):
            unit = "grams"

    timestamp = _resolve_timestamp(lowered, now_iso)

    return BabyActivity(
        timestamp=timestamp,
        activity_type=activity_type,
        quantity=quantity,
        unit=unit,
        notes="",
    )


def _resolve_timestamp(lowered: str, now_iso: str) -> str:
    now = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    match = _TIME_RE.search(lowered)
    if not match:
        return now_iso
    hour = int(match.group(1)) % 12
    minute = int(match.group(2) or 0)
    if match.group(3).lower() == "pm":
        hour += 12
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate > now + timedelta(minutes=1):
        candidate -= timedelta(days=1)
    return candidate.isoformat()


def _synthesize_template(save_result: dict) -> str:
    saved = save_result["saved"]
    qty = saved["quantity"]
    unit = saved["unit"]
    activity_type = saved["activity_type"]
    today_total = save_result["today_total"]
    today_unit = save_result["today_unit"]
    qty_str = _fmt_num(qty)
    total_str = _fmt_num(today_total)
    return (
        f"Got it! Logged {qty_str}{_unit_suffix(unit)} of {activity_type}. "
        f"Total for today is now {total_str}{_unit_suffix(today_unit)}."
    )


def _fmt_num(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


def _unit_suffix(unit: str) -> str:
    return {"oz": "oz", "grams": "g", "count": ""}.get(unit, f" {unit}")

"""LLM-backed nodes: entity extraction and natural-language summarisation.

Both calls go through ``google.genai`` (the SDK underneath ``google.adk``'s
own ``Gemini`` model wrapper), using constrained JSON schema output for
extraction so the model cannot hand the deterministic storage layer anything
but a well-typed record.

When no ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is configured — the expected
state for this local, non-deployed app run offline — both functions fall back
to a small deterministic heuristic so the app remains fully exercisable
without network access or credentials. The fallback is intentionally simple
and is never confused for the real model: callers can check ``used_llm`` on
the result.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from .activity import KNOWN_ACTIVITY_TYPES, KNOWN_UNITS, ActivityError, BabyActivity

_MODEL_NAME = os.environ.get("NANNY_GEMINI_MODEL", "gemini-flash-latest")


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


class _ExtractedActivity(BaseModel):
    """Constrained JSON schema handed to the model as ``response_schema``."""

    activity_type: str = Field(description=f"One of: {', '.join(KNOWN_ACTIVITY_TYPES)}")
    quantity: float = Field(description="Numeric amount; use 1 for simple counts.")
    unit: str = Field(description=f"One of: {', '.join(KNOWN_UNITS)}")
    timestamp: str = Field(
        description="Absolute ISO-8601 timestamp, resolved from any relative time phrase."
    )
    notes: str = Field(
        default="", description="Any extra descriptive detail, else empty string."
    )


@dataclass
class ExtractionResult:
    activity: BabyActivity
    used_llm: bool


@dataclass
class ResponseResult:
    text: str
    used_llm: bool


_EXTRACTION_SYSTEM_INSTRUCTION = """\
You extract structured infant-care activity records from free text for a
baby activity tracker. Always resolve relative or partial time phrases
("3 PM", "just now", "an hour ago") into an absolute ISO-8601 timestamp using
the provided current time as reference. activity_type must be exactly one of:
{types}. unit must be exactly one of: {units} (oz for milk/bottle volumes,
grams for solids, count for diaper/poop events). Never invent data that
is not present or implied in the text.
""".format(types=", ".join(KNOWN_ACTIVITY_TYPES), units=", ".join(KNOWN_UNITS))


async def extract_activity(text: str, *, now_iso: str) -> ExtractionResult:
    """Turn free-form natural language into a validated BabyActivity."""
    if _has_api_key():
        try:
            return await _extract_via_gemini(text, now_iso=now_iso)
        except Exception:
            pass
    return ExtractionResult(
        activity=_extract_heuristic(text, now_iso=now_iso), used_llm=False
    )


async def _extract_via_gemini(text: str, *, now_iso: str) -> ExtractionResult:
    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = f"Current time (ISO-8601): {now_iso}\n\nMessage: {text}"
    response = await client.aio.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_EXTRACTION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_ExtractedActivity,
        ),
    )
    parsed: _ExtractedActivity = response.parsed
    activity = BabyActivity(
        timestamp=parsed.timestamp,
        activity_type=parsed.activity_type,
        quantity=parsed.quantity,
        unit=parsed.unit,
        notes=parsed.notes,
    )
    return ExtractionResult(activity=activity, used_llm=True)


async def synthesize_response(save_result: dict) -> ResponseResult:
    """Craft a friendly confirmation from SaveActivityNode's transaction metadata."""
    if _has_api_key():
        try:
            return await _synthesize_via_gemini(save_result)
        except Exception:
            pass
    return ResponseResult(text=_synthesize_template(save_result), used_llm=False)


async def _synthesize_via_gemini(save_result: dict) -> ResponseResult:
    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = (
        "Write ONE short, warm, single-sentence confirmation for a parent "
        "using a baby tracker app, based on this saved record and running "
        f"totals (JSON): {save_result}"
    )
    response = await client.aio.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are Nanny, a friendly baby-tracking assistant. Reply with"
                " exactly one short sentence confirming what was logged and the"
                " running total for that activity type today. No preamble."
            ),
        ),
    )
    return ResponseResult(text=(response.text or "").strip(), used_llm=True)


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


# ---------------------------------------------------------------------------
# Offline heuristic fallback (no network / no API key)
# ---------------------------------------------------------------------------

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

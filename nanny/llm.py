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
from collections import defaultdict
from datetime import datetime, timedelta

from .activity import ActivityError, BabyActivity


def _has_api_key() -> bool:
    """True when an AI-Studio (Gemini Developer API) key is configured."""
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _use_vertex() -> bool:
    """True when google-genai is pointed at the Vertex AI backend.

    On Vertex the model is reached through the service account (ADC), not an
    API key — this is how it's authenticated when deployed to Agent Runtime
    (the runtime sets ``GOOGLE_GENAI_USE_VERTEXAI`` and the project for us). We
    mirror google-genai's own switch so the offline gate below doesn't mistake
    "no API key" for "no model" in that environment.
    """
    val = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
    return val in ("1", "true", "yes") and bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))


def _model_available() -> bool:
    """True when a real Gemini call can be made — via either backend.

    The agents' offline fallbacks gate on this: without it (i.e. neither an API
    key nor the Vertex backend configured) they serve a deterministic heuristic
    instead of calling a model. Keying only on the API key would wrongly force
    the offline path on a Vertex deployment, where there is no key by design.
    """
    return _has_api_key() or _use_vertex()


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


def build_insights_context(activities: list[dict], *, now_iso: str) -> dict:
    """Reduce the raw activity log to a compact, model-friendly summary.

    Deterministic aggregation only — per-type counts and totals for today and
    all-time, plus how many distinct days are logged — so the InsightsAgent
    reasons over a small structured summary rather than the raw log, and so the
    offline fallback below has something concrete to ground a reply in without
    any model call.
    """
    today = now_iso[:10]
    per_type_today: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "total": 0.0, "unit": ""}
    )
    per_type_all: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "total": 0.0, "unit": ""}
    )
    days: set[str] = set()
    for a in activities:
        atype = a.get("activity_type", "")
        try:
            qty = float(a.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        unit = a.get("unit", "")
        day = str(a.get("timestamp", ""))[:10]
        if day:
            days.add(day)
        all_bucket = per_type_all[atype]
        all_bucket["count"] += 1
        all_bucket["total"] += qty
        all_bucket["unit"] = unit
        if today and day == today:
            today_bucket = per_type_today[atype]
            today_bucket["count"] += 1
            today_bucket["total"] += qty
            today_bucket["unit"] = unit
    return {
        "total_records": len(activities),
        "days_logged": len(days),
        "today": today,
        "per_type_today": {k: dict(v) for k, v in per_type_today.items()},
        "per_type_all_time": {k: dict(v) for k, v in per_type_all.items()},
    }


def _summarize_insights(context: dict, question: str) -> str:
    """Deterministic, no-LLM insights reply used when no API key is configured.

    Grounds the response in the actual logged summary and stays explicitly
    non-diagnostic, mirroring the InsightsAgent's live instruction so the
    offline path is a faithful (if plainer) stand-in rather than a different
    contract.
    """
    total = context.get("total_records", 0)
    if not total:
        base = (
            "There's nothing logged yet, so there are no patterns to look at. "
            "Log a few feeds or diaper changes and check back."
        )
    else:
        today = context.get("per_type_today", {})
        parts = [
            f"{b['count']} {atype} ({_fmt_num(b['total'])}{_unit_suffix(b['unit'])})"
            if b["unit"] != "count"
            else f"{b['count']} {atype}"
            for atype, b in sorted(today.items())
        ]
        today_line = ", ".join(parts) if parts else "nothing logged yet today"
        base = (
            f"Across {total} logged record(s) over "
            f"{context.get('days_logged', 0)} day(s), today so far: {today_line}."
        )
    disclaimer = (
        " This is a plain summary of what you logged, not medical advice — for "
        "anything that concerns you, discuss the patterns with your pediatrician."
    )
    # Lead with who this is about, so even the no-LLM fallback reflects the
    # baby's age from the Baby tab rather than reading generically.
    baby = context.get("baby") or {}
    age_label = (baby.get("age") or {}).get("label")
    lead = f"For {baby.get('name') or 'Baby'} ({age_label}): " if age_label else ""
    if question and question.strip():
        return (
            lead + "I can't reach the research tools right now, so here's what "
            "your own log shows. " + base + disclaimer
        )
    return lead + base + disclaimer

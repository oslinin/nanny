"""Per-client baby profile (age, measurements) used to ground insights.

Mirrors ``nanny/sources.py``'s per-client file resolution: each client id gets
its own JSON file under ``data/<client_id>.profile.json``. The profile is
pre-populated with neutral default values (a ~6-month-old near the WHO median)
so a brand-new visitor's evidence-based answers are already grounded in a
plausible age and size; the parent edits them in the Baby tab.

Schema (all keys always present after ``get_profile``):

{
  "name": "Baby",
  "sex": "unspecified",        # "unspecified" | "female" | "male"
  "birthdate": "2026-01-07",   # ISO date (YYYY-MM-DD); drives the derived age
  "weight_kg": 7.5,
  "height_cm": 67.0
}

``snapshot`` returns those stored fields plus a derived ``age`` block (days,
weeks, months, and a friendly label), computed from ``birthdate`` against the
turn's ``now_iso`` — that snapshot is what both the ``/api/profile`` response
and the InsightsAgent's search context carry.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Same per-visitor data directory as nanny/stores.py and nanny/sources.py.
_DATA_DIR = Path(os.environ.get("NANNY_DATA_DIR", "./data")).resolve()

_ALLOWED_SEX = ("unspecified", "female", "male")

# Sanity bounds on parent-supplied measurements — a guard against nonsense
# input, not a clinical range.
_MAX_WEIGHT_KG = 50.0
_MAX_HEIGHT_CM = 200.0
_MAX_NAME_LEN = 64


def _sources_path(client_id: str) -> Path:
    # Same validation as _CLIENT_ID_RE in server.py.
    if (
        not all(c.isalnum() or c in ("-", "_") for c in client_id)
        or len(client_id) > 64
    ):
        client_id = "default"
    return _DATA_DIR / f"{client_id}.profile.json"


def _default_profile() -> dict[str, Any]:
    return {
        "name": "Baby",
        "sex": "unspecified",
        # A static default birthdate: the derived age simply tracks real time
        # from this point, so a fresh profile reads as a plausible infant age
        # rather than "0 days" until the parent sets the real date.
        "birthdate": "2026-01-07",
        "weight_kg": 7.5,
        "height_cm": 67.0,
    }


def get_profile(client_id: str) -> dict[str, Any]:
    """Reads the stored profile, filling in defaults for any missing key."""
    path = _sources_path(client_id)
    defaults = _default_profile()
    if not path.exists():
        return defaults
    try:
        data = json.loads(path.read_text())
    except Exception:
        return defaults
    if not isinstance(data, dict):
        return defaults
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data


def _write_profile(client_id: str, profile: dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _sources_path(client_id).write_text(json.dumps(profile, indent=2))


def _coerce_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Validates a partial update and returns only the recognised, clean keys.

    Raises ``ValueError`` (which the API maps to 400) on a malformed value so a
    bad birthdate or negative weight never lands in the stored profile.
    """
    clean: dict[str, Any] = {}

    if "name" in updates:
        name = str(updates["name"]).strip()[:_MAX_NAME_LEN]
        clean["name"] = name or "Baby"

    if "sex" in updates:
        sex = str(updates["sex"]).strip().lower()
        if sex not in _ALLOWED_SEX:
            raise ValueError(f"sex must be one of {list(_ALLOWED_SEX)}")
        clean["sex"] = sex

    if "birthdate" in updates:
        raw = str(updates["birthdate"]).strip()
        try:
            parsed = date.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError("birthdate must be an ISO date (YYYY-MM-DD)") from exc
        clean["birthdate"] = parsed.isoformat()

    for key, cap in (("weight_kg", _MAX_WEIGHT_KG), ("height_cm", _MAX_HEIGHT_CM)):
        if key in updates:
            try:
                value = float(updates[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a number") from exc
            if not (0 < value <= cap):
                raise ValueError(f"{key} must be between 0 and {cap:g}")
            clean[key] = round(value, 2)

    return clean


def set_profile(client_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merges a validated partial update into the stored profile and saves it."""
    profile = get_profile(client_id)
    profile.update(_coerce_updates(updates))
    _write_profile(client_id, profile)
    return profile


def derive_age(birthdate: str, now_iso: str) -> dict[str, Any]:
    """Derives an age block from ``birthdate`` as of ``now_iso``.

    Returns ``{}`` if either date can't be parsed. Months are calendar months
    (not days/30), so "3 months" lines up with how growth charts are read.
    """
    try:
        bd = date.fromisoformat(str(birthdate)[:10])
    except ValueError:
        return {}
    try:
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00")).date()
    except ValueError:
        return {}

    days = (now - bd).days
    if days < 0:
        days = 0
    weeks = days // 7
    months = (now.year - bd.year) * 12 + (now.month - bd.month)
    if now.day < bd.day:
        months -= 1
    months = max(months, 0)
    return {
        "days": days,
        "weeks": weeks,
        "months": months,
        "label": _age_label(days, weeks, months),
    }


def _age_label(days: int, weeks: int, months: int) -> str:
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''} old"
    if months < 3:
        return f"{weeks} weeks old"
    if months < 24:
        return f"{months} months old"
    years = months // 12
    rem = months % 12
    if rem:
        return f"{years} year{'s' if years != 1 else ''}, {rem} months old"
    return f"{years} year{'s' if years != 1 else ''} old"


def snapshot(client_id: str, *, now_iso: str) -> dict[str, Any]:
    """The profile plus a derived ``age`` block — the shape carried by both the
    ``/api/profile`` response and the InsightsAgent's grounding context."""
    profile = get_profile(client_id)
    snap: dict[str, Any] = {
        "name": profile["name"],
        "sex": profile["sex"],
        "birthdate": profile["birthdate"],
        "weight_kg": profile["weight_kg"],
        "height_cm": profile["height_cm"],
    }
    age = derive_age(profile["birthdate"], now_iso)
    if age:
        snap["age"] = age
    return snap

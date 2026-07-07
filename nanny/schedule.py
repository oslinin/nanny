"""Per-client care schedule for the SitterAgent.

A parent sets a daily schedule for their baby (feeds, naps, solids …) and the
SitterAgent turns it into a list of timed reminders it surfaces to the human
sitter through the day. This module is the deterministic side of that feature:

- ``parse_schedule`` — a small, dependency-free parser that turns the free-form
  schedule text into ``[{"time": "HH:MM", "text": ...}]``. It is both the
  offline fallback for the SitterAgent (when no model backend is configured,
  exactly like ``nanny/llm.py`` is for the other agents) and the safety net the
  deterministic save node falls back to if a live model returns nothing usable.
- persistence — mirrors ``nanny/sources.py``: each client id gets its own JSON
  file under ``data/<client_id>.schedule.json``. A fresh client with no file
  yet gets the built-in :data:`SEED_SCHEDULE_TEXT` (the schedule the feature is
  seeded with), so the reminders are visible out of the box.
- ``current_reminder`` / ``next_reminder`` / ``format_reminder`` — the small
  time helpers the SitterAgent and the ``/api/schedule`` frontend poll use to
  decide which instruction the sitter should see right now.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .activity import _parse_iso

# Same per-visitor data directory as nanny/stores.py and nanny/sources.py.
_DATA_DIR = Path(os.environ.get("NANNY_DATA_DIR", "./data")).resolve()

# The schedule the SitterAgent is seeded with — a fresh client sees these
# reminders before ever setting their own. Kept as the raw text the parent
# would type so the same ``parse_schedule`` path produces the default
# reminders (single source of truth). Continuation lines are indented on
# purpose: the parser attaches an indented line to the reminder above it.
SEED_SCHEDULE_TEXT = """\
Schedule

Instructions:

AM

9: 7 oz milk + vit. D or vit D and probiotics
10:30: nap, wake, potty, cloth diaper
11:00 solids
   (cook or reheat on stove, no microwave):
   1-2 ingredient calorie dense
Read, water, walker, play, baby carrier

PM
1: 7 oz milk
2: nap, wake, potty, cloth diaper
3: solids: one ingredient vegetables
Read, water, walker, play, baby carrier
"""

# A leading clock token: "9", "10:30", "11:00" — the hour, an optional ":MM",
# then a word boundary so "1-2 ingredient" (a continuation, not a time) is not
# mistaken for "1:00".
_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?\b")

# Lines that are section headers / titles rather than reminders.
_HEADER_WORDS = {"SCHEDULE", "INSTRUCTIONS"}


def parse_schedule(text: str) -> list[dict[str, str]]:
    """Parse free-form schedule text into ``[{"time": "HH:MM", "text": ...}]``.

    Deterministic and offline — the SitterAgent's fallback when no model is
    configured. Understands the seed's layout:

    - ``AM`` / ``PM`` header lines set the meridiem for the times below them.
    - A top-level line starting with a clock token ("9:", "10:30:", "11:00")
      begins a reminder; the rest of the line is its text.
    - An *indented* line (or a parenthetical continuation) is appended to the
      reminder above it, so multi-line entries stay together.
    - A top-level line with no clock token (e.g. "Read, water, walker …") is an
      ambient, untimed note and is skipped.

    Also accepts semicolon-separated one-liners (e.g. pasted into the single
    line chat box): "AM; 9: 7oz milk; 10:30: nap; PM; 1: 7oz milk".
    """
    reminders: list[dict[str, str]] = []
    meridiem: str | None = None

    for raw in text.splitlines():
        # A semicolon one-liner is treated as several top-level segments; a
        # normal line keeps its indentation so continuations can be detected.
        if ";" in raw:
            units = [(part.strip(), 0) for part in raw.split(";")]
        else:
            units = [(raw.strip(), len(raw) - len(raw.lstrip()))]

        for stripped, indent in units:
            if not stripped:
                continue
            head = stripped.upper().rstrip(":")
            if head == "AM":
                meridiem = "am"
                continue
            if head == "PM":
                meridiem = "pm"
                continue
            if head in _HEADER_WORDS:
                continue

            match = _TIME_RE.match(stripped)
            if match and indent == 0:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
                rest = re.sub(r"^[\s:.\-]+", "", stripped[match.end() :]).strip()
                if meridiem == "pm" and hour != 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    continue
                reminders.append({"time": f"{hour:02d}:{minute:02d}", "text": rest})
            elif indent > 0 and reminders:
                reminders[-1]["text"] = (reminders[-1]["text"] + " " + stripped).strip()
            # A top-level, non-time line is an untimed note — skipped.

    return reminders


def normalize_reminders(items: Any) -> list[dict[str, str]]:
    """Coerce a model's structured reminder list into the strict stored shape.

    Drops anything without a valid ``HH:MM`` time so a malformed model response
    can never persist a garbage reminder — the same "a node, not the LLM,
    guards storage" stance the rest of the graph takes.
    """
    out: list[dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        time = str(item.get("time", "")).strip()
        match = re.match(r"^(\d{1,2}):(\d{2})$", time)
        if not match:
            continue
        hour, minute = int(match.group(1)), int(match.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            continue
        out.append(
            {
                "time": f"{hour:02d}:{minute:02d}",
                "text": str(item.get("text", "")).strip(),
            }
        )
    return out


def _schedule_path(client_id: str) -> Path:
    # Same validation as _CLIENT_ID_RE in server.py / sources.py.
    if (
        not all(c.isalnum() or c in ("-", "_") for c in client_id)
        or len(client_id) > 64
    ):
        client_id = "default"
    return _DATA_DIR / f"{client_id}.schedule.json"


def get_schedule(client_id: str) -> dict[str, Any]:
    """Return this client's stored schedule, or the seeded default if none.

    Shape: ``{"raw": <text>, "reminders": [{"time", "text"}, ...]}``. A fresh
    client falls back to :data:`SEED_SCHEDULE_TEXT` so the sitter reminders are
    populated before the parent ever sets their own.
    """
    path = _schedule_path(client_id)
    if not path.exists():
        return {
            "raw": SEED_SCHEDULE_TEXT,
            "reminders": parse_schedule(SEED_SCHEDULE_TEXT),
        }
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {
            "raw": SEED_SCHEDULE_TEXT,
            "reminders": parse_schedule(SEED_SCHEDULE_TEXT),
        }
    reminders = normalize_reminders(data.get("reminders"))
    return {"raw": data.get("raw", ""), "reminders": reminders}


def save_schedule(
    client_id: str, raw: str, reminders: list[dict[str, str]]
) -> dict[str, Any]:
    """Persist a client's schedule (raw text + parsed reminders)."""
    reminders = normalize_reminders(reminders)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"raw": raw, "reminders": reminders}
    _schedule_path(client_id).write_text(json.dumps(payload, indent=2))
    return payload


def hhmm_from_iso(now_iso: str) -> str:
    """Reduce an ISO-8601 timestamp to a ``HH:MM`` clock string for comparison."""
    try:
        dt = _parse_iso(now_iso)
        return f"{dt.hour:02d}:{dt.minute:02d}"
    except (ValueError, TypeError):
        return "00:00"


def next_reminder(
    reminders: list[dict[str, str]], now_hhmm: str
) -> dict[str, str] | None:
    """The earliest reminder at or after ``now_hhmm`` — what's coming up next."""
    upcoming = [r for r in reminders if r.get("time", "") >= now_hhmm]
    if upcoming:
        return min(upcoming, key=lambda r: r["time"])
    return None


def current_reminder(
    reminders: list[dict[str, str]], now_hhmm: str
) -> dict[str, str] | None:
    """The instruction in effect right now — the latest reminder already due.

    Before the first reminder of the day, falls back to that first upcoming
    one so the sitter always has something relevant to see.
    """
    if not reminders:
        return None
    due = [r for r in reminders if r.get("time", "") <= now_hhmm]
    if due:
        return max(due, key=lambda r: r["time"])
    return min(reminders, key=lambda r: r["time"])


def _to_12h(hhmm: str) -> str:
    try:
        hour, minute = int(hhmm[:2]), int(hhmm[3:5])
    except (ValueError, IndexError):
        return hhmm
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def format_reminder(reminder: dict[str, str] | None, *, upcoming: bool = True) -> str:
    """A single warm sentence telling the sitter what to do."""
    if not reminder:
        return "All of today's scheduled care is done — nice work! \U0001f389"
    lead = "Next up" if upcoming else "Right now"
    return f"{lead} at {_to_12h(reminder['time'])}: {reminder['text']}"

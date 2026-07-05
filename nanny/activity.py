"""Core shared data schema.

A single strict structure — :class:`BabyActivity` — is threaded through every
node of the workflow graph via the ADK session state. Because each node
populates or forwards this exact shape, no node (deterministic or generative)
can invent fields or drift the schema, which is what guards against
hallucinated database writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum

# Canonical vocabularies. The first four activity types come straight from the
# PRD; "wet" is added to support the "Log 1 Wet Diaper" quick-tap control.
KNOWN_ACTIVITY_TYPES = ("poop", "milk", "bottle", "solids", "wet")
KNOWN_UNITS = ("oz", "count", "grams")


class ActivityError(ValueError):
    """Raised when a BabyActivity fails validation."""


@dataclass
class BabyActivity:
    """The strict structured record appended to the datastore.

    Fields mirror the PRD schema exactly.
    """

    timestamp: str = ""  # ISO-8601 format string
    activity_type: str = ""  # "poop" | "milk" | "bottle" | "solids" | "wet"
    quantity: float = 0.0  # e.g., 4.5, 1.0
    unit: str = ""  # "oz" | "count" | "grams"
    notes: str = ""  # e.g., "very runny", "sweet potato puree"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BabyActivity:
        """Build a BabyActivity from an arbitrary dict, ignoring unknown keys.

        Ignoring unknown keys keeps the strict shape even if an upstream
        producer (or a future LLM) tacks on extra fields.
        """
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean = {k: data[k] for k in data if k in allowed}
        if "quantity" in clean and clean["quantity"] is not None:
            clean["quantity"] = float(clean["quantity"])
        # Coerce str-like values (e.g. an Enum member from a structured LLM
        # response) to plain str so formatting/serialization never leaks a
        # wrapper type's repr instead of its value. An Enum's own str() may
        # render as "ClassName.member" rather than the value, so unwrap
        # `.value` explicitly rather than relying on str().
        for key in ("activity_type", "unit", "timestamp", "notes"):
            if key in clean and clean[key] is not None:
                value = clean[key]
                clean[key] = str(value.value) if isinstance(value, Enum) else str(value)
        return cls(**clean)

    def validate(self) -> BabyActivity:
        """Validate the record before it may reach deterministic storage.

        Returns self on success so calls can be chained; raises
        :class:`ActivityError` otherwise.
        """
        if not self.timestamp or not str(self.timestamp).strip():
            raise ActivityError("timestamp is required")
        try:
            _parse_iso(self.timestamp)
        except ValueError as exc:
            raise ActivityError(
                f"timestamp {self.timestamp!r} is not ISO-8601: {exc}"
            ) from exc
        if self.activity_type not in KNOWN_ACTIVITY_TYPES:
            raise ActivityError(
                f"unknown activity_type {self.activity_type!r} "
                f"(want one of {list(KNOWN_ACTIVITY_TYPES)})"
            )
        if self.unit not in KNOWN_UNITS:
            raise ActivityError(
                f"unknown unit {self.unit!r} (want one of {list(KNOWN_UNITS)})"
            )
        if self.quantity < 0:
            raise ActivityError(f"quantity may not be negative, got {self.quantity}")
        return self


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing 'Z' (UTC)."""
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)

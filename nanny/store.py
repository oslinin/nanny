"""Deterministic datastore backing the SaveActivityNode.

Implements the PRD's "sequential JSON logfile" option as JSON Lines: one JSON
object per line, append-only, crash-friendly, and free of any native/database
build dependency so the app runs anywhere Python runs. All writes are
serialized with a lock so concurrent HTTP requests can't interleave lines.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

from .activity import BabyActivity, _parse_iso


@dataclass
class SaveResult:
    """Operational execution status returned by a write — the transaction
    metadata the ResponderNode turns into a friendly summary."""

    ok: bool
    saved: BabyActivity
    today_count: int = 0  # count of same-type records logged today
    today_total: float = 0.0  # summed quantity of same-type records today
    today_unit: str = ""  # unit the total is expressed in
    total_records: int = 0  # total rows in the log

    def to_dict(self) -> dict:
        d = {
            "ok": self.ok,
            "saved": self.saved.to_dict(),
            "today_count": self.today_count,
            "today_total": self.today_total,
            "today_unit": self.today_unit,
            "total_records": self.total_records,
        }
        return d


class Store:
    """A concurrency-safe JSON-lines activity log."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def append(self, activity: BabyActivity) -> SaveResult:
        """Validate, append one record, and return running totals for the day."""
        activity.validate()
        line = json.dumps(activity.to_dict(), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            rows = self._read_all_locked()

        day = _day_key(activity.timestamp)
        result = SaveResult(
            ok=True,
            saved=activity,
            today_unit=activity.unit,
            total_records=len(rows),
        )
        for row in rows:
            if (
                row.activity_type == activity.activity_type
                and _day_key(row.timestamp) == day
            ):
                result.today_count += 1
                if row.unit == activity.unit:
                    result.today_total += row.quantity
        return result

    def all(self) -> list[BabyActivity]:
        """Return every record in the log, oldest first."""
        with self._lock:
            return self._read_all_locked()

    def _read_all_locked(self) -> list[BabyActivity]:
        if not os.path.exists(self.path):
            return []
        out: list[BabyActivity] = []
        with open(self.path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                out.append(BabyActivity.from_dict(json.loads(raw)))
        return out


def _day_key(ts: str) -> str:
    """Reduce an ISO-8601 timestamp to a YYYY-MM-DD grouping key."""
    try:
        return _parse_iso(ts).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ts[:10] if isinstance(ts, str) and len(ts) >= 10 else str(ts)

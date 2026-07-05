import os

import pytest

from nanny.activity import ActivityError, BabyActivity
from nanny.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "sub" / "log.jsonl"))


def test_append_creates_parent_dir_and_persists(store):
    a = BabyActivity(
        timestamp="2026-07-05T10:00:00+00:00",
        activity_type="bottle",
        quantity=4.0,
        unit="oz",
        notes="",
    )
    result = store.append(a)
    assert result.ok
    assert result.today_count == 1
    assert result.today_total == 4.0
    assert result.total_records == 1
    assert os.path.exists(store.path)


def test_append_rejects_invalid_activity(store):
    bad = BabyActivity(timestamp="", activity_type="bottle", quantity=1.0, unit="oz")
    with pytest.raises(ActivityError):
        store.append(bad)
    assert store.all() == []


def test_running_totals_group_by_day_and_type(store):
    store.append(
        BabyActivity(
            timestamp="2026-07-05T08:00:00+00:00",
            activity_type="bottle",
            quantity=4.0,
            unit="oz",
        )
    )
    store.append(
        BabyActivity(
            timestamp="2026-07-05T12:00:00+00:00",
            activity_type="bottle",
            quantity=2.0,
            unit="oz",
        )
    )
    result = store.append(
        BabyActivity(
            timestamp="2026-07-04T12:00:00+00:00",  # different day
            activity_type="bottle",
            quantity=9.0,
            unit="oz",
        )
    )
    # Third record is on a different day, so it should not roll into "today"'s total
    # relative to the first two entries' day.
    same_day_result = store.append(
        BabyActivity(
            timestamp="2026-07-05T14:00:00+00:00",
            activity_type="bottle",
            quantity=1.0,
            unit="oz",
        )
    )
    assert same_day_result.today_count == 3
    assert same_day_result.today_total == 7.0
    assert result.today_count == 1
    assert len(store.all()) == 4


def test_all_returns_records_in_append_order(store):
    for i in range(3):
        store.append(
            BabyActivity(
                timestamp=f"2026-07-05T0{i}:00:00+00:00",
                activity_type="poop",
                quantity=1.0,
                unit="count",
            )
        )
    rows = store.all()
    assert [r.timestamp for r in rows] == [
        "2026-07-05T00:00:00+00:00",
        "2026-07-05T01:00:00+00:00",
        "2026-07-05T02:00:00+00:00",
    ]

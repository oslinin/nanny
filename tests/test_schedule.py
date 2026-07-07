"""Tests for nanny/schedule.py — the SitterAgent's deterministic side.

Two concerns: the free-form schedule parser (also the agent's offline fallback)
and the per-client persistence (mirrors nanny/sources.py's per-client JSON
files, so — like tests/test_sources.py — every test reloads the module after
setting NANNY_DATA_DIR, since ``_DATA_DIR`` is resolved once at import time).
"""

import importlib

import nanny.schedule as schedule_mod


def _reload(tmp_path, monkeypatch):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    importlib.reload(schedule_mod)
    return schedule_mod


def test_parse_seed_schedule_resolves_am_pm_sections():
    reminders = schedule_mod.parse_schedule(schedule_mod.SEED_SCHEDULE_TEXT)
    times = [r["time"] for r in reminders]
    # AM section stays as-is; PM section is offset by 12 hours.
    assert times == ["09:00", "10:30", "11:00", "13:00", "14:00", "15:00"]
    first = reminders[0]
    assert first["text"] == "7 oz milk + vit. D or vit D and probiotics"
    # Indented continuation lines are folded into the reminder above them, so a
    # line like "1-2 ingredient calorie dense" is text — never a 1:00 reminder.
    solids = reminders[2]
    assert solids["time"] == "11:00"
    assert "1-2 ingredient calorie dense" in solids["text"]


def test_parse_semicolon_one_liner():
    text = "Instructions:; AM; 9: 7 oz milk; 10:30: nap; PM; 1: 7 oz milk; 3: solids"
    reminders = schedule_mod.parse_schedule(text)
    assert [r["time"] for r in reminders] == ["09:00", "10:30", "13:00", "15:00"]


def test_parse_skips_untimed_ambient_lines():
    reminders = schedule_mod.parse_schedule(
        "AM\n9: milk\nRead, water, walker, play, baby carrier\n"
    )
    assert [r["time"] for r in reminders] == ["09:00"]


def test_normalize_reminders_drops_malformed_times():
    items = [
        {"time": "09:00", "text": "milk"},
        {"time": "9:00", "text": "coerced"},  # single-digit hour normalizes
        {"time": "nope", "text": "dropped"},
        {"time": "25:00", "text": "dropped"},
        {"text": "no time dropped"},
        "not-a-dict",
    ]
    out = schedule_mod.normalize_reminders(items)
    assert out == [
        {"time": "09:00", "text": "milk"},
        {"time": "09:00", "text": "coerced"},
    ]


def test_get_schedule_defaults_to_seed_when_no_file(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    sched = mod.get_schedule("alice")
    # A fresh client sees the seeded default so reminders are visible out of box.
    assert sched["raw"] == mod.SEED_SCHEDULE_TEXT
    assert len(sched["reminders"]) == 6


def test_save_and_get_roundtrip_is_per_client(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.save_schedule(
        "alice", "Instructions:\n9: milk", [{"time": "09:00", "text": "milk"}]
    )
    assert mod.get_schedule("alice")["reminders"] == [{"time": "09:00", "text": "milk"}]
    # A second client is unaffected — still the seeded default.
    assert len(mod.get_schedule("bob")["reminders"]) == 6


def test_corrupt_file_falls_back_to_seed(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    (tmp_path / "alice.schedule.json").write_text("not json")
    assert mod.get_schedule("alice")["raw"] == mod.SEED_SCHEDULE_TEXT


def test_next_and_current_reminder():
    reminders = schedule_mod.parse_schedule(schedule_mod.SEED_SCHEDULE_TEXT)
    assert schedule_mod.next_reminder(reminders, "10:00")["time"] == "10:30"
    assert schedule_mod.current_reminder(reminders, "10:00")["time"] == "09:00"
    assert schedule_mod.current_reminder(reminders, "13:30")["time"] == "13:00"
    # Nothing left today.
    assert schedule_mod.next_reminder(reminders, "23:00") is None
    # Before the day starts, current falls back to the first upcoming one.
    assert schedule_mod.current_reminder(reminders, "06:00")["time"] == "09:00"


def test_format_reminder():
    assert "10:30 AM" in schedule_mod.format_reminder({"time": "10:30", "text": "nap"})
    assert "1:00 PM" in schedule_mod.format_reminder({"time": "13:00", "text": "milk"})
    assert "done" in schedule_mod.format_reminder(None)


def test_client_id_sanitization_falls_back_to_default(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.save_schedule("../../etc/passwd", "raw", [{"time": "09:00", "text": "x"}])
    # A malformed id collapses to the same file as the "default" id.
    assert mod.get_schedule("default")["reminders"] == [{"time": "09:00", "text": "x"}]

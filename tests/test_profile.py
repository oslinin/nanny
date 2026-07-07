"""Tests for nanny/profile.py — the per-client baby profile behind the Baby tab.

Pure filesystem + date logic (mirrors nanny/sources.py's per-client JSON file
resolution) — no Vertex credentials needed. ``_DATA_DIR`` is resolved once at
import time, so every test reloads the module after setting NANNY_DATA_DIR,
exactly like tests/test_sources.py.
"""

import importlib

import pytest

import nanny.profile as profile_mod


def _reload(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(profile_mod)
    return profile_mod


def test_defaults_when_no_file_exists(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    prefs = mod.get_profile("alice")
    assert prefs["name"] == "Baby"
    assert prefs["sex"] == "unspecified"
    assert prefs["birthdate"] == "2026-01-07"
    assert prefs["weight_kg"] == 7.5
    assert prefs["height_cm"] == 67.0


def test_set_profile_persists_and_is_per_client(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_profile("alice", {"name": "Mia", "weight_kg": 8.2})
    stored = mod.get_profile("alice")
    assert stored["name"] == "Mia"
    assert stored["weight_kg"] == 8.2
    # Untouched fields keep their defaults; a second client is unaffected.
    assert stored["sex"] == "unspecified"
    assert mod.get_profile("bob")["name"] == "Baby"


def test_partial_update_only_changes_supplied_keys(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_profile("alice", {"name": "Mia"})
    mod.set_profile("alice", {"height_cm": 70})
    stored = mod.get_profile("alice")
    assert stored["name"] == "Mia"  # not clobbered by the second update
    assert stored["height_cm"] == 70.0


def test_blank_name_falls_back_to_baby(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_profile("alice", {"name": "   "})
    assert mod.get_profile("alice")["name"] == "Baby"


@pytest.mark.parametrize(
    "updates",
    [
        {"sex": "other"},
        {"birthdate": "not-a-date"},
        {"weight_kg": -1},
        {"weight_kg": 999},
        {"height_cm": 0},
        {"weight_kg": "heavy"},
    ],
)
def test_invalid_values_raise(tmp_path, monkeypatch, updates):
    mod = _reload(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        mod.set_profile("alice", updates)


def test_corrupt_file_falls_back_to_defaults(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    (tmp_path / "alice.profile.json").write_text("not json")
    assert mod.get_profile("alice")["name"] == "Baby"


def test_derive_age_months_and_label(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    age = mod.derive_age("2026-01-07", "2026-07-07T12:00:00+00:00")
    assert age["months"] == 6
    assert age["days"] == 181
    assert age["label"] == "6 months old"


def test_derive_age_uses_weeks_when_very_young(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    age = mod.derive_age("2026-06-20", "2026-07-07T00:00:00+00:00")
    # 17 days -> under 3 months, reported in weeks.
    assert age["weeks"] == 2
    assert age["label"] == "2 weeks old"


def test_derive_age_days_for_newborn(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    age = mod.derive_age("2026-07-01", "2026-07-07T00:00:00+00:00")
    assert age["label"] == "6 days old"


def test_derive_age_years(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    age = mod.derive_age("2024-01-07", "2026-07-07T00:00:00+00:00")
    assert age["label"] == "2 years, 6 months old"


def test_derive_age_unparseable_returns_empty(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    assert mod.derive_age("garbage", "2026-07-07T00:00:00+00:00") == {}
    assert mod.derive_age("2026-01-07", "garbage") == {}


def test_snapshot_includes_derived_age(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_profile("alice", {"name": "Mia", "birthdate": "2026-01-07"})
    snap = mod.snapshot("alice", now_iso="2026-07-07T12:00:00+00:00")
    assert snap["name"] == "Mia"
    assert snap["age"]["label"] == "6 months old"


def test_client_id_sanitization_falls_back_to_default(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_profile("../../etc/passwd", {"name": "Evil"})
    assert mod.get_profile("default")["name"] == "Evil"


# --- GET/POST /api/profile endpoints --------------------------------------


def _reload_server(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import nanny.stores as stores_module

    importlib.reload(stores_module)
    importlib.reload(profile_mod)

    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


def test_profile_get_returns_seeded_defaults(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)

    resp = client.get("/api/profile", headers={"X-Nanny-Client-Id": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Baby"
    assert body["weight_kg"] == 7.5
    assert "age" in body  # derived from the default birthdate


def test_profile_post_persists_and_derives_age(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    resp = client.post(
        "/api/profile",
        json={"name": "Mia", "birthdate": "2026-01-07", "weight_kg": 8.0},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Mia"
    assert body["weight_kg"] == 8.0
    assert body["age"]["months"] >= 0

    # Persisted: a plain GET reflects it.
    assert client.get("/api/profile", headers=headers).json()["name"] == "Mia"


def test_profile_post_rejects_bad_values(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    resp = client.post(
        "/api/profile",
        json={"birthdate": "nope"},
        headers={"X-Nanny-Client-Id": "alice"},
    )
    assert resp.status_code == 400


def test_profile_post_empty_body_is_rejected(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    resp = client.post("/api/profile", json={}, headers={"X-Nanny-Client-Id": "alice"})
    assert resp.status_code == 400


def test_profile_post_requires_token_when_configured(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _reload_server(tmp_path, monkeypatch, NANNY_API_TOKEN="secret123")
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    unauth = client.post("/api/profile", json={"name": "Mia"}, headers=headers)
    assert unauth.status_code == 401

    ok = client.post(
        "/api/profile",
        json={"name": "Mia"},
        headers={**headers, "X-Nanny-Token": "secret123"},
    )
    assert ok.status_code == 200

    # GET stays open (read-only), like /api/corpus and /api/sources.
    assert client.get("/api/profile", headers=headers).status_code == 200


def test_profile_is_isolated_per_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)

    client.post(
        "/api/profile", json={"name": "Mia"}, headers={"X-Nanny-Client-Id": "alice"}
    )
    bob = client.get("/api/profile", headers={"X-Nanny-Client-Id": "bob"})
    assert bob.json()["name"] == "Baby"

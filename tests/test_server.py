"""Tests for the opt-in access-token guard and CORS support in server.py.

Both are configured via env vars read at module import time, so each test
that needs a specific configuration reloads the module after setting env
vars via monkeypatch — a plain import wouldn't pick up per-test overrides.
"""

import importlib

from fastapi.testclient import TestClient

QUICK_TAP_BODY = {
    "activity_type": "bottle",
    "quantity": 4,
    "unit": "oz",
    "notes": "",
}


def _reload_server(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("NANNY_DATA_PATH", str(tmp_path / "log.jsonl"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


def test_endpoints_work_without_token_by_default(tmp_path, monkeypatch):
    server_module = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    resp = client.post("/api/quick-tap", json=QUICK_TAP_BODY)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_token_required_when_configured(tmp_path, monkeypatch):
    server_module = _reload_server(tmp_path, monkeypatch, NANNY_API_TOKEN="secret123")
    client = TestClient(server_module.app)

    unauthenticated = client.post("/api/quick-tap", json=QUICK_TAP_BODY)
    assert unauthenticated.status_code == 401

    wrong_token = client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Token": "wrong"}
    )
    assert wrong_token.status_code == 401

    authenticated = client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Token": "secret123"}
    )
    assert authenticated.status_code == 200


def test_history_endpoint_is_not_gated_by_token(tmp_path, monkeypatch):
    # /api/history is read-only; only the two mutating endpoints require the token.
    server_module = _reload_server(tmp_path, monkeypatch, NANNY_API_TOKEN="secret123")
    client = TestClient(server_module.app)
    resp = client.get("/api/history")
    assert resp.status_code == 200


def test_cors_headers_present_for_allowed_origin(tmp_path, monkeypatch):
    server_module = _reload_server(
        tmp_path, monkeypatch, NANNY_ALLOWED_ORIGINS="https://example.github.io"
    )
    client = TestClient(server_module.app)
    resp = client.post(
        "/api/quick-tap",
        json=QUICK_TAP_BODY,
        headers={"Origin": "https://example.github.io"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "https://example.github.io"


def test_cors_headers_absent_by_default(tmp_path, monkeypatch):
    server_module = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    resp = client.post(
        "/api/quick-tap",
        json=QUICK_TAP_BODY,
        headers={"Origin": "https://example.github.io"},
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers

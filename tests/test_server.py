"""Tests for the opt-in access-token guard, CORS, and per-client isolation
in server.py.

All three are configured via env vars (or, for the client id, a header) read
at module import / request time, so each test that needs a specific
configuration reloads the module after setting env vars via monkeypatch — a
plain import wouldn't pick up per-test overrides.
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
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
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


def test_different_clients_get_isolated_history(tmp_path, monkeypatch):
    server_module = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Client-Id": "alice"}
    )
    client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Client-Id": "bob"}
    )
    client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Client-Id": "bob"}
    )

    alice_history = client.get(
        "/api/history", headers={"X-Nanny-Client-Id": "alice"}
    ).json()
    bob_history = client.get(
        "/api/history", headers={"X-Nanny-Client-Id": "bob"}
    ).json()

    assert len(alice_history) == 1
    assert len(bob_history) == 2


def test_missing_client_id_falls_back_to_default(tmp_path, monkeypatch):
    server_module = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    client.post("/api/quick-tap", json=QUICK_TAP_BODY)
    resp = client.get("/api/history")
    assert len(resp.json()) == 1
    assert (tmp_path / "default.jsonl").exists()


def test_malformed_client_id_falls_back_to_default(tmp_path, monkeypatch):
    server_module = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    # Path-traversal-shaped header must not escape the data dir or crash.
    client.post(
        "/api/quick-tap",
        json=QUICK_TAP_BODY,
        headers={"X-Nanny-Client-Id": "../../etc/passwd"},
    )
    assert (tmp_path / "default.jsonl").exists()
    assert not (tmp_path.parent.parent / "etc" / "passwd.jsonl").exists()


def test_database_session_service_used_when_db_url_configured(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.db"
    server_module = _reload_server(
        tmp_path, monkeypatch, NANNY_DB_URL=f"sqlite+aiosqlite:///{db_path}"
    )
    assert type(server_module.session_service).__name__ == "DatabaseSessionService"

    client = TestClient(server_module.app)
    resp = client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Client-Id": "carol"}
    )
    assert resp.status_code == 200
    assert db_path.exists()


def test_session_state_survives_reload_with_same_db_url(tmp_path, monkeypatch):
    # Simulates a Cloud Run restart: a brand-new server_module (and therefore
    # a brand-new DatabaseSessionService instance) pointed at the same
    # database must still see state written before the "restart".
    db_path = tmp_path / "sessions.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    server_module = _reload_server(tmp_path, monkeypatch, NANNY_DB_URL=db_url)
    client = TestClient(server_module.app)
    client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Client-Id": "dave"}
    )

    # Reload again with the identical env — a fresh session_service/runner,
    # standing in for a fresh container instance after a restart.
    restarted_module = _reload_server(tmp_path, monkeypatch, NANNY_DB_URL=db_url)
    restarted_client = TestClient(restarted_module.app)
    resp = restarted_client.post(
        "/api/chat",
        json={"text": "he pooped at 3"},
        headers={"X-Nanny-Client-Id": "dave"},
    )
    assert resp.status_code == 200
    # If the prior session's state hadn't survived, last_status would be
    # missing entirely rather than resolving through the graph normally.
    assert resp.json()["ok"] is True

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

    # nanny.stores caches DATA_DIR and per-client Store instances at module
    # level; reloading only nanny.server would leave both stale (pointing at
    # a previous test's tmp_path), so reload stores first — server.py's
    # `from .stores import get_store` then picks up the fresh function.
    import nanny.stores as stores_module

    importlib.reload(stores_module)

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


def test_history_endpoint_is_gated_by_token_when_configured(tmp_path, monkeypatch):
    # A visitor's activity log is private; when a token is configured, reads of
    # it require the token too — not just the mutating endpoints.
    server_module = _reload_server(tmp_path, monkeypatch, NANNY_API_TOKEN="secret123")
    client = TestClient(server_module.app)

    assert client.get("/api/history").status_code == 401
    assert (
        client.get("/api/history", headers={"X-Nanny-Token": "wrong"}).status_code
        == 401
    )
    assert (
        client.get("/api/history", headers={"X-Nanny-Token": "secret123"}).status_code
        == 200
    )


def test_history_endpoint_open_without_token_by_default(tmp_path, monkeypatch):
    # With no token configured (local dev), history stays open as before.
    server_module = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    assert client.get("/api/history").status_code == 200


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


class _FakeAgentEngine:
    """Stands in for a deployed Agent Runtime resource / local AdkApp.

    Mirrors just enough of the real ``async_stream_query`` /
    ``async_get_session`` / ``async_create_session`` contract (confirmed by
    reading vertexai's AdkApp/AgentEngine source) to exercise
    ``_AgentRuntimeBackend`` without any real GCP credentials or network
    access — neither of which this sandbox has.
    """

    def __init__(self):
        self.sessions: dict[str, dict] = {}
        self.calls: list[tuple] = []

    async def async_create_session(self, *, user_id, session_id, state=None, **kwargs):
        self.sessions[session_id] = {"id": session_id, "state": dict(state or {})}
        return self.sessions[session_id]

    async def async_get_session(self, *, user_id, session_id, **kwargs):
        return self.sessions.get(session_id)

    async def async_stream_query(self, *, message, user_id, session_id, **kwargs):
        state_delta = kwargs.get("state_delta") or {}
        self.calls.append((message, user_id, session_id, state_delta))
        session = self.sessions.setdefault(session_id, {"id": session_id, "state": {}})
        session["state"].update(state_delta)
        session["state"]["last_status"] = "ok"
        session["state"]["response_text"] = "fake agent runtime response"
        return
        yield  # pragma: no cover - makes this an async generator


def test_agent_runtime_backend_selected_when_resource_name_configured(
    tmp_path, monkeypatch
):
    fake_engine = _FakeAgentEngine()
    monkeypatch.setattr("vertexai.agent_engines.get", lambda resource_name: fake_engine)
    server_module = _reload_server(
        tmp_path,
        monkeypatch,
        GOOGLE_CLOUD_PROJECT="fake-project",
        NANNY_AGENT_ENGINE_RESOURCE_NAME="projects/fake/locations/us-east1/reasoningEngines/123",
    )
    assert type(server_module.backend).__name__ == "_AgentRuntimeBackend"

    client = TestClient(server_module.app)
    resp = client.post(
        "/api/quick-tap", json=QUICK_TAP_BODY, headers={"X-Nanny-Client-Id": "erin"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["response_text"] == "fake agent runtime response"

    # The dashboard passed the request through to the (fake) deployed agent
    # rather than running the graph in-process.
    assert len(fake_engine.calls) == 1
    _message, user_id, session_id, state_delta = fake_engine.calls[0]
    assert user_id == session_id == "erin"
    assert state_delta["input_mode"] == "quick_tap"
    assert state_delta["client_id"] == "erin"


def test_local_backend_selected_by_default(tmp_path, monkeypatch):
    server_module = _reload_server(tmp_path, monkeypatch)
    assert type(server_module.backend).__name__ == "_LocalRunnerBackend"


def _reload_server_with_schedule(tmp_path, monkeypatch, **env):
    """Like _reload_server but also reloads nanny.schedule so its per-client
    data dir points at tmp_path (the SitterAgent path persists through it)."""
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import nanny.schedule as schedule_module
    import nanny.stores as stores_module

    importlib.reload(stores_module)
    importlib.reload(schedule_module)
    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


def test_schedule_get_returns_seeded_default_for_fresh_client(tmp_path, monkeypatch):
    server_module = _reload_server_with_schedule(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    resp = client.get("/api/schedule", headers={"X-Nanny-Client-Id": "fresh"})
    assert resp.status_code == 200
    # A brand-new client sees the seeded reminders out of the box.
    assert len(resp.json()["reminders"]) == 6


def test_chat_instructions_sets_schedule_then_get_reflects_it(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    server_module = _reload_server_with_schedule(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    set_resp = client.post(
        "/api/chat",
        json={"text": "Instructions:; AM; 9: milk; 10:30: nap; PM; 1: milk"},
        headers=headers,
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["ok"] is True

    got = client.get("/api/schedule", headers=headers).json()
    assert [r["time"] for r in got["reminders"]] == ["09:00", "10:30", "13:00"]


def test_chat_nanny_prompt_returns_next_instruction(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    server_module = _reload_server_with_schedule(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    headers = {"X-Nanny-Client-Id": "bob"}

    resp = client.post("/api/chat", json={"text": "nanny"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # The bare "nanny" prompt surfaces an instruction, never logs an activity.
    assert body["activity"] is None
    assert body["response_text"]

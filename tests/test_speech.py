"""Tests for the optional server-side speech-to-text endpoints.

The real path calls Google Cloud Speech-to-Text, which needs credentials and
the ``google-cloud-speech`` package (neither present here). So these tests
mock ``nanny.speech.transcribe`` — the same boundary the endpoint calls — to
cover all of *our* logic (the enabled/disabled gate, token gating, the audio
flow) without importing the Google client. The mic button's default engine is
the browser's Web Speech API, which has no server component to test.
"""

import importlib

from fastapi.testclient import TestClient

import nanny.speech as speech_mod


def _reload_server(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("NANNY_API_TOKEN", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import nanny.stores as stores_module

    importlib.reload(stores_module)
    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


def test_stt_enabled_flag(monkeypatch):
    monkeypatch.delenv("NANNY_STT_ENABLED", raising=False)
    assert speech_mod.stt_enabled() is False
    monkeypatch.setenv("NANNY_STT_ENABLED", "true")
    assert speech_mod.stt_enabled() is True


def test_transcribe_status_reflects_flag(monkeypatch, tmp_path):
    server = _reload_server(monkeypatch, tmp_path)
    client = TestClient(server.app)
    assert client.get("/api/transcribe").json() == {"enabled": False}

    server = _reload_server(monkeypatch, tmp_path, NANNY_STT_ENABLED="true")
    client = TestClient(server.app)
    assert client.get("/api/transcribe").json() == {"enabled": True}


def test_transcribe_disabled_by_default(monkeypatch, tmp_path):
    server = _reload_server(monkeypatch, tmp_path)
    client = TestClient(server.app)
    resp = client.post(
        "/api/transcribe",
        files={"file": ("audio.webm", b"fake-audio", "audio/webm")},
    )
    assert resp.status_code == 501


def test_transcribe_returns_text_when_enabled(monkeypatch, tmp_path):
    server = _reload_server(monkeypatch, tmp_path, NANNY_STT_ENABLED="true")
    # Mock the Google call at the module boundary the endpoint uses.
    captured = {}

    def fake_transcribe(audio, mime="audio/webm"):
        captured["audio"] = audio
        captured["mime"] = mime
        return "he pooped at three"

    monkeypatch.setattr(server.speech, "transcribe", fake_transcribe)

    client = TestClient(server.app)
    resp = client.post(
        "/api/transcribe",
        files={"file": ("audio.webm", b"opus-bytes", "audio/webm")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"transcript": "he pooped at three"}
    assert captured["audio"] == b"opus-bytes"


def test_transcribe_rejects_empty_audio(monkeypatch, tmp_path):
    server = _reload_server(monkeypatch, tmp_path, NANNY_STT_ENABLED="true")
    monkeypatch.setattr(server.speech, "transcribe", lambda *a, **k: "x")
    client = TestClient(server.app)
    resp = client.post(
        "/api/transcribe", files={"file": ("audio.webm", b"", "audio/webm")}
    )
    assert resp.status_code == 400


def test_transcribe_token_gated_when_configured(monkeypatch, tmp_path):
    server = _reload_server(
        monkeypatch, tmp_path, NANNY_STT_ENABLED="true", NANNY_API_TOKEN="secret123"
    )
    monkeypatch.setattr(server.speech, "transcribe", lambda *a, **k: "x")
    client = TestClient(server.app)

    files = {"file": ("audio.webm", b"opus", "audio/webm")}
    assert client.post("/api/transcribe", files=files).status_code == 401
    ok = client.post(
        "/api/transcribe", files=files, headers={"X-Nanny-Token": "secret123"}
    )
    assert ok.status_code == 200

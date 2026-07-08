"""Tests for GET/POST /api/sources — the Corpus tab's backing endpoint.

Runs against the real local corpus backend (nanny/corpus.py), so the
unicef-document-row and personal-upload-merge logic is exercised end to end
without any cloud.
"""

import importlib

from fastapi.testclient import TestClient

import nanny.corpus as corpus_mod
import nanny.sources as sources_mod


def _reload_server(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    # google_search's availability now piggybacks on the model backend
    # (_model_available()) rather than its own env vars — clear these so
    # ambient sandbox state can't leak into a test that didn't ask for it.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    # Corpus uses the local backend in tests (no Gemini key / File Search).
    monkeypatch.setenv("NANNY_CORPUS_BACKEND", "local")
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import nanny.stores as stores_module

    importlib.reload(stores_module)
    importlib.reload(sources_mod)

    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


def test_sources_report_nothing_available_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("NANNY_RAG_ENABLED", raising=False)
    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)

    resp = client.get("/api/sources", headers={"X-Nanny-Client-Id": "alice"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["google_search"] == {"available": False, "enabled": True}
    assert body["documents"] == []


def test_google_search_toggle_persists_and_requires_token_when_configured(
    tmp_path, monkeypatch
):
    server = _reload_server(
        tmp_path,
        monkeypatch,
        GEMINI_API_KEY="fake-key",
        NANNY_API_TOKEN="secret123",
    )
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    unauth = client.post("/api/sources", json={"google_search": False}, headers=headers)
    assert unauth.status_code == 401

    resp = client.post(
        "/api/sources",
        json={"google_search": False},
        headers={**headers, "X-Nanny-Token": "secret123"},
    )
    assert resp.status_code == 200
    assert resp.json()["google_search"] == {"available": True, "enabled": False}

    # Persisted: a plain GET reflects it too.
    listing = client.get(
        "/api/sources", headers={**headers, "X-Nanny-Token": "secret123"}
    )
    assert listing.json()["google_search"]["enabled"] is False


def test_unicef_document_row_only_appears_once_the_shared_corpus_is_seeded(
    tmp_path, monkeypatch
):
    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    resp = client.get("/api/sources", headers=headers)
    assert resp.json()["documents"] == []

    corpus_mod.add_file_to_shared_unicef_corpus("shared-guide.txt", b"sleep")
    resp = client.get("/api/sources", headers=headers)
    docs = resp.json()["documents"]
    assert docs == [
        {
            "name": "The Art of Parenting.pdf",
            "source": "unicef",
            "enabled": True,
            "deletable": True,
        }
    ]


def test_removing_the_unicef_document_drops_it_from_this_clients_list_only(
    tmp_path, monkeypatch
):
    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    alice = {"X-Nanny-Client-Id": "alice"}
    bob = {"X-Nanny-Client-Id": "bob"}

    corpus_mod.add_file_to_shared_unicef_corpus("shared-guide.txt", b"sleep")

    off = client.post(
        "/api/sources",
        json={"document": {"source": "unicef", "enabled": False}},
        headers=alice,
    )
    assert off.status_code == 200
    # It's just a default entry in the list: removing it here is scoped to
    # this client — it vanishes entirely (not merely shown unchecked) rather
    # than sticking around, but the shared corpus itself is untouched, so
    # another client still sees it until they remove it too.
    assert off.json()["documents"] == []
    assert client.get("/api/sources", headers=alice).json()["documents"] == []
    assert client.get("/api/sources", headers=bob).json()["documents"] == [
        {
            "name": "The Art of Parenting.pdf",
            "source": "unicef",
            "enabled": True,
            "deletable": True,
        }
    ]


def test_toggling_upload_documents(tmp_path, monkeypatch):
    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    corpus_mod.add_file("alice", "my-notes.txt", b"my own notes")

    off_upload = client.post(
        "/api/sources",
        json={
            "document": {"source": "upload", "name": "my-notes.txt", "enabled": False}
        },
        headers=headers,
    )
    assert off_upload.status_code == 200
    upload_doc = next(
        d for d in off_upload.json()["documents"] if d["source"] == "upload"
    )
    assert upload_doc == {
        "name": "my-notes.txt",
        "source": "upload",
        "enabled": False,
        "deletable": True,
    }


def test_upload_document_update_requires_a_name(tmp_path, monkeypatch):
    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    resp = client.post(
        "/api/sources",
        json={"document": {"source": "upload", "enabled": False}},
        headers={"X-Nanny-Client-Id": "alice"},
    )
    assert resp.status_code == 400


def test_invalid_body_is_rejected(tmp_path, monkeypatch):
    server = _reload_server(tmp_path, monkeypatch)
    client = TestClient(server.app)
    resp = client.post("/api/sources", json={}, headers={"X-Nanny-Client-Id": "alice"})
    assert resp.status_code == 400

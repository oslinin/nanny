"""Tests for GET/POST /api/sources — the Corpus tab's backing endpoint.

Mirrors tests/test_corpus.py's approach: a minimal fake ``vertexai.rag``
stands in for Vertex so the unicef-document-row and personal-upload-merge
logic can be exercised without real GCP credentials.
"""

import importlib
import types

import pytest
from fastapi.testclient import TestClient

import nanny.corpus as corpus_mod
import nanny.sources as sources_mod


class _FakeRag:
    def __init__(self):
        self.corpora: dict[str, types.SimpleNamespace] = {}
        self.files: dict[str, list] = {}
        self._n = 0

    def list_corpora(self):
        return list(self.corpora.values())

    def create_corpus(self, display_name=None, description=None, **kw):
        self._n += 1
        name = f"projects/p/locations/l/ragCorpora/{self._n}"
        c = types.SimpleNamespace(name=name, display_name=display_name)
        self.corpora[name] = c
        self.files[name] = []
        return c

    def upload_file(self, corpus_name=None, path=None, display_name=None, **kw):
        self._n += 1
        f = types.SimpleNamespace(
            name=f"{corpus_name}/ragFiles/{self._n}", display_name=display_name
        )
        self.files[corpus_name] = [*self.files.get(corpus_name, []), f]
        return f

    def list_files(self, corpus_name=None, **kw):
        return list(self.files.get(corpus_name, []))

    def delete_file(self, name=None, corpus_name=None, **kw):
        self.files[corpus_name] = [
            f for f in self.files.get(corpus_name, []) if f.name != name
        ]


@pytest.fixture
def fake_rag(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    monkeypatch.setenv("NANNY_RAG_ENABLED", "true")
    fake = _FakeRag()
    import vertexai

    monkeypatch.setattr(vertexai, "init", lambda **kw: None)
    for attr in (
        "list_corpora",
        "create_corpus",
        "upload_file",
        "list_files",
        "delete_file",
    ):
        monkeypatch.setattr(f"vertexai.rag.{attr}", getattr(fake, attr), raising=False)
    return fake


def _reload_server(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import nanny.stores as stores_module

    importlib.reload(stores_module)
    importlib.reload(sources_mod)

    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


def test_sources_report_nothing_available_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
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
        GOOGLE_CSE_ID="cse-id",
        GOOGLE_CSE_API_KEY="cse-key",
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
    fake_rag, tmp_path, monkeypatch
):
    server = _reload_server(tmp_path, monkeypatch, NANNY_RAG_ENABLED="true")
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    resp = client.get("/api/sources", headers=headers)
    assert resp.json()["documents"] == []

    corpus_mod.add_file_to_shared_unicef_corpus("The Art of Parenting.pdf", b"data")
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
    fake_rag, tmp_path, monkeypatch
):
    server = _reload_server(tmp_path, monkeypatch, NANNY_RAG_ENABLED="true")
    client = TestClient(server.app)
    alice = {"X-Nanny-Client-Id": "alice"}
    bob = {"X-Nanny-Client-Id": "bob"}

    corpus_mod.add_file_to_shared_unicef_corpus("The Art of Parenting.pdf", b"data")

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


def test_toggling_upload_documents(fake_rag, tmp_path, monkeypatch):
    server = _reload_server(tmp_path, monkeypatch, NANNY_RAG_ENABLED="true")
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    corpus_mod.add_file("alice", "my-notes.pdf", b"data")

    off_upload = client.post(
        "/api/sources",
        json={
            "document": {"source": "upload", "name": "my-notes.pdf", "enabled": False}
        },
        headers=headers,
    )
    assert off_upload.status_code == 200
    upload_doc = next(
        d for d in off_upload.json()["documents"] if d["source"] == "upload"
    )
    assert upload_doc == {
        "name": "my-notes.pdf",
        "source": "upload",
        "enabled": False,
        "deletable": True,
    }


def test_upload_document_update_requires_a_name(fake_rag, tmp_path, monkeypatch):
    server = _reload_server(tmp_path, monkeypatch, NANNY_RAG_ENABLED="true")
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

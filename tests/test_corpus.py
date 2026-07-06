"""Tests for the parent-controlled Vertex RAG corpus.

The real feature calls ``vertexai.rag`` against Google Cloud, which needs
credentials this sandbox doesn't have — so, exactly like ``_FakeAgentEngine``
stands in for Agent Runtime in tests/test_server.py, these tests install a fake
``vertexai.rag`` and exercise all of *our* logic (per-client corpus scoping,
the enabled/disabled gate, extension + path-traversal validation, retrieval
scoping) without any real Vertex call. Only Vertex's own behavior is unmocked.
"""

import importlib
import types

import pytest
from fastapi.testclient import TestClient

import nanny.corpus as corpus_mod


class _FakeRag:
    """Minimal stateful stand-in for the ``vertexai.rag`` surface we call."""

    def __init__(self):
        self.corpora: dict[str, types.SimpleNamespace] = {}
        self.files: dict[str, list] = {}
        self.retrieve_calls: list = []
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

    def RagResource(self, rag_corpus=None, rag_file_ids=None):
        return types.SimpleNamespace(rag_corpus=rag_corpus)

    def RagRetrievalConfig(self, top_k=None, **kw):
        return types.SimpleNamespace(top_k=top_k)

    async def async_retrieve_contexts(
        self, text=None, rag_resources=None, rag_retrieval_config=None, **kw
    ):
        self.retrieve_calls.append((text, rag_resources))
        ctx = types.SimpleNamespace(text="a relevant passage from the parent's book")
        return types.SimpleNamespace(contexts=types.SimpleNamespace(contexts=[ctx]))

    def corpus_name_for(self, display_name):
        return next(
            c.name for c in self.corpora.values() if c.display_name == display_name
        )


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
        "RagResource",
        "RagRetrievalConfig",
        "async_retrieve_contexts",
    ):
        monkeypatch.setattr(f"vertexai.rag.{attr}", getattr(fake, attr), raising=False)
    return fake


def _fresh_server(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("NANNY_API_TOKEN", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import nanny.stores as stores_module

    importlib.reload(stores_module)
    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


# --- corpus.py unit level -------------------------------------------------


def test_rag_enabled_flag(monkeypatch):
    monkeypatch.delenv("NANNY_RAG_ENABLED", raising=False)
    assert corpus_mod.rag_enabled() is False
    monkeypatch.setenv("NANNY_RAG_ENABLED", "true")
    assert corpus_mod.rag_enabled() is True


def test_add_list_delete_is_per_client(fake_rag):
    corpus_mod.add_file("alice", "book.pdf", b"%PDF-1.4 data")
    corpus_mod.add_file("bob", "notes.txt", b"hello")

    assert corpus_mod.list_files("alice") == ["book.pdf"]
    assert corpus_mod.list_files("bob") == ["notes.txt"]
    # Each client got their own corpus, keyed by a deterministic display name.
    assert {c.display_name for c in fake_rag.list_corpora()} == {
        "nanny-corpus-alice",
        "nanny-corpus-bob",
    }

    assert corpus_mod.delete_file("alice", "book.pdf") is True
    assert corpus_mod.list_files("alice") == []
    assert corpus_mod.delete_file("alice", "not-there") is False


def test_get_or_create_reuses_existing_corpus(fake_rag):
    first = corpus_mod.get_or_create_corpus("alice")
    second = corpus_mod.get_or_create_corpus("alice")
    assert first == second
    assert len(fake_rag.list_corpora()) == 1


# --- HTTP endpoints -------------------------------------------------------


def test_corpus_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("NANNY_RAG_ENABLED", raising=False)
    server = _fresh_server(monkeypatch, tmp_path)
    client = TestClient(server.app)

    listing = client.get("/api/corpus")
    assert listing.status_code == 200
    assert listing.json() == {"enabled": False, "files": []}

    upload = client.post(
        "/api/corpus", files={"file": ("book.pdf", b"x", "application/pdf")}
    )
    assert upload.status_code == 501


def test_corpus_upload_then_list_when_enabled(fake_rag, monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path, NANNY_RAG_ENABLED="true")
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    up = client.post(
        "/api/corpus",
        files={"file": ("book.pdf", b"%PDF data", "application/pdf")},
        headers=headers,
    )
    assert up.status_code == 200
    assert up.json()["filename"] == "book.pdf"

    listing = client.get("/api/corpus", headers=headers)
    assert listing.json() == {"enabled": True, "files": ["book.pdf"]}


def test_corpus_rejects_unsupported_extension(fake_rag, monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path, NANNY_RAG_ENABLED="true")
    client = TestClient(server.app)
    resp = client.post(
        "/api/corpus",
        files={"file": ("malware.exe", b"x", "application/octet-stream")},
        headers={"X-Nanny-Client-Id": "alice"},
    )
    assert resp.status_code == 400


def test_corpus_upload_strips_path_from_filename(fake_rag, monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path, NANNY_RAG_ENABLED="true")
    client = TestClient(server.app)
    resp = client.post(
        "/api/corpus",
        files={"file": ("../../etc/passwd.txt", b"x", "text/plain")},
        headers={"X-Nanny-Client-Id": "alice"},
    )
    assert resp.status_code == 200
    # Only the basename is kept — no path components reach the stored name.
    assert resp.json()["filename"] == "passwd.txt"


def test_corpus_delete_missing_is_404(fake_rag, monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path, NANNY_RAG_ENABLED="true")
    client = TestClient(server.app)
    resp = client.delete("/api/corpus/nope.pdf", headers={"X-Nanny-Client-Id": "alice"})
    assert resp.status_code == 404


# --- retrieval tool -------------------------------------------------------


async def test_retrieval_tool_scopes_to_the_calling_client(fake_rag):
    corpus_mod.add_file("alice", "book.pdf", b"data")
    from nanny.research import _PerClientRagRetrieval

    tool = _PerClientRagRetrieval()
    ctx = types.SimpleNamespace(state={"client_id": "alice"})
    out = await tool.run_async(
        args={"query": "how much milk per day?"}, tool_context=ctx
    )

    assert "relevant passage" in out
    _text, resources = fake_rag.retrieve_calls[-1]
    assert resources[0].rag_corpus == fake_rag.corpus_name_for("nanny-corpus-alice")


async def test_retrieval_tool_handles_no_corpus(fake_rag):
    from nanny.research import _PerClientRagRetrieval

    tool = _PerClientRagRetrieval()
    ctx = types.SimpleNamespace(state={"client_id": "never-uploaded"})
    out = await tool.run_async(args={"query": "anything"}, tool_context=ctx)
    assert "not uploaded" in out.lower()

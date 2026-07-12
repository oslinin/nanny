"""Tests for the hybrid reference corpus (nanny/corpus.py).

The **local BM25 backend** is the default with no Gemini key, so it's exercised
for real (no mocks) — add/list/delete per client, shared corpus, chunking,
overlap-gated retrieval. The **Gemini File Search backend** can't be reached
from the sandbox, so a fake genai client stands in to prove the dispatch and
grounding-chunk parsing. Backend selection itself (key present/absent, probe
failure) is unit-tested with a fake client too.
"""

import importlib
import types

import pytest
from fastapi.testclient import TestClient

import nanny.corpus as corpus_mod


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch, tmp_path):
    """Default every test to the local backend in a fresh data dir, and reset
    the File Search singletons so probes/clients don't leak across tests."""
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NANNY_CORPUS_BACKEND", "local")
    monkeypatch.setattr(corpus_mod, "_fs_client_singleton", None, raising=False)
    monkeypatch.setattr(corpus_mod, "_fs_probe", None, raising=False)


# --- local backend, unit level --------------------------------------------


def test_rag_always_enabled():
    assert corpus_mod.rag_enabled() is True


def test_add_list_delete_is_per_client():
    corpus_mod.add_file("alice", "book.txt", b"sleep routines for infants")
    corpus_mod.add_file("bob", "notes.md", b"feeding schedule notes")

    assert corpus_mod.list_files("alice") == ["book.txt"]
    assert corpus_mod.list_files("bob") == ["notes.md"]

    assert corpus_mod.delete_file("alice", "book.txt") is True
    assert corpus_mod.list_files("alice") == []
    assert corpus_mod.delete_file("alice", "not-there") is False


def test_resolve_is_none_until_a_file_is_added():
    assert corpus_mod.resolve_corpus_name("alice") is None
    corpus_mod.add_file("alice", "book.txt", b"content about naps")
    assert corpus_mod.resolve_corpus_name("alice") is not None


def test_retrieval_is_scoped_and_overlap_gated():
    corpus_mod.add_file(
        "alice",
        "sleep.txt",
        b"Six month olds typically nap two to three times per day.",
    )
    corpus_mod.add_file(
        "alice",
        "milk.txt",
        b"A six month old drinks about 24 to 32 ounces of milk per day.",
    )
    handle = corpus_mod.resolve_corpus_name("alice")

    milk = corpus_mod.retrieve(handle, "how much milk per day")
    assert any(fn == "milk.txt" for _t, fn in milk)
    # Off-topic query shares no terms -> nothing (not arbitrary passages).
    assert corpus_mod.retrieve(handle, "quantum chromodynamics") == []
    # A different client sees nothing.
    assert corpus_mod.retrieve(corpus_mod.resolve_corpus_name("bob"), "milk") == []


def test_long_text_is_split_into_multiple_chunks():
    big = ("infant sleep guidance " * 400).encode()  # ~9 KB -> many chunks
    corpus_mod.add_file("alice", "guide.txt", big)
    handle = corpus_mod.resolve_corpus_name("alice")
    rows = corpus_mod._read_rows(corpus_mod.Path(handle))
    assert len(rows) > 1
    assert all(r["filename"] == "guide.txt" for r in rows)


def test_pdf_text_is_extracted():
    pytest.importorskip("pypdf")
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    # A blank PDF extracts to no text -> recorded with a placeholder chunk so it
    # still lists (rather than vanishing).
    corpus_mod.add_file("alice", "scan.pdf", buf.getvalue())
    assert corpus_mod.list_files("alice") == ["scan.pdf"]


# --- shared UNICEF corpus --------------------------------------------------


def test_shared_unicef_corpus_starts_unresolved_then_seeds():
    assert corpus_mod.resolve_shared_unicef_corpus() is None
    corpus_mod.add_file_to_shared_unicef_corpus(
        "The Art of Parenting.txt", b"responsive parenting and sleep"
    )
    assert corpus_mod.resolve_shared_unicef_corpus() is not None
    hits = corpus_mod.retrieve(
        corpus_mod.resolve_shared_unicef_corpus(), "parenting sleep"
    )
    assert hits and hits[0][1] == "The Art of Parenting.txt"


def test_shared_corpus_cannot_collide_with_a_client_id():
    corpus_mod.add_file_to_shared_unicef_corpus("guide.txt", b"shared data")
    corpus_mod.add_file("shared_unicef", "notes.txt", b"client data")
    assert corpus_mod.resolve_corpus_name("shared_unicef") != (
        corpus_mod.resolve_shared_unicef_corpus()
    )


# --- backend selection -----------------------------------------------------


def test_backend_is_local_without_a_key(monkeypatch):
    monkeypatch.delenv("NANNY_CORPUS_BACKEND", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert corpus_mod._use_file_search() is False


def test_file_search_eligible_even_with_vertex_model_backend(monkeypatch):
    # File Search uses the Developer API (its client is built vertexai=False), so
    # a Vertex *model* backend doesn't disable it — it just needs a key.
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert corpus_mod._has_dev_key() is True


def test_auto_falls_back_to_local_when_file_search_probe_fails(monkeypatch):
    monkeypatch.setenv("NANNY_CORPUS_BACKEND", "auto")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    class _Boom:
        class file_search_stores:
            @staticmethod
            def list():
                raise RuntimeError("File Search not enabled for this project")

    monkeypatch.setattr(corpus_mod, "_fs_client", lambda: _Boom)
    assert corpus_mod._use_file_search() is False


# --- Gemini File Search backend (fake client) ------------------------------


class _FakeDocs:
    def __init__(self, store):
        self._store = store

    def list(self, *, parent):
        return list(self._store.docs.get(parent, []))

    def delete(self, *, name):
        for store, docs in self._store.docs.items():
            self._store.docs[store] = [d for d in docs if d.name != name]


class _FakeStores:
    def __init__(self, state):
        self._s = state
        self.documents = _FakeDocs(state)

    def list(self):
        return [
            types.SimpleNamespace(name=n, display_name=d)
            for n, d in self._s.stores.items()
        ]

    def create(self, *, config):
        self._s.n += 1
        name = f"fileSearchStores/store-{self._s.n}"
        self._s.stores[name] = config["display_name"]
        self._s.docs[name] = []
        return types.SimpleNamespace(name=name, display_name=config["display_name"])

    def upload_to_file_search_store(self, *, file_search_store_name, file, config):
        self._s.n += 1
        self._s.docs[file_search_store_name].append(
            types.SimpleNamespace(
                name=f"{file_search_store_name}/documents/d{self._s.n}",
                display_name=config["display_name"],
            )
        )


class _FakeModels:
    def __init__(self, state):
        self._s = state

    def generate_content(self, *, model, contents, config):
        store = config.tools[0].file_search.file_search_store_names[0]
        chunks = [
            types.SimpleNamespace(
                retrieved_context=types.SimpleNamespace(
                    text=f"passage from {d.display_name}",
                    document_name=d.display_name,
                    title=d.display_name,
                )
            )
            for d in self._s.docs.get(store, [])
        ]
        return types.SimpleNamespace(
            candidates=[
                types.SimpleNamespace(
                    grounding_metadata=types.SimpleNamespace(grounding_chunks=chunks)
                )
            ]
        )


class _FakeClient:
    def __init__(self):
        self.stores: dict[str, str] = {}
        self.docs: dict[str, list] = {}
        self.n = 0
        self.file_search_stores = _FakeStores(self)
        self.models = _FakeModels(self)


@pytest.fixture
def fake_file_search(monkeypatch):
    monkeypatch.setenv("NANNY_CORPUS_BACKEND", "file_search")
    fake = _FakeClient()
    monkeypatch.setattr(corpus_mod, "_fs_client", lambda: fake)
    return fake


def test_file_search_add_list_delete_and_retrieve(fake_file_search):
    corpus_mod.add_file("alice", "book.pdf", b"%PDF fake")
    assert corpus_mod.list_files("alice") == ["book.pdf"]

    handle = corpus_mod.resolve_corpus_name("alice")
    assert handle is not None
    passages = corpus_mod.retrieve(handle, "anything")
    assert passages == [("passage from book.pdf", "book.pdf")]

    assert corpus_mod.delete_file("alice", "book.pdf") is True
    assert corpus_mod.list_files("alice") == []


def test_file_search_retrieval_errors_degrade_to_empty(fake_file_search, monkeypatch):
    corpus_mod.add_file("alice", "book.pdf", b"data")
    handle = corpus_mod.resolve_corpus_name("alice")

    def _boom(*a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr(fake_file_search.models, "generate_content", _boom)
    # A retrieval failure must never break the agent turn.
    assert corpus_mod.retrieve(handle, "anything") == []


# --- HTTP endpoints (local backend) ---------------------------------------


def _fresh_server(monkeypatch, tmp_path):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NANNY_CORPUS_BACKEND", "local")
    monkeypatch.delenv("NANNY_API_TOKEN", raising=False)
    import nanny.stores as stores_module

    importlib.reload(stores_module)
    import nanny.server as server_module

    importlib.reload(server_module)
    return server_module


def test_corpus_upload_then_list(monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = {"X-Nanny-Client-Id": "alice"}

    up = client.post(
        "/api/corpus",
        files={"file": ("book.txt", b"infant sleep guidance", "text/plain")},
        headers=headers,
    )
    assert up.status_code == 200
    assert up.json()["filename"] == "book.txt"

    listing = client.get("/api/corpus", headers=headers)
    assert listing.json() == {"enabled": True, "files": ["book.txt"]}


def test_corpus_rejects_unsupported_extension(monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path)
    client = TestClient(server.app)
    resp = client.post(
        "/api/corpus",
        files={"file": ("malware.exe", b"x", "application/octet-stream")},
        headers={"X-Nanny-Client-Id": "alice"},
    )
    assert resp.status_code == 400


def test_corpus_upload_strips_path_from_filename(monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path)
    client = TestClient(server.app)
    resp = client.post(
        "/api/corpus",
        files={"file": ("../../etc/passwd.txt", b"x", "text/plain")},
        headers={"X-Nanny-Client-Id": "alice"},
    )
    assert resp.status_code == 200
    assert resp.json()["filename"] == "passwd.txt"


def test_corpus_delete_missing_is_404(monkeypatch, tmp_path):
    server = _fresh_server(monkeypatch, tmp_path)
    client = TestClient(server.app)
    resp = client.delete("/api/corpus/nope.pdf", headers={"X-Nanny-Client-Id": "alice"})
    assert resp.status_code == 404


# --- retrieval tool (research.py, local backend) --------------------------


async def test_retrieval_tool_scopes_to_client_and_honors_toggles():
    corpus_mod.add_file("alice", "keep.txt", b"naps happen twice a day for infants")
    corpus_mod.add_file("alice", "hide.txt", b"infants nap in the afternoon too")
    from nanny.research import _PerClientRagRetrieval

    tool = _PerClientRagRetrieval()
    ctx = types.SimpleNamespace(
        state={
            "client_id": "alice",
            "enabled_sources": {
                "google_search": True,
                "unicef": False,
                "uploads": {"hide.txt": False},
            },
        }
    )
    out = await tool.run_async(args={"query": "when do infants nap"}, tool_context=ctx)
    assert "keep.txt" not in out  # filenames aren't in the passage text
    # The kept file's content is present; the hidden file's is filtered out.
    assert "twice a day" in out
    assert "afternoon" not in out


async def test_retrieval_tool_includes_shared_unicef_when_enabled():
    corpus_mod.add_file_to_shared_unicef_corpus(
        "The Art of Parenting.txt", b"responsive parenting supports infant sleep"
    )
    from nanny.research import _PerClientRagRetrieval

    tool = _PerClientRagRetrieval()
    ctx = types.SimpleNamespace(
        state={
            "client_id": "alice",
            "enabled_sources": {"google_search": True, "unicef": True, "uploads": {}},
        }
    )
    out = await tool.run_async(args={"query": "infant sleep"}, tool_context=ctx)
    assert "responsive parenting" in out


async def test_retrieval_tool_skips_shared_unicef_when_disabled():
    corpus_mod.add_file_to_shared_unicef_corpus("guide.txt", b"infant sleep guidance")
    from nanny.research import _PerClientRagRetrieval

    tool = _PerClientRagRetrieval()
    ctx = types.SimpleNamespace(
        state={
            "client_id": "alice",
            "enabled_sources": {"google_search": True, "unicef": False, "uploads": {}},
        }
    )
    out = await tool.run_async(args={"query": "infant sleep"}, tool_context=ctx)
    assert "no relevant passages" in out.lower()

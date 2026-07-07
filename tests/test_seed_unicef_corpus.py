"""Tests for the one-time operator seeding script (nanny/seed_unicef_corpus.py)."""

import types

import pytest

import nanny.corpus as corpus_mod
import nanny.seed_unicef_corpus as seed_mod


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


@pytest.fixture
def fake_rag(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    monkeypatch.setenv("NANNY_RAG_ENABLED", "true")
    fake = _FakeRag()
    import vertexai

    monkeypatch.setattr(vertexai, "init", lambda **kw: None)
    for attr in ("list_corpora", "create_corpus", "upload_file", "list_files"):
        monkeypatch.setattr(f"vertexai.rag.{attr}", getattr(fake, attr), raising=False)
    return fake


def test_seed_requires_a_path(capsys):
    assert seed_mod.main([]) == 1
    assert "usage" in capsys.readouterr().err.lower()


def test_seed_refuses_when_rag_disabled(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("NANNY_RAG_ENABLED", raising=False)
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF data")
    assert seed_mod.main([str(pdf)]) == 1
    assert "NANNY_RAG_ENABLED" in capsys.readouterr().err


def test_seed_refuses_a_missing_file(fake_rag, tmp_path, capsys):
    missing = tmp_path / "nope.pdf"
    assert seed_mod.main([str(missing)]) == 1
    assert "no such file" in capsys.readouterr().err


def test_seed_uploads_into_the_shared_corpus(fake_rag, tmp_path, capsys):
    pdf = tmp_path / "guide.pdf"
    pdf.write_bytes(b"%PDF data")
    assert seed_mod.main([str(pdf)]) == 0
    assert "seeded" in capsys.readouterr().out.lower()
    name = corpus_mod.resolve_shared_unicef_corpus()
    assert name is not None
    assert [f.display_name for f in fake_rag.list_files(corpus_name=name)] == [
        "guide.pdf"
    ]

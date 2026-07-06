"""Tests for nanny/sources.py — per-client evidence-source preferences.

Pure filesystem logic (mirrors nanny/stores.py's per-client JSON file
resolution) plus one thin layer of availability checks over env vars and
nanny/corpus.py — no Vertex credentials needed except where a fake is
installed (mirroring tests/test_corpus.py's approach).

``_DATA_DIR`` is resolved once at import time (the same pattern as
nanny/stores.py), so every test reloads the module after setting
NANNY_DATA_DIR — exactly like tests/test_research.py reloads nanny.stores.
"""

import importlib

import nanny.sources as sources_mod


def _reload(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(sources_mod)
    return sources_mod


def test_defaults_when_no_file_exists(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    prefs = mod.get_prefs("alice")
    assert prefs == {"google_search": True, "unicef": True, "uploads": {}}


def test_set_google_search_persists(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_google_search("alice", False)
    assert mod.get_prefs("alice")["google_search"] is False
    # A second client's prefs are unaffected.
    assert mod.get_prefs("bob")["google_search"] is True


def test_set_unicef_persists(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_unicef("alice", False)
    assert mod.get_prefs("alice")["unicef"] is False


def test_set_upload_enabled_defaults_missing_files_to_enabled(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_upload_enabled("alice", "notes.pdf", False)
    prefs = mod.get_prefs("alice")
    assert prefs["uploads"] == {"notes.pdf": False}
    # A filename never explicitly set defaults to enabled (fresh uploads work
    # immediately without an extra step).
    assert prefs["uploads"].get("fresh-upload.pdf", True) is True


def test_corrupt_file_falls_back_to_defaults(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    path = tmp_path / "alice.sources.json"
    path.write_text("not json")
    assert mod.get_prefs("alice") == {
        "google_search": True,
        "unicef": True,
        "uploads": {},
    }


def test_availability_google_search_needs_both_env_vars(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("NANNY_RAG_ENABLED", raising=False)
    mod = _reload(tmp_path, monkeypatch)
    assert mod.availability("alice")["google_search"] is False

    mod = _reload(
        tmp_path, monkeypatch, GOOGLE_CSE_ID="cse-id", GOOGLE_CSE_API_KEY="cse-key"
    )
    assert mod.availability("alice")["google_search"] is True


def test_availability_uploads_and_unicef_need_rag_enabled(monkeypatch, tmp_path):
    monkeypatch.delenv("NANNY_RAG_ENABLED", raising=False)
    mod = _reload(tmp_path, monkeypatch)
    avail = mod.availability("alice")
    assert avail["uploads"] is False
    assert avail["unicef"] is False


def test_availability_unicef_also_needs_the_shared_corpus_seeded(monkeypatch, tmp_path):
    mod = _reload(tmp_path, monkeypatch, NANNY_RAG_ENABLED="true")
    monkeypatch.setattr(mod.corpus_mod, "resolve_shared_unicef_corpus", lambda: None)
    avail = mod.availability("alice")
    # uploads only needs RAG on; unicef additionally needs the corpus seeded.
    assert avail["uploads"] is True
    assert avail["unicef"] is False

    monkeypatch.setattr(
        mod.corpus_mod,
        "resolve_shared_unicef_corpus",
        lambda: "projects/p/locations/l/ragCorpora/1",
    )
    assert mod.availability("alice")["unicef"] is True


def test_client_id_sanitization_falls_back_to_default(tmp_path, monkeypatch):
    mod = _reload(tmp_path, monkeypatch)
    mod.set_google_search("../../etc/passwd", False)
    # A malformed id collapses to the same file as the "default" id.
    assert mod.get_prefs("default")["google_search"] is False

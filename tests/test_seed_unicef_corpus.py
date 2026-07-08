"""Tests for the one-time operator seeding script (nanny/seed_unicef_corpus.py).

Runs against the real local corpus backend (no cloud) — the seed script just
calls ``corpus.add_file_to_shared_unicef_corpus``.
"""

import pytest

import nanny.corpus as corpus_mod
import nanny.seed_unicef_corpus as seed_mod


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("NANNY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NANNY_CORPUS_BACKEND", "local")


def test_seed_requires_a_path(capsys):
    assert seed_mod.main([]) == 1
    assert "usage" in capsys.readouterr().err.lower()


def test_seed_refuses_a_missing_file(tmp_path, capsys):
    missing = tmp_path / "nope.pdf"
    assert seed_mod.main([str(missing)]) == 1
    assert "no such file" in capsys.readouterr().err


def test_seed_uploads_into_the_shared_corpus(tmp_path, capsys):
    guide = tmp_path / "guide.txt"
    guide.write_text("responsive parenting supports healthy infant sleep")
    assert seed_mod.main([str(guide)]) == 0
    assert "seeded" in capsys.readouterr().out.lower()

    handle = corpus_mod.resolve_shared_unicef_corpus()
    assert handle is not None
    hits = corpus_mod.retrieve(handle, "infant sleep")
    assert hits and hits[0][1] == "guide.txt"

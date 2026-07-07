"""Per-client source preferences for the InsightsAgent.

Mirrors nanny/stores.py's per-client file resolution: each client id gets its
own JSON file under data/<client_id>.sources.json. The schema mirrors the
design doc:

{
  "google_search": true,
  "unicef": true,
  "uploads": { "my-file.pdf": false }
}

Defaults (when the file or a key is missing):
- google_search: true
- unicef: true
- uploads: absent keys default to true (fresh uploads are usable immediately)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import nanny.corpus as corpus_mod
from nanny.llm import _model_available

# Same per-visitor data directory as nanny/stores.py
_DATA_DIR = Path(os.environ.get("NANNY_DATA_DIR", "./data")).resolve()


def _sources_path(client_id: str) -> Path:
    # Same validation as _CLIENT_ID_RE in server.py
    if (
        not all(c.isalnum() or c in ("-", "_") for c in client_id)
        or len(client_id) > 64
    ):
        client_id = "default"
    return _DATA_DIR / f"{client_id}.sources.json"


def _default_prefs() -> dict[str, Any]:
    return {
        "google_search": True,
        "unicef": True,
        "uploads": {},
    }


def get_prefs(client_id: str) -> dict[str, Any]:
    """Reads the file, filling in defaults for any missing keys."""
    path = _sources_path(client_id)
    if not path.exists():
        return _default_prefs()
    try:
        data = json.loads(path.read_text())
    except Exception:
        return _default_prefs()
    defaults = _default_prefs()
    # Merge: keep any extra keys, fill in missing defaults
    for k, v in defaults.items():
        if k not in data:
            data[k] = v
    return data


def _write_prefs(client_id: str, prefs: dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _sources_path(client_id).write_text(json.dumps(prefs, indent=2))


def set_google_search(client_id: str, enabled: bool) -> dict[str, Any]:
    prefs = get_prefs(client_id)
    prefs["google_search"] = enabled
    _write_prefs(client_id, prefs)
    return prefs


def set_unicef(client_id: str, enabled: bool) -> dict[str, Any]:
    prefs = get_prefs(client_id)
    prefs["unicef"] = enabled
    _write_prefs(client_id, prefs)
    return prefs


def set_upload_enabled(client_id: str, filename: str, enabled: bool) -> dict[str, Any]:
    prefs = get_prefs(client_id)
    prefs.setdefault("uploads", {})[filename] = enabled
    _write_prefs(client_id, prefs)
    return prefs


def availability(client_id: str) -> dict[str, bool]:
    """Which sources are actually configured server-side, independent of prefs."""
    # google_search: ADK's built-in tool piggybacks on the model backend
    # InsightsAgent already needs — no separate credential of its own — so
    # it's available exactly when a real model call can be made at all.
    google_search = _model_available()
    # unicef: RAG enabled AND shared corpus actually seeded
    unicef = (
        corpus_mod.rag_enabled()
        and corpus_mod.resolve_shared_unicef_corpus() is not None
    )
    # uploads: RAG enabled (same as existing refs-panel gating)
    uploads = corpus_mod.rag_enabled()
    return {
        "google_search": google_search,
        "unicef": unicef,
        "uploads": uploads,
    }

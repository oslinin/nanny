"""Per-client ``Store`` resolution, shared by the dashboard (``server.py``)
and the Agent Runtime app (``agent_engine_app.py``).

Deployed on Vertex AI Agent Runtime, ``SaveActivityNode``/``HistoryNode``
(``nanny/workflow.py``) run on the agent side, not the dashboard side — this
module is what they call into, so it has to be importable from both without
either side depending on the other.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .store import Store

DATA_DIR = Path(
    os.environ.get(
        "NANNY_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")
    )
)

_stores: dict[str, Store] = {}
_stores_lock = threading.Lock()


def get_store(client_id: str) -> Store:
    if client_id not in _stores:
        with _stores_lock:
            _stores.setdefault(client_id, Store(str(DATA_DIR / f"{client_id}.jsonl")))
    return _stores[client_id]

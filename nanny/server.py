"""Local HTTP server exposing the dual-mode UI and the ADK workflow runtime.

Step 1 of the PRD ("The UI framework invokes the ADK workflow runtime via a
local API endpoint") is implemented here: two POST endpoints — one for
quick-tap (pre-formatted JSON, bypasses the LLM) and one for chat (raw text,
routed through the ClassifierNode's LLM extraction) — both drive the same
underlying ``google.adk.runners.Runner`` invocation of the workflow graph.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from .activity import KNOWN_ACTIVITY_TYPES, KNOWN_UNITS
from .store import Store
from .workflow import DEFAULT_CLIENT_ID, build_app

logging.basicConfig(level=os.environ.get("NANNY_LOG_LEVEL", "INFO"))
logger = logging.getLogger("nanny.server")

APP_NAME = "nanny_app"

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DATA_DIR = Path(
    os.environ.get(
        "NANNY_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")
    )
)

# All three are opt-in via env var and off by default, so local (same-origin,
# single-user) usage is unaffected. Set them once this server is reachable
# from the public internet (e.g. Cloud Run) rather than only from localhost:
#
# - NANNY_ALLOWED_ORIGINS: comma-separated origins allowed to call the API
#   cross-origin (e.g. a GitHub Pages frontend on a different domain).
# - NANNY_API_TOKEN: if set, POST /api/quick-tap and /api/chat require an
#   `X-Nanny-Token` header matching this value — a guard against random
#   internet traffic, not real per-user access control (every visitor with
#   the token shares API access, though each gets their own session/log via
#   X-Nanny-Client-Id below).
# - NANNY_DB_URL: a SQLAlchemy URL (e.g. a Cloud SQL Postgres instance). When
#   set, ADK session state is stored there via DatabaseSessionService instead
#   of in memory, so it survives a Cloud Run restart. The activity log itself
#   (this module's per-client Store files) is unaffected by this setting —
#   see README's Deployment section.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("NANNY_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
_API_TOKEN = os.environ.get("NANNY_API_TOKEN")
_DB_URL = os.environ.get("NANNY_DB_URL")

# Untrusted client-supplied header value: validated against a strict
# allow-list before it's ever used to build a filesystem path or session id.
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


async def _require_api_token(x_nanny_token: str | None = Header(default=None)) -> None:
    if _API_TOKEN and x_nanny_token != _API_TOKEN:
        raise HTTPException(401, "missing or invalid X-Nanny-Token header")


def _client_id(x_nanny_client_id: str | None = Header(default=None)) -> str:
    """Resolves the per-visitor id used to key both the ADK session and that
    visitor's own activity log.

    Falls back to a fixed id when the header is missing or malformed, rather
    than erroring, so curl/tooling that doesn't send it keeps working exactly
    as the single-user default did before per-client isolation existed.
    """
    if x_nanny_client_id and _CLIENT_ID_RE.match(x_nanny_client_id):
        return x_nanny_client_id
    return DEFAULT_CLIENT_ID


class QuickTapRequest(BaseModel):
    activity_type: str
    quantity: float
    unit: str
    notes: str = ""


class ChatRequest(BaseModel):
    text: str


class TurnResponse(BaseModel):
    ok: bool
    response_text: str
    activity: dict | None = None
    save_result: dict | None = None
    used_llm_extraction: bool | None = None
    used_llm_response: bool | None = None


_stores: dict[str, Store] = {}
_stores_lock = threading.Lock()


def _get_store(client_id: str) -> Store:
    if client_id not in _stores:
        with _stores_lock:
            _stores.setdefault(client_id, Store(str(DATA_DIR / f"{client_id}.jsonl")))
    return _stores[client_id]


def _build_session_service() -> BaseSessionService:
    if not _DB_URL:
        return InMemorySessionService()
    # Imported lazily: requires the optional `db` dependency group
    # (google-adk[db] + a DB driver like pg8000), not needed for local dev.
    from google.adk.sessions import DatabaseSessionService

    return DatabaseSessionService(db_url=_DB_URL)


adk_app = build_app(_get_store)
session_service = _build_session_service()
runner = Runner(app=adk_app, session_service=session_service)

app = FastAPI(title="Nanny")

if _ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Nanny-Token", "X-Nanny-Client-Id"],
    )


async def _ensure_session(client_id: str) -> None:
    existing = await session_service.get_session(
        app_name=APP_NAME, user_id=client_id, session_id=client_id
    )
    if existing is None:
        await session_service.create_session(
            app_name=APP_NAME, user_id=client_id, session_id=client_id, state={}
        )


async def _run_turn(
    client_id: str, state_delta: dict, display_text: str
) -> TurnResponse:
    await _ensure_session(client_id)
    state_delta = {**state_delta, "client_id": client_id}
    async for _ in runner.run_async(
        user_id=client_id,
        session_id=client_id,
        new_message=types.Content(role="user", parts=[types.Part(text=display_text)]),
        state_delta=state_delta,
    ):
        pass

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=client_id, session_id=client_id
    )
    final_state = session.state
    if final_state.get("last_status") != "ok":
        return TurnResponse(
            ok=False,
            response_text=final_state.get("response_text", "Something went wrong."),
        )
    return TurnResponse(
        ok=True,
        response_text=final_state.get("response_text", ""),
        activity=final_state.get("activity"),
        save_result=final_state.get("save_result"),
        used_llm_extraction=final_state.get("used_llm_extraction"),
        used_llm_response=final_state.get("used_llm_response"),
    )


@app.post(
    "/api/quick-tap",
    response_model=TurnResponse,
    dependencies=[Depends(_require_api_token)],
)
async def quick_tap(
    req: QuickTapRequest, client_id: str = Depends(_client_id)
) -> TurnResponse:
    if req.activity_type not in KNOWN_ACTIVITY_TYPES:
        raise HTTPException(
            400, f"unknown activity_type, want one of {list(KNOWN_ACTIVITY_TYPES)}"
        )
    if req.unit not in KNOWN_UNITS:
        raise HTTPException(400, f"unknown unit, want one of {list(KNOWN_UNITS)}")

    now_iso = datetime.now(UTC).astimezone().isoformat()
    payload = {
        "timestamp": now_iso,
        "activity_type": req.activity_type,
        "quantity": req.quantity,
        "unit": req.unit,
        "notes": req.notes,
    }
    display = f"[quick-tap] +{req.quantity:g}{req.unit} {req.activity_type}"
    return await _run_turn(
        client_id,
        {
            "input_mode": "quick_tap",
            "quick_tap_payload": payload,
            "now_iso": now_iso,
        },
        display,
    )


@app.post(
    "/api/chat", response_model=TurnResponse, dependencies=[Depends(_require_api_token)]
)
async def chat(req: ChatRequest, client_id: str = Depends(_client_id)) -> TurnResponse:
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    now_iso = datetime.now(UTC).astimezone().isoformat()
    return await _run_turn(
        client_id,
        {
            "input_mode": "chat",
            "chat_text": req.text,
            "now_iso": now_iso,
        },
        req.text,
    )


@app.get("/api/history")
async def history(client_id: str = Depends(_client_id)) -> list[dict]:
    return [a.to_dict() for a in _get_store(client_id).all()]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

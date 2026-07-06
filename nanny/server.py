"""Local HTTP server / Cloud Run dashboard exposing the dual-mode UI.

Step 1 of the PRD ("The UI framework invokes the ADK workflow runtime via a
local API endpoint") is implemented here: two POST endpoints — one for
quick-tap (pre-formatted JSON, bypasses the LLM) and one for chat (raw text,
routed through ClassifierAgent's LLM extraction) — both drive one turn
through the workflow graph (``nanny/workflow.py``).

Two backends implement that turn, chosen once at import time:

- ``_LocalRunnerBackend`` (default): runs the graph in-process via
  ``google.adk.runners.Runner`` + ``InMemorySessionService``, exactly as
  this app has always worked locally. No GCP credentials needed.
- ``_AgentRuntimeBackend`` (opt-in, ``NANNY_AGENT_ENGINE_RESOURCE_NAME``):
  calls a graph already deployed to Vertex AI Agent Runtime
  (``nanny/agent_engine_app.py``) instead of running it in-process — this
  module then acts purely as the thin, IAM-credentialed dashboard/bridge a
  public frontend can't be trusted to call directly. Requires real GCP
  credentials (the Cloud Run service account's ADC), which is why it can't
  be the default — this sandbox and most local dev environments don't have
  them.

Either way, `/api/quick-tap`, `/api/chat`, and `/api/history` present the
exact same contract to the frontend.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from .activity import KNOWN_ACTIVITY_TYPES, KNOWN_UNITS
from .stores import get_store
from .workflow import DEFAULT_CLIENT_ID, build_app

logging.basicConfig(level=os.environ.get("NANNY_LOG_LEVEL", "INFO"))
logger = logging.getLogger("nanny.server")

APP_NAME = "nanny_app"

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# All opt-in via env var and off by default, so local (same-origin,
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
# - NANNY_AGENT_ENGINE_RESOURCE_NAME: the deployed Vertex AI Agent Runtime
#   resource (from `uv run python -m nanny.agent_engine_app`). When set,
#   this dashboard calls that deployed graph instead of running one
#   in-process — see module docstring.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("NANNY_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
_API_TOKEN = os.environ.get("NANNY_API_TOKEN")
_AGENT_ENGINE_RESOURCE_NAME = os.environ.get("NANNY_AGENT_ENGINE_RESOURCE_NAME")

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


class InsightsRequest(BaseModel):
    # Empty question ⇒ proactive mode: the agent surfaces its own observation.
    question: str = ""


class TurnResponse(BaseModel):
    ok: bool
    response_text: str
    activity: dict | None = None
    save_result: dict | None = None
    used_llm_extraction: bool | None = None
    used_llm_response: bool | None = None


class _Backend(Protocol):
    async def ensure_session(self, client_id: str) -> None: ...
    async def run_turn(
        self, client_id: str, state_delta: dict, display_text: str
    ) -> dict: ...


class _LocalRunnerBackend:
    """Runs the workflow graph in-process. No GCP credentials required."""

    def __init__(self) -> None:
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app=build_app(get_store), session_service=self._session_service
        )

    async def ensure_session(self, client_id: str) -> None:
        existing = await self._session_service.get_session(
            app_name=APP_NAME, user_id=client_id, session_id=client_id
        )
        if existing is None:
            await self._session_service.create_session(
                app_name=APP_NAME, user_id=client_id, session_id=client_id, state={}
            )

    async def run_turn(
        self, client_id: str, state_delta: dict, display_text: str
    ) -> dict:
        async for _ in self._runner.run_async(
            user_id=client_id,
            session_id=client_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=display_text)]
            ),
            state_delta=state_delta,
        ):
            pass
        session = await self._session_service.get_session(
            app_name=APP_NAME, user_id=client_id, session_id=client_id
        )
        return session.state


class _AgentRuntimeBackend:
    """Calls a graph already deployed to Vertex AI Agent Runtime.

    Requires real GCP credentials (the Cloud Run service account's ADC) —
    ``vertexai.agent_engines`` unconditionally resolves a project via
    ``google.auth.default()``, even just to read session state, so this
    cannot be constructed in an environment without them (confirmed while
    building this: it raises ``DefaultCredentialsError`` immediately).
    """

    def __init__(self, resource_name: str) -> None:
        import vertexai
        from vertexai import agent_engines

        vertexai.init(
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1"),
        )
        self._agent = agent_engines.get(resource_name)

    async def ensure_session(self, client_id: str) -> None:
        existing = await self._agent.async_get_session(
            user_id=client_id, session_id=client_id
        )
        if not existing:
            await self._agent.async_create_session(
                user_id=client_id, session_id=client_id, state={}
            )

    async def run_turn(
        self, client_id: str, state_delta: dict, display_text: str
    ) -> dict:
        async for _ in self._agent.async_stream_query(
            message=display_text,
            user_id=client_id,
            session_id=client_id,
            state_delta=state_delta,
        ):
            pass
        session = await self._agent.async_get_session(
            user_id=client_id, session_id=client_id
        )
        return session["state"]


backend: _Backend = (
    _AgentRuntimeBackend(_AGENT_ENGINE_RESOURCE_NAME)
    if _AGENT_ENGINE_RESOURCE_NAME
    else _LocalRunnerBackend()
)

app = FastAPI(title="Nanny")

if _ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Nanny-Token", "X-Nanny-Client-Id"],
    )


async def _query(client_id: str, state_delta: dict, display_text: str) -> dict:
    """Runs one turn through the graph (whichever backend is active) and
    returns the resulting session state."""
    await backend.ensure_session(client_id)
    state_delta = {**state_delta, "client_id": client_id}
    return await backend.run_turn(client_id, state_delta, display_text)


async def _run_turn(
    client_id: str, state_delta: dict, display_text: str
) -> TurnResponse:
    final_state = await _query(client_id, state_delta, display_text)
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
    final_state = await _query(client_id, {"input_mode": "get_history"}, "history")
    return final_state.get("history") or []


@app.post(
    "/api/insights",
    response_model=TurnResponse,
    dependencies=[Depends(_require_api_token)],
)
async def insights(
    req: InsightsRequest, client_id: str = Depends(_client_id)
) -> TurnResponse:
    """Evidence-grounded insights over this client's log.

    An empty question is proactive mode (the agent surfaces its own
    observation); a non-empty one is answered. Rides the same
    ``_run_turn``/backend path as chat, so it works identically whether the
    graph runs in-process or on Agent Runtime.
    """
    now_iso = datetime.now(UTC).astimezone().isoformat()
    display = req.question.strip() or "[insights] what do the patterns say?"
    final_state = await _query(
        client_id,
        {"input_mode": "insights", "question": req.question, "now_iso": now_iso},
        display,
    )
    # Insights logs nothing, so unlike chat/quick-tap it must NOT surface the
    # activity/save_result still sitting in the persistent session state from
    # an earlier turn — only the generated text is about this turn.
    if final_state.get("last_status") != "ok":
        return TurnResponse(
            ok=False,
            response_text=final_state.get("response_text", "Something went wrong."),
        )
    return TurnResponse(
        ok=True,
        response_text=final_state.get("response_text", ""),
        used_llm_response=final_state.get("used_llm_response"),
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

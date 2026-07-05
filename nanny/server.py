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
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from .activity import KNOWN_ACTIVITY_TYPES, KNOWN_UNITS
from .store import Store
from .workflow import build_app

logging.basicConfig(level=os.environ.get("NANNY_LOG_LEVEL", "INFO"))
logger = logging.getLogger("nanny.server")

APP_NAME = "nanny_app"
USER_ID = "parent"
SESSION_ID = "nanny-session"

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DATA_PATH = os.environ.get(
    "NANNY_DATA_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "activity_log.jsonl"),
)


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


store = Store(DATA_PATH)
adk_app = build_app(store)
session_service = InMemorySessionService()
runner = Runner(app=adk_app, session_service=session_service)

app = FastAPI(title="Nanny")


async def _ensure_session():
    existing = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    if existing is None:
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID, state={}
        )


async def _run_turn(state_delta: dict, display_text: str) -> TurnResponse:
    await _ensure_session()
    async for _ in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=types.Content(role="user", parts=[types.Part(text=display_text)]),
        state_delta=state_delta,
    ):
        pass

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
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


@app.post("/api/quick-tap", response_model=TurnResponse)
async def quick_tap(req: QuickTapRequest) -> TurnResponse:
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
        {
            "input_mode": "quick_tap",
            "quick_tap_payload": payload,
            "now_iso": now_iso,
        },
        display,
    )


@app.post("/api/chat", response_model=TurnResponse)
async def chat(req: ChatRequest) -> TurnResponse:
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    now_iso = datetime.now(UTC).astimezone().isoformat()
    return await _run_turn(
        {
            "input_mode": "chat",
            "chat_text": req.text,
            "now_iso": now_iso,
        },
        req.text,
    )


@app.get("/api/history")
async def history() -> list[dict]:
    return [a.to_dict() for a in store.all()]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

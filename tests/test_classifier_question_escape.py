"""Regression coverage for ClassifierAgent's ``is_question`` escape hatch.

Without it, ``_ExtractedActivity``'s schema (activity_type/quantity/unit all
required) leaves the model no way to say "there's no activity here" when the
message is a question — it has to hallucinate something to satisfy the
schema. See the live repro this guards against: chatting "is my baby eating
enough" used to get logged as a fabricated 0oz milk entry.

The real LLM deciding ``is_question`` correctly can only be verified live
(network required, not run in this suite — see the manual repro in the PR/
session notes). What's tested here, deterministically:

1. Every synthetic-JSON producer (sentinel, heuristic response) still
   round-trips through the now-stricter schema — guards against the exact
   failure mode of adding a new required field and breaking the offline
   paths that don't go through the real model at all.
2. classifier_postprocess_node's new branch actually rejects a turn and
   saves nothing when ``is_question`` comes back true — simulated by
   monkeypatching only the shared low-level response helper, not the
   routing logic itself, so this exercises the real node/graph.
"""

import json
from datetime import UTC, datetime

import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import nanny.agents as agents_module
from nanny.agents import _ExtractedActivity, _sentinel_extraction_json
from nanny.store import Store
from nanny.workflow import build_app

APP_NAME = "nanny_app"
USER_ID = "test-user"


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "log.jsonl"))


def test_sentinel_json_validates_against_the_stricter_schema():
    payload = _sentinel_extraction_json("2026-07-06T12:00:00+00:00")
    parsed = _ExtractedActivity.model_validate_json(payload)
    assert parsed.is_question is True


def test_heuristic_success_json_validates_against_the_stricter_schema(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    state = {"chat_text": "he pooped a lot", "now_iso": "2026-07-06T12:00:00+00:00"}
    resp = agents_module._classifier_heuristic_response(state)
    payload = resp.content.parts[0].text
    parsed = _ExtractedActivity.model_validate_json(payload)
    assert parsed.is_question is False
    assert parsed.activity_type == "poop"


@pytest.mark.asyncio
async def test_question_flagged_by_classifier_is_rejected_not_fabricated(
    store, monkeypatch
):
    """Simulates the real model setting is_question=True (as the updated
    instruction asks it to for text like "is my baby eating enough") by
    monkeypatching only the shared text-generation helper — the routing
    decision itself runs for real through classifier_postprocess_node."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fake_heuristic_response(state):
        state["used_llm_extraction"] = False
        return agents_module._text_response(
            json.dumps(
                {
                    "is_question": True,
                    "activity_type": "bottle",
                    "quantity": 0.0,
                    "unit": "oz",
                    "timestamp": state.get("now_iso") or "",
                    "notes": "",
                }
            )
        )

    monkeypatch.setattr(
        agents_module, "_classifier_heuristic_response", fake_heuristic_response
    )

    now_iso = datetime.now(UTC).isoformat()
    text = "is my baby eating enough"
    adk_app = build_app(lambda _client_id: store)
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="s-question", state={}
    )
    runner = Runner(app=adk_app, session_service=session_service)
    async for _ in runner.run_async(
        user_id=USER_ID,
        session_id="s-question",
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
        state_delta={"input_mode": "chat", "chat_text": text, "now_iso": now_iso},
    ):
        pass
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="s-question"
    )
    state = session.state

    assert state["last_status"] == "error"
    assert "question" in state["error"]
    assert len(store.all()) == 0

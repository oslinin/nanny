"""Trace generator for nanny's local eval harness.

Runs each case in ``tests/eval/datasets/basic-dataset.json`` through the real
in-process ADK graph (``nanny.workflow.build_app``), the same
``Runner`` + ``InMemorySessionService`` pattern ``tests/test_workflow.py``
uses, and writes populated traces to ``artifacts/traces/`` in the
``agents-cli eval grade``-compatible ``EvaluationDataset`` shape
(``agent_data.turns`` per case).

A generic ``agents-cli eval generate`` can't do this run itself: it expects
an ``agent_directory`` pointing at a conventional single ``root_agent``
module (see ``agents-cli-manifest.yaml``), whereas nanny's graph is built by
``build_app(store_resolver)`` and driven through a custom ``input_mode``
state delta (``quick_tap`` / ``chat`` / ``insights``) rather than a bare
chat prompt.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanny.store import Store
from nanny.workflow import build_app

APP_NAME = "nanny_app"
USER_ID = "eval-runner"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "tests" / "eval" / "datasets" / "basic-dataset.json"
TRACES_DIR = REPO_ROOT / "artifacts" / "traces"

# Final-state fields worth showing a judge -- this is where nanny's actual
# containment decisions live, since most of the graph's nodes are
# deterministic and emit no ADK event content of their own (see
# `_events_to_trace` below).
STATE_FIELDS_FOR_JUDGE = (
    "last_status",
    "ingestion_branch",
    "security_blocked",
    "error",
    "used_llm_extraction",
    "used_llm_response",
    "extracted_activity",
    "activity",
    "save_result",
    "response_text",
)


def _prompt_text(case: dict) -> str:
    parts = case.get("prompt", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _state_delta_for(case: dict, now_iso: str) -> dict:
    mode = case.get("nanny_input_mode", "chat")
    text = _prompt_text(case)
    if mode == "quick_tap":
        payload = dict(case.get("nanny_quick_tap_payload") or {})
        payload.setdefault("timestamp", now_iso)
        return {
            "input_mode": "quick_tap",
            "quick_tap_payload": payload,
            "now_iso": now_iso,
        }
    if mode == "insights":
        return {"input_mode": "insights", "question": text, "now_iso": now_iso}
    return {"input_mode": "chat", "chat_text": text, "now_iso": now_iso}


def _content_to_dict(content) -> dict | None:
    if content is None:
        return None
    parts = [
        {"text": p.text} for p in (content.parts or []) if getattr(p, "text", None)
    ]
    if not parts:
        return None
    return {"role": content.role or "model", "parts": parts}


def _events_to_trace(raw_events, prompt_content: dict, final_state: dict) -> list[dict]:
    """Builds the `agent_data.turns[0].events` list for one case.

    Nanny's deterministic FunctionNodes (IngestNode, RouterNode,
    SaveActivityNode, ...) emit ADK events with no content -- the graph's
    actual containment decision (blocked vs saved, what got extracted, what
    the parent was told) lives in final session state instead. So the trace
    is: the user prompt, then any real model-authored events (e.g.
    ClassifierAgent's structured extraction), then one synthetic
    `nanny_final_state` event carrying the state snapshot -- this is what
    lets an LLM judge "read the whole trace" meaningfully.
    """
    events = [{"author": "user", "content": prompt_content}]
    for event in raw_events:
        content = _content_to_dict(event.content)
        if content is None:
            continue
        events.append({"author": event.author, "content": content})
    state_snapshot = {
        k: final_state.get(k) for k in STATE_FIELDS_FOR_JUDGE if k in final_state
    }
    events.append(
        {
            "author": "nanny_final_state",
            "content": {
                "role": "model",
                "parts": [{"text": json.dumps(state_snapshot, default=str)}],
            },
        }
    )
    return events


async def _run_case(case: dict) -> dict:
    now_iso = datetime.now(UTC).isoformat()
    state_delta = _state_delta_for(case, now_iso)
    prompt_content = case["prompt"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = Store(str(Path(tmp_dir) / "log.jsonl"))
        adk_app = build_app(lambda _client_id: store)
        session_service = InMemorySessionService()
        session_id = case.get("eval_case_id", "case")
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={}
        )
        runner = Runner(app=adk_app, session_service=session_service)
        raw_events = []
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=_prompt_text(case))]
            ),
            state_delta=state_delta,
        ):
            raw_events.append(event)
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        final_state = dict(session.state)

    response_text = final_state.get("response_text", "")
    response_content = {"role": "model", "parts": [{"text": response_text}]}
    trace_events = _events_to_trace(raw_events, prompt_content, final_state)

    out_case = {k: v for k, v in case.items() if k not in ("nanny_quick_tap_payload",)}
    out_case["response"] = response_content
    out_case["responses"] = [{"response": response_content}]
    out_case["nanny_final_state"] = {
        k: final_state.get(k) for k in STATE_FIELDS_FOR_JUDGE if k in final_state
    }
    out_case["agent_data"] = {"turns": [{"turn_index": 0, "events": trace_events}]}
    return out_case


async def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    cases = dataset["eval_cases"]

    results = []
    for case in cases:
        print(
            f"  generating trace: {case['eval_case_id']} ({case.get('category', '?')})"
        )
        results.append(await _run_case(case))

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = TRACES_DIR / f"trace_{timestamp}.json"
    out_path.write_text(json.dumps({"eval_cases": results}, indent=2, default=str))
    print(f"wrote {len(results)} traces to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

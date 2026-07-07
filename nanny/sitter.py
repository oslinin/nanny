"""The SitterAgent: a proactive schedule concierge for the human sitter.

A fifth real ``google.adk.agents.LlmAgent`` (alongside the Classifier,
Responder, and Insights agents) with two jobs, chosen by ``sitter_action`` in
the turn state:

- ``set_schedule`` — the parent typed a message starting with "Instructions:".
  The agent parses that free-form daily schedule into a list of timed
  reminders (``[{"time": "HH:MM", "text": ...}]``) that a deterministic node
  (``sitter_save_node`` in ``nanny/workflow.py``) then persists via
  ``nanny/schedule.py``. Those reminders are what the frontend surfaces to the
  sitter in blue through the day (every 20 minutes) and what the "nanny"
  prompt reads back.
- ``next`` — the sitter (or the parent) typed "nanny". The agent reads the
  already-stored reminders and phrases the next thing to do as one warm line;
  it writes no schedule.

Like the other agents this degrades gracefully: with no model backend
configured it runs the deterministic ``nanny/schedule.py`` parser/lookup
instead of calling Gemini (``_sitter_offline_fallback_callback``), and if a
live model call fails at runtime it falls back to the same path. The parent's
schedule text is screened for prompt injection / secrets first, exactly like
chat input to the ClassifierAgent.
"""

from __future__ import annotations

import json
import logging
import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel, Field

from . import schedule as schedule_mod
from .llm import _model_available
from .security import screen_text

logger = logging.getLogger("nanny.sitter")

_MODEL_NAME = os.environ.get("NANNY_GEMINI_MODEL", "gemini-flash-latest")


class _Reminder(BaseModel):
    time: str = Field(
        description='24-hour clock time as "HH:MM" (e.g. "09:00", "13:30").'
    )
    text: str = Field(description="What the sitter should do at that time.")


class _SitterResponse(BaseModel):
    """Constrained output for the SitterAgent.

    ``reminders`` is filled only for a ``set_schedule`` turn (the parsed
    schedule); for a ``next`` turn it is left empty. ``message`` is always the
    one human-facing sentence shown in the chat.
    """

    reminders: list[_Reminder] = Field(
        default_factory=list,
        description=(
            "The parsed timed reminders when setting a schedule; leave empty "
            "when only reporting the next instruction."
        ),
    )
    message: str = Field(
        description="One short, warm sentence to show in the chat window."
    )


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def _sitter_response_payload(state) -> dict:
    """The deterministic SitterResponse — offline fallback and model-error path.

    Mirrors the live agent's contract: parse on ``set_schedule``, look up the
    next instruction on ``next``.
    """
    action = state.get("sitter_action") or "next"
    if action == "set_schedule":
        reminders = schedule_mod.parse_schedule(state.get("schedule_text") or "")
        message = (
            f"Got it — I'll remind the sitter of {len(reminders)} thing(s) "
            "through the day."
        )
        return {"reminders": reminders, "message": message}
    reminders = []
    try:
        reminders = json.loads(state.get("sitter_reminders_json") or "[]")
    except (ValueError, TypeError):
        reminders = []
    now_hhmm = schedule_mod.hhmm_from_iso(state.get("now_iso") or "")
    upcoming = schedule_mod.next_reminder(reminders, now_hhmm)
    return {"reminders": [], "message": schedule_mod.format_reminder(upcoming)}


def _sitter_security_callback(callback_context, llm_request):
    """Screens the parent's schedule text before the model call (same guard the
    ClassifierAgent applies to chat input). A "next" turn carries no
    user-supplied text, so this is a no-op for it."""
    state = callback_context.state
    text = state.get("schedule_text") or ""
    if not text.strip():
        return None
    reason = screen_text(text)
    if reason is None:
        return None
    logger.warning("SitterAgent: blocked schedule text: %s", reason)
    state["security_blocked"] = True
    state["error"] = reason
    state["used_llm_response"] = False
    # Schema-valid sentinel; sitter_save_node routes to the error branch on the
    # security_blocked flag before this content is ever used.
    return _text_response(json.dumps({"reminders": [], "message": ""}))


def _sitter_offline_fallback_callback(callback_context, llm_request):
    """Runs the deterministic schedule parser / lookup instead of calling Gemini
    when no model backend (AI-Studio key or Vertex) is configured."""
    if _model_available():
        return None
    state = callback_context.state
    state["used_llm_response"] = False
    return _text_response(json.dumps(_sitter_response_payload(state)))


def _sitter_model_error_callback(*, callback_context, llm_request, error):
    """Degrades to the deterministic path when a configured model call fails at
    runtime (invalid key, quota exhausted, timeout) instead of aborting."""
    logger.warning(
        "SitterAgent: model call failed (%s); falling back to deterministic path",
        error,
    )
    state = callback_context.state
    state["used_llm_response"] = False
    return _text_response(json.dumps(_sitter_response_payload(state)))


_SITTER_INSTRUCTION = """\
You are Nanny's Sitter Agent. You help a parent keep a daily care schedule for
their baby and proactively remind the human sitter what to do.

Action: {sitter_action}
Current time (ISO-8601): {now_iso}

If the action is "set_schedule", parse the schedule below into a list of timed
reminders. Use 24-hour "HH:MM" times, honoring "AM"/"PM" section headers
(so "1" under PM is "13:00"). Keep each reminder's text faithful to what the
parent wrote. Then write one short confirmation sentence in `message`.
Schedule to parse:
{schedule_text}

If the action is "next", DO NOT invent a schedule — leave `reminders` empty and
write in `message` a single warm sentence telling the sitter the next thing to
do and at what time: the earliest stored reminder at or after the current time.
If none remain today, say the day's scheduled care is done. Stored reminders
(JSON): {sitter_reminders_json}
"""


def build_sitter_agent() -> LlmAgent:
    """The proactive schedule agent (the "Instructions:" and "nanny" paths)."""
    return LlmAgent(
        name="sitter_agent",
        model=_MODEL_NAME,
        mode="single_turn",
        instruction=_SITTER_INSTRUCTION,
        output_schema=_SitterResponse,
        output_key="sitter_response",
        # Security guard runs first; only if it doesn't block does the offline
        # fallback get a chance to short-circuit the real call.
        before_model_callback=[
            _sitter_security_callback,
            _sitter_offline_fallback_callback,
        ],
        on_model_error_callback=_sitter_model_error_callback,
    )
